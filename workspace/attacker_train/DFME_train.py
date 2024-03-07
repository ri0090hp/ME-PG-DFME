import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class TrainerDFME():
    def __init__(self, target_net, student_net, generator, Gan_batch, test_loader, target_data):
        self.target_net = target_net
        self.student_net = student_net
        self.generator = generator

        self.optimizer_G = optim.SGD( self.generator.parameters(), lr=1e-4  , weight_decay=5e-4, momentum=0.9 )
        self.optimizer_S = optim.SGD( self.student_net.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9 )
        self.Gan_batch = Gan_batch

        self.test_loader = test_loader
        self.target_data = target_data
        self.acc_count = []
        self.best_acc = 0.0

    def estimate_Gan_loss(self, x, epsilon = 1e-7, m = 5, verb=False, num_classes=10, pre_x=True):

        self.target_net.eval()
        self.student_net.eval()

        with torch.no_grad():
            # Sample unit noise vector
            x = x.to('cuda')
            N = x.size(0)
            C = x.size(1)
            S = x.size(2)
            dim = S**2 * C

            u = np.random.randn(N * m * dim).reshape(-1, m, dim) # generate random points from normal distribution

            d = np.sqrt(np.sum(u ** 2, axis = 2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
            u = torch.Tensor(u / d).view(-1, m, C, S, S)
            u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim = 1) # Shape N, m + 1, S^2

            u = (u.view(-1, m + 1, C, S, S)).to('cuda')

            evaluation_points = (x.view(-1, 1, C, S, S) + epsilon * u).view(-1, C, S, S)
            if pre_x:
                evaluation_points = torch.tanh(evaluation_points) # Apply args.G_activation function

            # Compute the approximation sequentially to allow large values of m
            pred_victim = []
            pred_clone = []
            max_number_points = 32*156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

            for i in (range(N * m // max_number_points + 1)):
                pts = evaluation_points[i * max_number_points: (i+1) * max_number_points]
                pts = pts.to('cuda')

                pred_victim_pts = self.target_net(pts).detach()
                pred_clone_pts = self.student_net(pts)

                pred_victim.append(pred_victim_pts)
                pred_clone.append(pred_clone_pts)

            pred_victim = torch.cat(pred_victim, dim=0)#.to('cuda')GPU
            pred_clone = torch.cat(pred_clone, dim=0)#.to('cuda')


            loss_fn = F.l1_loss
            pred_victim = F.log_softmax(pred_victim, dim=1).detach()
            logit_correction = "mean"

            if logit_correction == 'min':
                pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
            elif logit_correction == 'mean':
                pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()
            else:
                raise ValueError()

            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim = 1).view(-1, m + 1)

            # Compute difference following each direction
            differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
            differences = differences.view(-1, m, 1, 1, 1)


            # Formula for Forward Finite Differences
            gradient_estimates = 1 / epsilon * differences * u[:, :-1]
            gradient_estimates *= dim

            gradient_estimates = gradient_estimates.mean(dim = 1).view(-1, C, S, S) / (num_classes * N)

            self.student_net.train()
            loss_G = loss_values[:, -1].mean()

            return gradient_estimates.detach(), loss_G
        
    def estimate_student_loss(self, s_logit, t_logit, return_t_logits=False):
        """Kl/ L1 Loss for student"""
        print_logits =  False
        losss = "l1"
        if losss == "l1":
            loss_fn = F.l1_loss
            loss = loss_fn(s_logit, t_logit.detach())
        elif losss == "kl":
            loss_fn = F.kl_div
            s_logit = F.log_softmax(s_logit, dim=1)
            t_logit = F.softmax(t_logit, dim=1)
            loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
        else:
            raise ValueError()

        return loss
    
    def student_train(self, E_epoch, epoch):
        for E in range(E_epoch):
            z = torch.randn((self.Gan_batch, 256))
            fake = self.generator(z).detach()
            fake = fake.to('cuda')
            self.optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = self.target_net(fake)

            # Correction for the fake logits
            t_logit = F.log_softmax(t_logit, dim=1).detach()
            t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            self.student_net.train()
            s_logit = self.student_net(fake)

            loss_S = self.estimate_student_loss(s_logit, t_logit)
            loss_S.backward()
            self.optimizer_S.step()


    def Gan_train(self, G_epoch):
        for _ in range(G_epoch):
            #Sample Random 
            z = torch.randn((self.Gan_batch, 256))
            self.optimizer_G.zero_grad()
            self.generator.train()
            #Get fake image from generator
            fake = self.generator(z, pre_x=1) # pre_x returns the output of G before applying the activation
            fake = fake.to('cuda')
            ## APPOX GRADIENT
            approx_grad_wrt_x, loss_G = approx_grad_wrt_x, loss_G = self.estimate_Gan_loss(fake)

            fake.backward(approx_grad_wrt_x)
            self.optimizer_G.step()

    def getting_acc(self, query_times):
        self.student_net.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                if self.target_data == "MedMNIST":
                    labels = labels.squeeze().long()

                outputs = self.student_net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        substitute_mid_accuracy = (correct_predictions / total_predictions) * 100
        self.acc_count.append(substitute_mid_accuracy)

        print(query_times, " : Extract_Model Accuracy =", substitute_mid_accuracy)
        if substitute_mid_accuracy > self.best_acc:
            self.update_model(substitute_mid_accuracy)
            print("アップデート")

        return substitute_mid_accuracy
    
    def update_model(self, accuracy):
        self.best_model = copy.deepcopy(self.student_net)
        self.best_acc = accuracy
    
    # def load_best_model(self):
    #     self.student_net = self.best_model
    #     self.optimizer = optim.Adam(self.student_net.parameters(), lr = self.lr, weight_decay=0.0)

    def load_best_model(self):
        self.student_net.load_state_dict(self.best_model.state_dict())
        self.optimizer_S = optim.SGD( self.student_net.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9 )
        







        
