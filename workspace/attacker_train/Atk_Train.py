import torch
import torch.nn.functional as F
import torch.optim as optim
import copy


class AtkLeaning():
    def __init__(self, target_data, test_loader, student_net, lr, amsgrad):
        self.target_data = target_data
        self.test_data = test_loader
        self.student_net = student_net
        self.lr = lr
        self.optimizer =  optim.Adam(student_net.parameters(), lr = self.lr, weight_decay=0.0)
        # self.optimizer =  optim.Adam(student_net.parameters(), lr = lr, amsgrad = amsgrad)
        self.loss_count = []
        self.acc_count = []
        self.best_acc = 0.0
    
    def attacker_training(self, dataset):
        query_data = dataset[0].to('cuda')
        target_pred = dataset[1].to('cuda')

        #training student_model
        self.optimizer.zero_grad()
        attacker_pred = self.student_net(query_data)   

        loss = F.kl_div(F.log_softmax(attacker_pred, dim=1), target_pred, reduction='batchmean')
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def getting_acc(self):
        self.student_net.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i, bacth in enumerate(self.test_data):
                inputs, labels = bacth
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                if self.target_data == "MedMNIST":
                    labels = labels.squeeze().long()
                
                outputs = self.student_net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = (correct_predictions / total_predictions) * 100
        self.acc_count.append(accuracy)
        if accuracy > self.best_acc:
            self.update_model(accuracy)
        return accuracy
    
    def update_model(self, accuracy):
        self.best_model = copy.deepcopy(self.student_net)
        self.best_acc = accuracy
    
    # def load_best_model(self):
    #     self.student_net = self.best_model
    #     self.optimizer = optim.Adam(self.student_net.parameters(), lr = self.lr, weight_decay=0.0)

    def load_best_model(self):
        self.student_net.load_state_dict(self.best_model.state_dict())


    