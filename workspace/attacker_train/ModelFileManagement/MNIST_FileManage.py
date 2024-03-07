import sys
sys.path.append('/workspace/network/')

import torch
import networks_attacker
import networks_target
from .Base_FileManage import FileManagement

class MNISTFileManagement(FileManagement):
    def __init__(self, file_name, model_size):
        super().__init__(file_name, model_size)
        self.target_dir = self.cloud_model_dir / "MNIST"
        self.save_dir = self.checkpoints_student_dir / "MNIST"
        self.classes = 10

    def load_targetmodel(self):
        load_path = self.target_dir / self.file_name
        # target_net = networks_target.TargetNetwork_MNIST()
        target_net = networks_target.AttackerNetworkSmall_MNIST(self.classes)
        target_net.load_state_dict(torch.load(str(load_path))['model_state_dict'])
        target_net = target_net.to('cuda')
        return target_net, str(load_path)

    def get_studentmodel(self):
        if self.model_size == "L":
            student_net = networks_attacker.AttackerNetworks_MNIST_L(self.classes)
        elif self.model_size == "M":
            student_net = networks_attacker.AttackerNetworks_MNIST_M(self.classes)
        elif self.model_size == "S":
            student_net = networks_attacker.AttackerNetworks_MNIST_S(self.classes)
        student_net = student_net.to('cuda')
        return student_net


    def set_save_substitute_model(self, num_queries, accuracy, student_net, batch_size, substitute_model_epochs, substitute_model_batch_size, lr, Fracdata, Fractal = None):
        self.access_times = num_queries
        self.accuracy = accuracy
        self.student_net = student_net.to('cpu')
        self.batch_size = batch_size
        self.substitute_model_epochs = substitute_model_epochs
        self.substitute_model_batch_size = substitute_model_batch_size
        self.lr = lr
        self.Fracdata = Fracdata
        self.Fractal = Fractal
        # Save traning_result
        self.save_substitute_model()

    def save_substitute_model(self):

        if self.Fractal == True:
            save_name = f"size_{self.model_size}_query_{self.access_times}_{self.Fracdata}_MNIST"
             # Load the image
            # save_dir = Path(self.save_dir)
            load_path = self.save_dir / self.Fracdata
            save_path = load_path / save_name

        elif self.Fractal == False:
            save_name = f"size_{self.model_size}_query_{self.access_times}_Batch_{self.batch_size}_MNIST"
            save_path = self.save_dir / save_name

        torch.save({
            'results': self.accuracy,
            'model_state_dict': self.student_net.state_dict(),
            'query': self.access_times,
            'query_batches':self.batch_size,
            'train_epoch': self.substitute_model_epochs,
            'attack_batches': self.substitute_model_batch_size,
            'lr': self.lr
        }, str(save_path))