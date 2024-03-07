import medmnist
from medmnist import INFO
import Data.Flactal_Datasets as Flactal_Datasets
import Data.dataset as dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

# Dataset import
class DataLoad(): 
    def __init__(self, target_data, batch_size, data_flag = None, Fractal = None, FracName = None, edit_data = None):
        self.target_data = target_data
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.Fractal = Fractal
        self.FracName = FracName
        self.edit_data = edit_data

    def import_data(self):
        # Getting dataset
        train_loader,test_loader = dataset.create_loaders(self.target_data, self.batch_size, self.data_flag)

        #getting data_param for MedMNIST
        if self.target_data == "MedMNIST":
            info = INFO[self.data_flag]
            self.n_channels = info['n_channels']
            self.n_classes = len(info['label'])

        #switching Fractaldata
        if self.Fractal == True:
            if len(self.edit_data) == 2:
                train_loader = Flactal_Datasets.edit_data(self.FracName, self.batch_size, self.edit_data[0], self.edit_data[1])
            else:
                train_loader = Flactal_Datasets.getting_data(self.FracName, self.batch_size)
        
        return train_loader, test_loader
    
    def second_import(self, second_data, data_flag = None):

        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST)
                ]
            )
        
        if second_data == "MNIST":
            train_dataset = datasets.MNIST(root='/workspace/Data/MNIST_dataset/', train=True, download=True, transform=transform)
            train_load = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        elif second_data == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(root='/workspace/Data/FashionMNIST_dataset/', train=True, download=True, transform=transform)
            train_load = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        elif second_data == "MedMNIST":
            # selecting MedMNIST_Tag
            print(data_flag)
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            train_dataset = DataClass(split='train', download=True, transform=transform)  # Assuming that DataClass can be treated as a dataset
            train_load = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # Creating a DataLoader from the dataset

        return train_load