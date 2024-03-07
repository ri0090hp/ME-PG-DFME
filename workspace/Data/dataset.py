import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import medmnist
from medmnist import INFO
import sys
sys.path.append('/workspace')

#download experimental image
def download_image(image_data_name, data_flag = None): 
    if image_data_name == "MNIST":

        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST)
                ]
            )
        train_dataset = datasets.MNIST(root='/workspace/Data/MNIST_dataset/', train=True, 
                                            download=True, transform=transform)
        test_dataset = datasets.MNIST(root='/workspace/Data/MNIST_dataset/', train=False, 
                                            download=False, transform=transform)

    elif image_data_name == "FashionMNIST":

        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST)
                ]
            )
        train_dataset = datasets.FashionMNIST(root='/workspace/Data/FashionMNIST_dataset/', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='/workspace/Data//FashionMNIST_dataset/', train=False, download=True, transform=transform)
    
    elif image_data_name == "MedMNIST": 

        #selecting MedMNIST_Tag
        info = INFO[data_flag]

        DataClass = getattr(medmnist, info['python_class'])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST)
            ])
        # load the data
        train_dataset = DataClass(split='train', download=True, transform=transform)
        test_dataset = DataClass(split='val', download=True, transform=transform)

    else:
        raise ValueError("Invalid dataset name.")

    return train_dataset, test_dataset


def create_loaders(image_data_name, batch_size, data_flag = None, attack = 0, attack_batch = 0): #change
    #データセット呼び出し
    train_val_dataset, test_dataset = download_image(image_data_name, data_flag) #change

    if attack == 0:
        # targetデータローダーの作成
        train_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    elif attack == 1:
        # attackデータローダーの作成
        train_loader = DataLoader(train_val_dataset, batch_size = attack_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    print(train_val_dataset)
    print("===================")
    print(test_dataset)
    return train_loader, test_loader

