import os
import sys
sys.path.append('/workspace')

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

transform = transforms.Compose([
    transforms.Grayscale(),
    # transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

def limited_FlacData(FracName, num_classes,num_samples):
    # Load the image
    load_path = os.path.join("/workspace/Data/Fractal/", FracName)
    dataset = datasets.ImageFolder(load_path, transform=transform)
    # ランダムにクラスを選択
    selected_classes = random.sample(range(len(dataset.classes)), num_classes)
    
    # 各クラスからランダムに画像を選択
    selected_indices = []
    
    for c in selected_classes:
        class_indices = [i for i, label in enumerate(dataset.targets) if label == c]
        selected_indices.extend(random.sample(class_indices, num_samples))#ここでインデックス
        
    # Subsetを作成して新しいデータセットを作成
    new_dataset = Subset(dataset, selected_indices)
    
    return new_dataset


def getting_data(FracName, batch_size):
     #using original data
    load_path = os.path.join("/workspace/Data/Fractal/", FracName)
    dataset = datasets.ImageFolder(load_path, transform=transform)
    # DataLoaderの作成
    Fractaldata = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    return Fractaldata

def edit_data(FracName, batch_size, num_classes, num_samples):
    dataset = limited_FlacData(FracName, num_classes,num_samples)
    # DataLoaderの作成
    Fractaldata = DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    return Fractaldata
