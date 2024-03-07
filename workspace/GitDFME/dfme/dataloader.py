from torchvision import datasets, transforms
import torch
import medmnist
from medmnist import INFO

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.NEAREST)
    ])

    if args.dataset == 'MNIST':
        print(" \nLoading MNIST data \n")
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='/workspace/Data/MNIST_dataset/', train=True, download=True,
                           transform=transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='/workspace/Data/MNIST_dataset/', train=False, download=True,
                           transform=transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        classes = 10

    elif args.dataset == 'FashionMNIST':
        print(" \nLoading FashionMNIST data \n")
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root='/workspace/Data/FashionMNIST_dataset/', train=True, download=True,
                                  transform=transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(root='/workspace/Data/FashionMNIST_dataset/', train=False, download=True,
                                  transform=transform),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        classes = 10

    elif args.dataset == 'MedMNIST':
        print(" \nLoading MedMNIST data : ", args.medflag,"\n")

        info = INFO[args.medflag]
        DataClass = getattr(medmnist, info['python_class'])
        # load the data
        train_dataset = DataClass(split='train', download=True, transform=transform)
        test_dataset = DataClass(split='val', download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        classes = len(info['label'])

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_loader, test_loader, classes
