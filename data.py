import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10,FashionMNIST
from sampling import create_iid_data, create_non_iid_data,create_dir_non_iid_data,create_path_non_iid_data
import logging
log = logging.getLogger(__name__)

def load_datasets(dataset: str, num_clients: int, batch_size: int, split: str, class_per_client: int, alpha: float, seed: int):
    
    if dataset == "CIFAR10":
        transform = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.CenterCrop((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        )
        trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))
    elif dataset == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.Resize((32,32)),
                transforms.CenterCrop((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), 
                    (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
        testset = CIFAR100("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))
    elif dataset == "STL10":
        transform = transforms.Compose(
        [
            transforms.Resize((96,96)),
            transforms.CenterCrop((96,96)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4467, 0.4398, 0.4066), 
                (0.2241, 0.2215, 0.2239)
            ),
        ]
        )
        trainset = STL10("./dataset", split = "train", download=True, transform=transform)
        testset = STL10("./dataset", split = "test", download=True, transform=transform)
        num_classes = len(set(trainset.labels))
    
    elif dataset == "F-MNIST":
        transform = transforms.Compose(
        [
            transforms.ToTensor()]
        )
        trainset = FashionMNIST("./dataset", train=True, download=True, transform=transform)
        testset = FashionMNIST("./dataset", train=False, download=True, transform=transform)
        num_classes = len(set(trainset.targets))

    else:
        print("THE DATASET IS NOT ADDED YET. EXITING NOW")
        import sys
        sys.exit()
    

    if split == "IID":
        trainloaders,valloaders, client_samples = create_iid_data(trainset,num_clients,batch_size,seed)
    elif split == "N-IID":
        trainloaders,valloaders, client_samples = create_non_iid_data(trainset,num_classes,num_clients,class_per_client,batch_size,seed)
    elif split == "DIR-N-IID":
        trainloaders,valloaders, client_samples = create_dir_non_iid_data(trainset,alpha,num_classes,num_clients,batch_size,seed)
    elif split == "P-N-IID":
        trainloaders,valloaders, client_samples = create_path_non_iid_data(trainset,num_classes, classes_per_client, num_clients, batch_size, seed)

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader,client_samples,num_classes



