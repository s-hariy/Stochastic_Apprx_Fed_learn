import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import dirichlet

def create_dataloaders(datasets,batch_size,seed):
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    client_samples = []
    for i in range(len(datasets)):
        ds = datasets[i]
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        client_samples.append(len(ds_train))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders,valloaders, client_samples

def create_iid_data(trainset,num_clients,batch_size,seed):
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    trainloaders, valloaders, client_samples = create_dataloaders(datasets,batch_size,seed)
    return trainloaders,valloaders, client_samples

def create_non_iid_data(trainset,num_classes,num_clients,class_per_client,batch_size,seed):
    clients_data = {}
    class_indices = {i:np.where(np.array(trainset.targets)==i)[0] for i in range (num_classes)}
    
    for i in range(num_clients):
        client_classes = [(i+j)%10 for j in range(class_per_client)]
        client_indices = np.concatenate([class_indices[cls] for cls in client_classes])
        clients_data[i] = Subset(trainset, client_indices)

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples


def dirichlet_allocation(trainset,alpha,num_classes,num_clients,batch_size):
    """
    Allocate indices of trainset among clients based on Dirichlet distribution.
    :param labels: Array of trainset labels
    :param num_clients: Number of clients
    :param alpha: Concentration parameter for Dirichlet distribution
    :return: List of indices for each client
    """
    try:
        labels = np.array(trainset.targets)
    except:
        labels = np.array(trainset.labels)
    # Dirichlet distribution
    distribution = dirichlet.rvs([alpha] * num_clients, size=num_classes)

    # Indices allocation for each client
    client_indices = {i: np.array([], dtype='int') for i in range(num_clients)}
    for k in range(num_classes):
        # Indices of class k
        class_k_indices = np.where(labels == k)[0]

        # Multinomial distribution to split indices among clients
        indices_split = np.random.multinomial(len(class_k_indices), distribution[k])

        # Distribute indices of class k to clients
        start = 0
        for i in range(num_clients):
            end = start + indices_split[i]
            client_indices[i] = np.concatenate((client_indices[i], class_k_indices[start:end]))
            start = end
    return client_indices

def create_dir_non_iid_data(trainset,alpha,num_classes,num_clients,batch_size,seed):
    # Apply Dirichlet allocation
    client_data_indices = dirichlet_allocation(trainset,alpha,num_classes,num_clients,batch_size)
    clients_data = {i: Subset(trainset, client_data_indices[i]) for i in range(num_clients)}
        
    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples

def create_path_non_iid_data(trainset,num_classes, classes_per_client, num_clients, batch_size, seed):
    clients_data = {}
    class_indices = {i: np.where(np.array(trainset.targets) == i)[0] for i in range(num_classes)}

    # Shuffling indices within each class for randomness
    for indices in class_indices.values():
        np.random.shuffle(indices)

    for i in range(num_clients):
        # Randomly determine the number of classes for this client
        num_classes_for_client = np.random.randint(1, classes_per_client + 1)
        client_classes = np.random.choice(num_classes, num_classes_for_client, replace=False)

        client_indices = []
        for cls in client_classes:
            # Variable number of samples per class per client
            num_samples = np.random.randint(1, len(class_indices[cls]) // (num_clients // num_classes_for_client) + 1)
            client_indices.extend(class_indices[cls][:num_samples])
            # Optional: Remove the selected indices to avoid duplication
            class_indices[cls] = class_indices[cls][num_samples:]

        clients_data[i] = Subset(trainset, client_indices)

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples

