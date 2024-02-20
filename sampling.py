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



def CIFAR10_SuperClass_NIID_DIR(train_ds,alpha,num_classes,num_clients,batch_size,seed):

    labels_train = np.array(train_ds.targets)
    y_train = np.array(train_ds.targets)

    superclass = [[0,1,8,9], [2,3,4,5,6,7]]
    nclass=num_classes
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(num_clients)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(num_clients)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0

    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)

    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*num_clients)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=s-num_clients, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=num_clients-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==num_clients

    for r, clust in enumerate(superclass):
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*5000)

        min_size = 0
        min_require_size = 15

        print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])

        #### Assigning samples to each client
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])

            dataidx = net_dataidx_map[j]
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp

        cnt+=n_parties

    clients_data = {}
    for i in range(num_clients):

        clients_data[i] = Subset(train_ds, net_dataidx_map[i])

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples


def CIFAR100_SuperClass_NIID_DIR(train_ds,alpha,num_classes,num_clients,batch_size,seed):
    labels_train = np.array(train_ds.targets)
    y_train = np.array(train_ds.targets)

    superclass = [[0, 83, 53, 82], [1, 54, 43, 51, 70, 92, 62],
                [2, 97, 27, 65, 64, 36, 28, 61, 99, 18, 77, 79, 80, 34, 88, 42, 38, 44,
                63, 50, 78, 66, 84, 8, 39, 55, 72, 93, 91, 3, 4, 29, 31, 7, 24, 20,
                26, 45, 74, 5, 25, 15, 19, 32, 9, 16, 10, 22, 40, 11, 35, 98, 46, 6,
                14, 57, 94, 56, 13, 58, 37, 81, 90, 89, 85, 21, 48, 86, 87, 41, 75,
                12, 71, 49, 17, 60, 76, 33, 68],
                [23, 69, 30, 95, 67, 73], [47, 96, 59, 52]]

    nclass=num_classes
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(num_clients)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(num_clients)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0

    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)

    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*num_clients)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=s-num_clients, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=num_clients-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==num_clients

    for r, clust in enumerate(superclass):
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*num_clients)
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*500)

        min_size = 0
        min_require_size = 15
        #beta = 0.5
        #np.random.seed(2021)
        #print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])

        #### Assigning samples to each client
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])

            dataidx = net_dataidx_map[j]
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp

        cnt+=n_parties


    clients_data = {}
    for i in range(num_clients):
        clients_data[i] = Subset(train_ds, net_dataidx_map[i])

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples


def FMNIST_SuperClass_NIID_Dir(train_ds,alpha,num_classes,num_clients,batch_size,seed):

    idxs_train = np.arange(len(train_ds))
    labels_train = np.array(train_ds.targets)
    y_train = np.array(train_ds.targets)


    superclass = [[0, 3, 1], [2, 4, 6, 8], [5, 9, 7]]
    nclass=num_classes
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(num_clients)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(num_clients)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0

    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)

    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*num_clients)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=s-num_clients, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=num_clients-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==num_clients

    for r, clust in enumerate(superclass):
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*num_clients)
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*6000)

        min_size = 0
        min_require_size = 15
        #beta = 0.1
        #np.random.seed(2021)
        print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])

        #### Assigning samples to each client
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])

            dataidx = net_dataidx_map[j]
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp

        cnt+=n_parties
    clients_data = {}
    for i in range(num_clients):

        clients_data[i] = Subset(train_ds, net_dataidx_map[i])

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples


def STL10_SuperClass_NIID_DIR(train_ds,alpha,num_classes,num_clients,batch_size,seed):

    labels_train = np.array(train_ds.labels)
    y_train = np.array(train_ds.labels)

    superclass = [[2, 8, 9], [0, 1, 7, 3, 4, 5, 6]]
    nclass=num_classes
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(num_clients)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(num_clients)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0

    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)

    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*num_clients)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=s-num_clients, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<num_clients:
        inds = np.random.choice(len(n_parties_ratio), size=num_clients-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==num_clients

    for r, clust in enumerate(superclass):
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*num_clients)
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*500)

        min_size = 0
        min_require_size = 15
        #alpha = 0.1
        #np.random.seed(2021)
        print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])

        #### Assigning samples to each client
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])

            dataidx = net_dataidx_map[j]
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp

            for key in tmp.keys():
                idxs_test = np.where(y_test==key)[0]
                net_dataidx_map_test[j] = np.hstack([net_dataidx_map_test[j], idxs_test])

            dataidx = net_dataidx_map_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[j] = tmp

        cnt+=n_parties
    clients_data = {}
    for i in range(num_clients):

        clients_data[i] = Subset(train_ds, net_dataidx_map[i])

    trainloaders, valloaders, client_samples = create_dataloaders(clients_data,batch_size,seed)

    return trainloaders, valloaders, client_samples