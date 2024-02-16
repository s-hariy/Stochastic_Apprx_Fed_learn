from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from constants import *

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

    
def train(net, trainloader, epochs: int, server_rounds:int):
    """Train the network on the training set."""
    add_epoch_acc = 0
    add_epoch_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    if METHOD == "FedAvg" or METHOD == "FedProx":
        lr = LR
        for i in range(server_rounds-1):
            lr*=DECAY_RATE
    elif METHOD == "Stochastic":
        lr = LR/server_rounds
    elif METHOD == "Stochastic_each_epoch":
        optimizer = torch.optim.SGD(net.parameters(),lr=lr)
        lambda1 = lambda e: (1/((server_rounds-1)*epochs+e+1))
        scheduler =  torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        before_lr = optimizer.param_groups[0]["lr"]
        if METHOD == "Stochastic_each_epoch" and epoch!=epochs-1:
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        
        add_epoch_loss+=epoch_loss
        add_epoch_acc+=epoch_acc
            
        print("Round %d :Epoch %d: SGD lr %f -> %f" % (server_rounds,epoch, before_lr, after_lr))
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")



def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for images, labels in testloader:
            y_true+=labels
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().detach().numpy().tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    precision,recall,fscore,support = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return loss, accuracy,precision,recall,fscore,support