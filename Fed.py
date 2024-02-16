import argparse
import os
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from constants import *
from logger import setup_logger
from data import load_datasets
from model import get_model
from pickler import pickle_results
from seeder import seed_everything
from typing import Dict, List, Optional, Tuple
from utils import *



def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Configuration')

    parser.add_argument('--method', type=str, default="FedAvg", choices=["FedAvg", "FedProx", "Stochastic", "Stochastic_each_epoch"], help='Federated learning method')
    parser.add_argument('--dataset', type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "STL10","F-MNIST"], help='Dataset to use')
    parser.add_argument('--architecture', type=str, default="Resnet9", choices=["CNN", "LeNet", "Resnet9"], help='Model architecture to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    # parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Number of classes per client for N-IID')
    parser.add_argument('--split', type=str, default="DIR-N-IID", choices=["IID", "DIR-N-IID", "SC-DIR-N-IID"], help='Data distribution among clients')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for dirichlet distribution')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate')
    parser.add_argument('--mom', type=float, default=0.0, help='momentum')
    parser.add_argument('--mu', type=float, default=0.25, help='mu for FedProx')
    parser.add_argument('--min_fit_clients', type=int, default=10, help='Minimum number of clients for training')
    parser.add_argument('--fraction_fit_clients', type=float, default=1.0, help='Minimum number of clients for training')
    parser.add_argument('--min_evaluate_clients', type=int, default=5, help='Minimum number of clients for evaluation')
    parser.add_argument('--fraction_evaluate_clients', type=float, default=1.0, help='Minimum number of clients for evaluation')
    parser.add_argument('--server_round', type=int, default=100, help='Number of server rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to use for computation')

    args = parser.parse_args()

    # Construct the folder name based on arguments
    args.folder = f"{args.dataset}/{args.num_clients}_clients_{args.min_fit_clients}_E_{args.local_epochs}_R_{args.server_round}/{args.split}/class_per_client_{args.classes_per_client}/bs{args.batch_size}/lr_{args.lr}_m{args.mom}_{args.architecture}/{args.method}"
    return args


def main(args):
    log = setup_logger(f"./{args.dataset}_{args.method}_{args.split}_{args.architecture}_{args.lr}.log")

    log.info(
        f"Training on {args.device} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    server_side_loss = []
    server_side_accuracy = []
    server_side_precision = []
    server_side_recall = []
    server_side_fscore = []
    server_side_support = []
    server_side_train_loss = []
    server_side_train_accuracy = []
        
    seed_everything(args.seed)
    trainloaders, valloaders, testloader, client_samples, num_classes = load_datasets(args.dataset, args.num_clients, args.batch_size, args.split, args.classes_per_client, args.alpha, args.seed)
    global_model = get_model(args.dataset,args.architecture)
    print(global_model)

    def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]


    def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        net.load_state_dict(state_dict, strict=True)


    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, cid, net, trainloader, valloader):
            self.cid = cid
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader

        def get_parameters(self, config):
            log.info(f"[Client {self.cid}] get_parameters")
            return get_parameters(self.net)

        def fit(self, parameters, config):
            server_round = config["server_round"]
            local_epochs = config["local_epochs"]
            log.info(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
            set_parameters(self.net, parameters)
            train(self.net, self.trainloader,local_epochs,server_round,)
            return get_parameters(self.net), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            log.info(f"[Client {self.cid}] evaluate, config: {config}")
            set_parameters(self.net, parameters)
            loss, accuracy,precision,recall,fscore,support= test(self.net, self.valloader)
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


    def client_fn(cid) -> FlowerClient:
        net = global_model.to(args.device)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader)


    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        """
        config = {
            "server_round": server_round,  
            "local_epochs": args.local_epochs, 
        }
        return config



    # The `evaluate` function will be by Flower called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = global_model.to(args.device)
        set_parameters(net, parameters)  # Update model with the latest parameters
        server_train_loss = 0
        server_train_accuracy = 0
        for train_loader in trainloaders:
       
            train_loss, train_accuracy,train_precision,train_recall,train_fscore,train_support = test(net, train_loader)
            server_train_loss+=train_loss
            server_train_accuracy+=train_accuracy
        server_side_train_loss.append(server_train_loss/len(trainloaders))
        server_side_train_accuracy.append(server_train_accuracy/len(trainloaders))

        
        valloader = testloader
        set_parameters(net, parameters)  # Update model with the latest parameters
        val_loss, val_accuracy,val_precision,val_recall,val_fscore,val_support= test(net, valloader)
        log.info(f"Server-side evaluation loss {val_loss} / accuracy {val_accuracy} / precision {val_precision} / recall {val_recall} / fscore {val_fscore}")
        server_side_loss.append(val_loss)
        server_side_accuracy.append(val_accuracy)
        server_side_precision.append(val_precision)
        server_side_recall.append(val_recall)
        server_side_fscore.append(val_fscore)
        server_side_support.append(val_support)
        log.info("Computed metrics")
        return val_loss, {"accuracy": val_accuracy}

    if args.method == "FedAvg" or args.method == "Stochastic" or args.method == "Stochastic_each_epoch":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit_clients,
            fraction_evaluate=args.fraction_evaluate_clients,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model)),
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config, 
        )
    elif args.method == "FedProx":
        strategy = fl.server.strategy.FedProx(
            fraction_fit=args.fraction_fit_clients,
            fraction_evaluate=args.fraction_evaluate_clients,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(global_model)),
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,  
            proximal_mu = 0.25,
        )
    else:
        log.info("METHOD NOT IMPLEMENTED. EXITING")
        import sys
        sys.exit()

    client_resources = None
    if args.device == "cuda":
        client_resources = {"num_gpus": 1}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.server_round), 
        strategy=strategy,
        client_resources=client_resources,
    )


    pickle_results(server_side_train_loss,server_side_accuracy,server_side_loss,server_side_precision,server_side_recall,server_side_fscore,server_side_train_accuracy)


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)

    
