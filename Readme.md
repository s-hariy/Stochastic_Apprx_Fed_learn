## Instructions for running the code
1. Clone the repo.
2. Install Python >=3.8.18
3. Install requirements from requirements.txt
4. Check the available user arguments
```
  --method {FedAvg,FedProx,Stochastic,Stochastic_each_epoch}
                        Federated learning method
  --dataset {CIFAR10,CIFAR100,STL10,F-MNIST}
                        Dataset to use
  --architecture {CNN,LeNet,Resnet9}
                        Model architecture to use
  --seed SEED           Random seed
  --num_clients NUM_CLIENTS
                        Number of clients
  --classes_per_client CLASSES_PER_CLIENT
                        Number of classes per client for N-IID
  --split {IID,DIR-N-IID,SC-DIR-N-IID}
                        Data distribution among clients
  --alpha ALPHA         Alpha parameter for dirichlet distribution
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --decay_rate DECAY_RATE
                        Decay rate
  --mom MOM             momentum
  --mu MU               mu for FedProx
  --min_fit_clients MIN_FIT_CLIENTS
                        Minimum number of clients for training
  --fraction_fit_clients FRACTION_FIT_CLIENTS
                        Minimum number of clients for training
  --min_evaluate_clients MIN_EVALUATE_CLIENTS
                        Minimum number of clients for evaluation
  --fraction_evaluate_clients FRACTION_EVALUATE_CLIENTS
                        Minimum number of clients for evaluation
  --server_round SERVER_ROUND
                        Number of server rounds
  --local_epochs LOCAL_EPOCHS
                        Number of local epochs
  --device DEVICE       Device to use for computation
```
5. Run the code
```
python Fed.py
```
6. Example
```
python Fed.py --method FedAvg --dataset CIFAR10 --architecture Resnet9
```
