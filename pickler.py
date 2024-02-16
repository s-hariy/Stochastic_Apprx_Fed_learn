import os
import pickle
from constants import FOLDER

def pickle_results(server_side_train_loss,server_side_accuracy,server_side_loss,server_side_precision,server_side_recall,server_side_fscore,server_side_train_accuracy):
    os.makedirs(FOLDER,exist_ok=True)

    with open(f'{FOLDER}/server_side_train_loss.pkl', 'wb') as f:
        pickle.dump(server_side_train_loss, f)
        
    with open(f'{FOLDER}/server_side_accuracy.pkl', 'wb') as f:
        pickle.dump(server_side_accuracy, f)
        
    with open(f'{FOLDER}/server_side_loss.pkl', 'wb') as f:
        pickle.dump(server_side_loss, f)

    with open(f'{FOLDER}/server_side_precision.pkl', 'wb') as f:
        pickle.dump(server_side_precision, f)
        
    with open(f'{FOLDER}/server_side_recall.pkl', 'wb') as f:
        pickle.dump(server_side_recall, f)

    with open(f'{FOLDER}/server_side_fscore.pkl', 'wb') as f:
        pickle.dump(server_side_fscore, f)
        
    with open(f'{FOLDER}/server_side_train_accuracy.pkl', 'wb') as f:
        pickle.dump(server_side_train_accuracy, f)