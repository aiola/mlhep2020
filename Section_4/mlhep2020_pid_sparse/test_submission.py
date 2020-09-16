import os
from sys import argv
import sys
import pandas
import numpy as np
import torch
from sparse_model import SparseNet
from sparse_vd import LinearSVDO
device = torch.device('cpu')

baseline_neurons = 50000
label_class_correspondence = {'Electron': 0, 'Ghost': 1, 'Kaon': 2, 'Muon': 3, 'Pion': 4, 'Proton': 5}
class_label_correspondence = {0: 'Electron', 1: 'Ghost', 2: 'Kaon', 3: 'Muon', 4: 'Pion', 5: 'Proton'}

def get_class_ids(labels):
    """
    Convert particle type names into class ids.

    Parameters:
    -----------
    labels : array_like
        Array of particle type names ['Electron', 'Muon', ...].

    Return:
    -------
    class ids : array_like
        Array of class ids [1, 0, 3, ...].
    """
    return np.array([label_class_correspondence[alabel] for alabel in labels])

def get_score(logloss):
    k = -1. / 0.9
    b = 1.2 / 0.9
    score = b + k * logloss
    score = max(score, 0)
    score = min(score, 1)
    return score

def calc_score_train(input_file="train.csv.gz"):
    train = pandas.read_csv(input_file)
    features = list(set(train.columns) - {'Label', 'Class', 'ID'})
    features = sorted(features)

    model = SparseNet(input_dim=len(features), device=device).to(device)
    model.load_state_dict(torch.load('./model_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    
    X = torch.tensor(train[features].values).float()
    preds = model(X)
    labels = get_class_ids(train.Label.values)

    nllloss  = torch.nn.NLLLoss()
    score = nllloss(preds, torch.tensor(labels).long())
    
    effecive_number_parameters = 0
    total_number_parameters = 0
    for module in model.children():
        if isinstance(module, LinearSVDO):
            effecive_number_parameters += module.count_parameters()[0]
            total_number_parameters += module.count_parameters()[1]
        else:
            for param in module.parameters():
                effecive_number_parameters += param.numel()
                total_number_parameters += param.numel()
    
    final_score = get_score(score.detach().numpy()) * np.log(1 + baseline_neurons / effecive_number_parameters)
    return final_score
    
if __name__ == "__main__":
    input_file = argv[1]
    print("Score on train dataset is: {}".format(calc_score_train(input_file)))