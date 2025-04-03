""" Federated Unlearning Methods 

- Pretraining: pretrain before unlearning to get a pretrained model
- Simply Retrain: baseline, retrain from scratch 
- FedGradBalance: our unlearning method combining information from both unlearning and remaining dataset
"""

import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *


def unlearning_framework(func): 
    """ unlearning framework decorator to measure the execution time of unlearning methods """
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        print(">>>", method_name)

        start_time = time.time()
        unlearn_model = func(*args, **kwargs)
        print("completed. time: %.4f seconds" % (time.time() - start_time))
    return wrapper


@unlearning_framework
def Pretraining(client_loaders, clean_test_loader, poisoned_test_loader, network, params):
    """ Pretraining the global and client models before unlearning
    
    :param client_loaders: list of dataloaders for training on each client (for pretraining)
    :param clean_test_loader: dataloader for clean test dataset (for validation)
    :param poisoned_test_loader: dataloader for poisoned test dataset (for validation)
    :param network: the initial random model (to be pretrained)
    :param params: hyperparameters for training

    :return: the final global model and client models after pretraining
    """
    lr = params["lr"]
    milestones = params["milestones"]
    gamma = params["gamma"]
    momentum = params["momentum"]
    global_epochs = params["global_epochs"]
    local_epochs = params["local_epochs"]
    global_model = deepcopy(network.cuda())
    num_clients = params["num_clients"]
    client_models = [deepcopy(network.cuda()) for _ in range(num_clients)]

    lr *= gamma ** (len(milestones))
    
    opts = [optim.SGD(client_model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4) for client_model in client_models]
    schedulers = [optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma) for opt in opts]
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(global_epochs):
        for client_idx, client_model in enumerate(client_models):
            client_model.load_state_dict(global_model.state_dict())
            for local_epoch in range(local_epochs):
                train(client_model, client_loaders[client_idx], opts[client_idx], criterion)
            schedulers[client_idx].step()

        FedAvg(global_model, client_models)
        
        if (epoch + 1) % 10 == 0: 
            print(f"Round {epoch}/{global_epochs}: ")
            val_accuracy = test(global_model, poisoned_test_loader)
            test_accuracy = test(global_model, clean_test_loader)
            print(f" BA: {val_accuracy:.4f}; CA: {test_accuracy:.4f}")
    return global_model, client_models 


@unlearning_framework
def SimplyRetrain(clean_loaders, clean_test_loader, poisoned_test_loader, network, params):
    """ Simply Retrain from scratch after unlearning
    
    :param clean_loaders: list of dataloaders for remaining dataset on each client
    :param clean_test_loader: dataloader for clean test dataset
    :param poisoned_test_loader: dataloader for poisoned test dataset
    :param network: the intial random model
    :param params: hyperparameters for training

    :return: the final retrained model
    """
    lr = params["lr"]
    milestones = params["milestones"]
    gamma = params["gamma"]
    momentum = params["momentum"]
    global_epochs = params["global_epochs"]
    local_epochs = params["local_epochs"]
    num_clients = params["num_clients"]

    criterion = nn.CrossEntropyLoss()
    
    retrain_global_model = deepcopy(network.cuda())
    retrain_client_models = [deepcopy(network.cuda()) for _ in range(num_clients)]
    opts = [torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4) for client_model in retrain_client_models]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma) for opt in opts]

    for epoch in range(global_epochs):
        for client_idx, client_model in enumerate(retrain_client_models):
            client_model.load_state_dict(retrain_global_model.state_dict())
            for local_epoch in range(local_epochs):
                train(client_model, clean_loaders[client_idx], opts[client_idx], criterion)
            schedulers[client_idx].step()
        FedAvg(retrain_global_model, retrain_client_models)
        
        if (epoch + 1) % 10 == 0: 
            print(f"Round {epoch}/{global_epochs}: ")
            val_accuracy = test(retrain_global_model, poisoned_test_loader)
            test_accuracy = test(retrain_global_model, clean_test_loader)
            print(f"BA: {val_accuracy:.4f}; CA: {test_accuracy:.4f}")
    
    return retrain_global_model


@unlearning_framework
def FedGradBalance(global_model, client_models, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, params, mask_threshold=0.5):
    """ Federated Gradient Balacing 
    consisted of two phase: balanced forgetting and utility refinement

    :param global_model, client_models: the initial global and client models after pretraining
    :param poisoned_loaders: list of dataloaders for unlearning dataset on each client
    :param clean_loaders: list of dataloaders for remaining dataset on each client
    :param clean_test_loader: dataloader for clean test dataset
    :param poisoned_test_loader: dataloader for poisoned test dataset
    :param params: hyperparameters for training
    :param mask_threshold: threshold for oriented saliency compression

    :return: the final unlearned model
    """

    # parameter initialization
    ul_lr = params["lr"]
    ft_lr = params["lr"] * params["gamma"] ** (len(params["milestone"]))
    momentum = params["momentum"]
    global_epochs = 5
    local_epochs = params["local_epochs"]
    criterion = nn.CrossEntropyLoss()    
    backdoor_rate = params["backdoor_rate"]
    backdoor_rates = [backdoor_rate if i in params["eraser_indices"] else 0. for i in range(params["num_clients"])]
    retained_global_model = deepcopy(global_model)
    retained_client_models = deepcopy(client_models)


    """ mask for oriented saliency compression"""
    mask_f = saliency_mask(global_model, combine_dataloader(poisoned_loaders), threshold=mask_threshold)
    mask_r = saliency_mask(global_model, combine_dataloader(clean_loaders), threshold=mask_threshold)


    """ balanced forgetting """
    for epoch in range(global_epochs):
        print(f"Round {epoch}/{global_epochs}", end=": ")

        for client_idx, client_model in enumerate(retained_client_models): 
            client_model.load_state_dict(retained_global_model.state_dict())
            
            for local_epoch in range(local_epochs): 
                opt = torch.optim.SGD(client_model.parameters(), lr=ul_lr, momentum=momentum)

                if client_idx in params["eraser_indices"]: 
                    for (data_r, target_r), (data_f, target_f) in zip(clean_loaders[client_idx], poisoned_loaders[client_idx]): 
                        client_model.train()
                        opt.zero_grad()
                        data_f, target_f, data_r, target_r = data_f.cuda(), target_f.cuda(), data_r.cuda(), target_r.cuda()
                        loss_f = criterion(client_model(data_f), target_f)
                        loss_r = criterion(client_model(data_r), target_r)
                        loss = -backdoor_rates[client_idx] * loss_f + (1 - backdoor_rates[client_idx]) * loss_r
                        loss.backward()
                        opt.step()
                else: 
                    for data, target in clean_loaders[client_idx]: 
                        client_model.train()
                        opt.zero_grad()
                        data, target = data.cuda(), target.cuda()
                        loss = criterion(client_model(data), target)
                        loss.backward()
                        opt.step()

            for m, param, old_param in zip(mask_f.values(), client_model.parameters(), retained_global_model.parameters()): 
                param = m * param + (1 - m) * old_param 

        FedAvg(retained_global_model, retained_client_models, weights=None)

        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f" BA: {val_accuracy:.4f}; CA: {test_accuracy:.4f}")
    

    """ utility refinement """
    global_epochs = 10
    retained_train_loader = deepcopy(clean_loaders)
    retained_client_models = deepcopy(client_models)

    for epoch in range(global_epochs):
        print(f"Round {epoch}/{global_epochs}", end=": ")

        for client_idx, client_model in enumerate(retained_client_models):
            client_model.load_state_dict(retained_global_model.state_dict())
            for local_epoch in range(local_epochs):
                opt = torch.optim.SGD(client_model.parameters(), lr=ft_lr, momentum=momentum)
                for data, target in retained_train_loader[client_idx]: 
                    client_model.train()
                    data, target = data.cuda(), target.cuda()
                    opt.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    opt.step()
            
            for m, param, old_param in zip(mask_r.values(), client_model.parameters(), retained_global_model.parameters()): 
                param = m * param + (1 - m) * old_param 
        
        FedAvg(retained_global_model, retained_client_models)
        
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"BA: {val_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model
