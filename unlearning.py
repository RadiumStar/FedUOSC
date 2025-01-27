""" Federated Unlearning Method Library
"""

from utils import *

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import time

def unlearning_framework(func): 
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        print(">>>", method_name)

        start_time = time.time()
        unlearn_model = func(*args, **kwargs)
        print("unlearning completed. time: %.4f seconds" % (time.time() - start_time))

        MIA_Infer(kwargs['attack_model'], unlearn_model, kwargs['MIA_loader'])
    return wrapper


@unlearning_framework
def SimplyRetrain(client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, network, attack_model, MIA_loader):
    lr = 0.01
    momentum = 0.9
    milestones = [100]
    gamma = 0.1
    global_epochs = 100
    local_epochs = 1
    criterion = nn.CrossEntropyLoss()

    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)
    
    retrain_global_model = deepcopy(network.to(DEVICE))
    # retrain_global_model = load_model_bydict(retrain_global_model, info="cifar100-pretrained")
    retrain_client_models = [deepcopy(network.to(DEVICE)) for _ in range(num_clients)]
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
            f_accuracy = test(retrain_global_model, forget_train_loaders)
            r_accuracy = test(retrain_global_model, retain_train_loaders)
            val_accuracy = test(retrain_global_model, poisoned_test_loader)
            test_accuracy = test(retrain_global_model, clean_test_loader)
            print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")
    
    # save_models(retrain_global_model, info="retrain_%s_%s_%dclients_%depochs" % (network.__name__, "mnist", num_clients, global_epochs))
    return retrain_global_model


@unlearning_framework
def FedRapidRetrain(client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, eraser_idx, network, attack_model, MIA_loader):
    # step 0: parameter initialization
    global_epochs = 100
    local_epochs = 1
    t0 = 5
    lr = 0.01
    momentum = 0.9
    gamma = 0.2
    criterion = nn.CrossEntropyLoss()

    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)

    retrain_global_model = network.to(DEVICE)
    # retrain_global_model = load_model_bydict(retrain_global_model, info="cifar100-pretrained")
    retrain_client_models = [deepcopy(retrain_global_model) for _ in range(num_clients)]
    
    retained_train_loader = deepcopy(clean_loaders)
    
    # step 1: Rapid Retrain
    for epoch in range(global_epochs):
        # if epoch == 60: lr *= gamma 
        # elif epoch == 120: lr *= gamma
        # elif epoch == 160: lr *= gamma
        for client_idx, client_model in enumerate(retrain_client_models):
            client_model.load_state_dict(retrain_global_model.state_dict())
            if (epoch + 1) % t0 == 0:
                for local_epoch in range(local_epochs):
                    client_model.train()
                    opt = AdaHessian(client_model.parameters(), lr=0.001)
                    for data, target in retained_train_loader[client_idx]:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        opt.zero_grad(set_to_none=True)
                        output = client_model(data)
                        loss = criterion(output, target)
                        grads = torch.autograd.grad(loss, client_model.parameters(), create_graph=True)
                        for param, grad in zip(client_model.parameters(), grads):
                            param.grad = grad
                        opt.step()
                # schedulers[client_idx].step()
            else: 
                for local_epoch in range(local_epochs):
                    client_model.train()
                    opt = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)
                    for data, target in retained_train_loader[client_idx]:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        opt.zero_grad(set_to_none=True)
                        output = client_model(data)
                        loss = criterion(output, target)
                        grads = torch.autograd.grad(loss, client_model.parameters())
                        for param, grad in zip(client_model.parameters(), grads):
                            param.grad = grad
                        opt.step()
                # schedulers[client_idx].step()

        FedAvg(retrain_global_model, retrain_client_models)
        
        if (epoch + 1) % 1 == 0: 
            print(f"Round {epoch}/{global_epochs}: ")
            f_accuracy = test(retrain_global_model, forget_train_loaders)
            r_accuracy = test(retrain_global_model, retain_train_loaders)
            val_accuracy = test(retrain_global_model, poisoned_test_loader)
            test_accuracy = test(retrain_global_model, clean_test_loader)
            print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")
    
    return retrain_global_model


@unlearning_framework
def FedRecovery(global_model_trajectory, client_model_trajectory, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, eraser_indices, network, attack_model, MIA_loader): 
    # step 0: parameter initialization
    std = 0.01
    lr = 0.01
    global_epochs = len(global_model_trajectory) - 1
    criterion = nn.CrossEntropyLoss()
    eraser_idx = eraser_indices[0]

    # step 1: train on the whole dataset(repeat what we do in main.py because we need trajectory)
    # have done in main.py

    # step 2: unlearning
    start_time = time.time()
    grad_dicts, Delta_Fs, ps, residuals = [], [], [], []
    unlearn_model_dict = deepcopy(global_model_trajectory[-1])
    with torch.no_grad():
        # step 2.1: aggregate the gradients for the global model
        for i in range(global_epochs):
            grad_dict = {layer: (global_model_trajectory[i + 1][layer] - global_model_trajectory[i][layer]) for layer in global_model_trajectory[i].keys()}
            Delta_Fs.append(torch.cat([param.reshape((-1, 1)) for param in grad_dict.values()], dim=0).squeeze().to(DEVICE))
            grad_dicts.append(grad_dict)
        
        # step 2.2: compute the gradient residuals 
        for i in range(global_epochs): 
            residual = {}
            for layer in unlearn_model_dict.keys(): 
                delta_f_erase = (client_model_trajectory[i + 1][layer] - global_model_trajectory[i][layer]) / num_clients
                residual[layer] = 1.0 / (num_clients - 1) * (grad_dicts[i][layer] - delta_f_erase)
            residuals.append(residual)

        # step 2.3: compute weight p for each gradient residuals
        sum_Delta_Fs = sum(torch.norm(Delta_F) ** 2 for Delta_F in Delta_Fs)
        ps = [torch.norm(Delta_F) ** 2 / sum_Delta_Fs * len(Delta_Fs) for Delta_F in Delta_Fs]

        # step 2.4: subtract a weighted sum of residuals
        for i in range(global_epochs - 1):
            for layer in unlearn_model_dict.keys(): 
                if unlearn_model_dict[layer].dtype == torch.float32: 
                    unlearn_model_dict[layer] -= ps[i] * residuals[i][layer] 
    
    # step 2.5: add gaussian noise
    for layer in unlearn_model_dict.keys():
        if unlearn_model_dict[layer].dtype == torch.float32: 
            noise = torch.empty_like(unlearn_model_dict[layer]).normal_(0, std)
            unlearn_model_dict[layer] += noise

    print("unlearning completed. time: %.4f seconds" % (time.time() - start_time))
    
    retained_global_model = network.to(DEVICE)
    retained_global_model.load_state_dict(unlearn_model_dict)
    retained_client_models = [network.to(DEVICE) for _ in range(num_clients - 1)]
    retained_train_loader = deepcopy(clean_loaders)
    retained_train_loader.pop(eraser_idx)

    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)

    for client_idx, client_model in enumerate(retained_client_models):
        client_model.load_state_dict(retained_global_model.state_dict())
        opt = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
        train(client_model, retained_train_loader[client_idx], opt, criterion)
    
    FedAvg(retained_global_model, retained_client_models)
    
    f_accuracy = test(retained_global_model, forget_train_loaders)
    r_accuracy = test(retained_global_model, retain_train_loaders)
    val_accuracy = test(retained_global_model, poisoned_test_loader)
    test_accuracy = test(retained_global_model, clean_test_loader)
    print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model
    

@unlearning_framework
def FedPGD(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, eraser_indices, network, attack_model, MIA_loader):     
    # step 0: parameter initialization
    lr = 0.01
    distance_threshold = 2.2
    local_epochs = 5
    clip_grad = 5
    eraser_idx = eraser_indices[0]
    criterion = nn.CrossEntropyLoss()

    # step 1: define reference model
    ref_model = get_reference_model(global_model, client_models[eraser_idx], num_clients)

    # compute radius of l2-norm ball
    dist_ref_random_lst = [get_distance(ref_model, network.to(DEVICE)) for _ in range(num_clients)]
    radius = np.mean(dist_ref_random_lst) / 3
    print(f'Radius for model_ref: {radius}')

    unlearn_model = deepcopy(ref_model)
    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)
    opt = torch.optim.SGD(unlearn_model.parameters(), lr=lr, momentum=0.9)
    unlearn_model.train()

    # step 2: unlearning
    stop = False
    for epoch in range(local_epochs):
        if stop: break  # early stopping based on distance threshold
        for data, target in forget_train_loaders: 
            unlearn_model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            output = unlearn_model(data)
            loss = -criterion(output, target)
            loss.backward()
            if clip_grad > 0: nn.utils.clip_grad_norm_(unlearn_model.parameters(), clip_grad)
            opt.step()
            
            with torch.no_grad():
                dist = get_distance(unlearn_model, ref_model) 
                if dist > radius: # PGD
                    dist_vec = nn.utils.parameters_to_vector(unlearn_model.parameters()) - nn.utils.parameters_to_vector(ref_model.parameters())
                    dist_vec = dist_vec / torch.norm(dist_vec) * np.sqrt(radius)
                    proj_vec = nn.utils.parameters_to_vector(ref_model.parameters()) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, unlearn_model.parameters())
                    dist = get_distance(unlearn_model, ref_model) 

                if get_distance(unlearn_model, client_models[eraser_idx]) > distance_threshold: 
                    stop = True
                    break

        print(f"Ul Round {epoch}/{local_epochs}: ")
        f_accuracy = test(unlearn_model, forget_train_loaders)
        r_accuracy = test(unlearn_model, retain_train_loaders)
        val_accuracy = test(unlearn_model, poisoned_test_loader)
        test_accuracy = test(unlearn_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")
    
    # step 3: post-training
    retained_global_model = deepcopy(unlearn_model)
    retained_train_loader = deepcopy(clean_loaders)
    retained_train_loader.pop(eraser_idx)
    retained_client_models = deepcopy(client_models)
    retained_client_models.pop(eraser_idx)

    global_epochs = 10
    local_epochs = 1

    for epoch in range(global_epochs):
        for client_idx, client_model in enumerate(retained_client_models):
            client_model.load_state_dict(retained_global_model.state_dict())
            for local_epoch in range(local_epochs):
                opt = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
                train(client_model, retained_train_loader[client_idx], opt, criterion)
        
        FedAvg(retained_global_model, retained_client_models)
        
        print(f"Round {epoch}/{global_epochs}: ")
        f_accuracy = test(retained_global_model, forget_train_loaders)
        r_accuracy = test(retained_global_model, retain_train_loaders)
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model


@unlearning_framework
def FedPostTrain(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, eraser_indices, network, attack_model, MIA_loader): 
    global_epochs = 25
    local_epochs = 1
    lr = 0.01
    criterion = nn.CrossEntropyLoss()

    # post-training
    retained_global_model = deepcopy(global_model)
    retained_train_loader = deepcopy(clean_loaders)
    retained_client_models = deepcopy(client_models)
    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)

    for epoch in range(global_epochs):
        print(f"Round {epoch}/{global_epochs}", end=": ")

        for client_idx, client_model in enumerate(retained_client_models):
            client_model.load_state_dict(retained_global_model.state_dict())
            for local_epoch in range(local_epochs):
                opt = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
                train(client_model, retained_train_loader[client_idx], opt, criterion)
        
        FedAvg(retained_global_model, retained_client_models)
        
        f_accuracy = test(retained_global_model, forget_train_loaders)
        r_accuracy = test(retained_global_model, retain_train_loaders)
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model


@unlearning_framework
def FedAsc(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, eraser_indices, network, attack_model, MIA_loader): 
    # step 0: parameter initialization
    lr = 0.01
    local_epochs = 1
    clip_grad = 5
    distance_threshold = 3.2
    eraser_idx = eraser_indices[0]
    criterion = nn.CrossEntropyLoss()

    unlearn_model = deepcopy(client_models[eraser_idx])
    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)
    opt = torch.optim.SGD(unlearn_model.parameters(), lr = lr, momentum=0.9)
    unlearn_model.train()

    # step 1: unlearning
    for eraser_idx in eraser_indices: 
        for epoch in range(local_epochs):
            for data, target in poisoned_loaders[eraser_idx]: 
                data, target = data.to(DEVICE), target.to(DEVICE)
                opt.zero_grad()
                output = unlearn_model(data)
                loss = -criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(unlearn_model.parameters(), clip_grad)
                opt.step()
    
    # step 2: post-training
    retained_global_model = deepcopy(unlearn_model)
    retained_train_loader = deepcopy(clean_loaders)
    retained_client_models = deepcopy(client_models)

    f_accuracy = test(retained_global_model, forget_train_loaders)
    r_accuracy = test(retained_global_model, retain_train_loaders)
    val_accuracy = test(retained_global_model, poisoned_test_loader)
    test_accuracy = test(retained_global_model, clean_test_loader)
    print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    global_epochs = 10
    local_epochs = 1

    for epoch in range(global_epochs):
        print(f"Round {epoch}/{global_epochs}", end=": ")

        for client_idx, client_model in enumerate(retained_client_models):
            client_model.load_state_dict(retained_global_model.state_dict())
            for local_epoch in range(local_epochs):
                opt = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
                train(client_model, retained_train_loader[client_idx], opt, criterion)
        
        FedAvg(retained_global_model, retained_client_models)
        
        f_accuracy = test(retained_global_model, forget_train_loaders)
        r_accuracy = test(retained_global_model, retain_train_loaders)
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model


@unlearning_framework
def FedGradDiff(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
                num_clients, backdoor_rates, network, attack_model, MIA_loader, mask_threshold=0.5):
    """ Gradient Difference 
    """

    # step 0: parameter initialization
    ul_lr, ft_lr = 0.01, 0.0004
    momentum = 0.9
    global_epochs = 3
    local_epochs = 1
    clip_grad = 5.
    criterion = nn.CrossEntropyLoss()

    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)

    gammas = backdoor_rates
    compress_mode = 'top-k'

    # step 1: define reference model
    retained_global_model = deepcopy(global_model)
    retained_client_models = deepcopy(client_models)

    mask_f = saliency_mask(global_model, forget_train_loaders, threshold=mask_threshold, mode=compress_mode)
    mask_r = saliency_mask(global_model, retain_train_loaders, threshold=mask_threshold, mode=compress_mode)
    poisoners = [i for i, rate in enumerate(backdoor_rates) if rate > 0]

    # step 2: unlearning
    for epoch in range(global_epochs):
        print(f"Round {epoch}/{global_epochs}", end=": ")

        for client_idx, client_model in enumerate(retained_client_models): 
            client_model.load_state_dict(retained_global_model.state_dict())
            
            for local_epoch in range(local_epochs): 
                opt = torch.optim.SGD(client_model.parameters(), lr=ul_lr, momentum=momentum)

                if client_idx in poisoners: 
                    for (data_r, target_r), (data_f, target_f) in zip(clean_loaders[client_idx], poisoned_loaders[client_idx]): 
                        client_model.train()
                        opt.zero_grad()
                        data_f, target_f, data_r, target_r = data_f.to(DEVICE), target_f.to(DEVICE), data_r.to(DEVICE), target_r.to(DEVICE)
                        loss_f = criterion(client_model(data_f), target_f)
                        loss_r = criterion(client_model(data_r), target_r)
                        loss = -gammas[client_idx] * loss_f + (1 - gammas[client_idx]) * loss_r
                        loss.backward()
                        nn.utils.clip_grad_norm_(client_model.parameters(), clip_grad)
                        opt.step()
                else: 
                    for data, target in clean_loaders[client_idx]: 
                        client_model.train()
                        opt.zero_grad()
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        loss = criterion(client_model(data), target)
                        loss.backward()
                        nn.utils.clip_grad_norm_(client_model.parameters(), clip_grad)
                        opt.step()

            for m, param, old_param in zip(mask_f.values(), client_model.parameters(), retained_global_model.parameters()): 
                param = m * param + (1 - m) * old_param 

        # weights = [init_unlearning_weight if i == 0 else (1 - init_unlearning_weight) / (num_clients - 1) for i in range(num_clients)]
        FedAvg(retained_global_model, retained_client_models, weights=None)

        f_accuracy = test(retained_global_model, forget_train_loaders)
        r_accuracy = test(retained_global_model, retain_train_loaders)
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")
    
    global_epochs = 10
    local_epochs = 1
    post_training_compress = False 

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
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    opt.zero_grad()
                    output = client_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    opt.step()
            
            for m, param, old_param in zip(mask_r.values(), client_model.parameters(), retained_global_model.parameters()): 
                param = m * param + (1 - m) * old_param 
        
        FedAvg(retained_global_model, retained_client_models)
        
        f_accuracy = test(retained_global_model, forget_train_loaders)
        r_accuracy = test(retained_global_model, retain_train_loaders)
        val_accuracy = test(retained_global_model, poisoned_test_loader)
        test_accuracy = test(retained_global_model, clean_test_loader)
        print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

    return retained_global_model
