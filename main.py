from utils import *
from model import MODELS_DICT, CNN, ResNet18, ResNet34, ResNet50 
from unlearning import *

import torch
import torch.nn as nn
import torch.optim as optim
import time
import yaml

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_pretrain_model = True                     
    use_pretrain_model = False   
    store_history = False

    # step 0: parameters initialization
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    lab_desc = "cifar100_su_iid"
    params = config[lab_desc]

    lr = params["lr"]
    milestones = params["milestones"]
    gamma = params["gamma"]
    momentum = params["momentum"]
    batch_size = params["batch_size"]
    global_epochs = params["global_epochs"]
    local_epochs = params["local_epochs"]
    num_clients = params["num_clients"]
    eraser_indices = params["eraser_idx"]

    dataset_name = params["dataset"]
    num_classes = params["num_classes"]
    network = MODELS_DICT[params["network"]](num_classes=num_classes)

    backdoor_rate = params["backdoor_rate"]
    backdoor_rates = [backdoor_rate if i in eraser_indices else 0. for i in range(num_clients)]
    target_label = params["target_label"]

    scenario = params["scenario"]
    dirichlet = params["dirichlet"]

    print("Current device (default GPU): %d; Seed: %d" % (torch.cuda.current_device(), seed))
    print(lab_desc, ": \n", params)
    criterion = nn.CrossEntropyLoss()

    # step 1: load and process dataset
    print(">>> 1. load and process dataset %s" % dataset_name)
    
    client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader = dataprocessor(dataset_name, num_clients, batch_size=batch_size, 
                                                                                                scenario=scenario, backdoor_rates=backdoor_rates, 
                                                                                                target_label=target_label, alpha=dirichlet)
    
    forget_train_loaders = combine_dataloader(poisoned_loaders)
    retain_train_loaders = combine_dataloader(clean_loaders)

    # step 2: train the federated model on the whole dataset
    start_time = time.time()
    print(">>> 2. train the model on the whole dataset")
    global_model = deepcopy(network.to(DEVICE))
    client_models = [deepcopy(network.to(DEVICE)) for _ in range(num_clients)]

    if use_pretrain_model: 
        global_model = load_model_bydict(global_model, info="cifar100-pretrained")
    elif test_pretrain_model: 
        global_model = load_model_bydict(global_model, info=f"fedavg_{lab_desc}", root=f"models/{dataset_name}/")
        global_epochs = 0
        lr *= gamma ** (len(milestones))
    

    opts = [optim.SGD(client_model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4) for client_model in client_models]
    schedulers = [optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma) for opt in opts]

    global_model_trajectory = [deepcopy(global_model.state_dict())]
    client_model_trajectory = [deepcopy(global_model.state_dict())]
    
    for epoch in range(global_epochs):
        for client_idx, client_model in enumerate(client_models):
            client_model.load_state_dict(global_model.state_dict())
            for local_epoch in range(local_epochs):
                train(client_model, client_loaders[client_idx], opts[client_idx], criterion)
            schedulers[client_idx].step()
            if store_history and client_idx in eraser_indices: client_model_trajectory.append(deepcopy(client_model.state_dict()))

        FedAvg(global_model, client_models)
        
        if (epoch + 1) % 10 == 0: 
            print(f"Round {epoch}/{global_epochs}: ")
            f_accuracy = test(global_model, forget_train_loaders)
            r_accuracy = test(global_model, retain_train_loaders)
            val_accuracy = test(global_model, poisoned_test_loader)
            test_accuracy = test(global_model, clean_test_loader)
            print(f"UA: {f_accuracy:.4f}; BA: {val_accuracy:.4f}; RA: {r_accuracy:.4f}; CA: {test_accuracy:.4f}")

        if store_history: global_model_trajectory.append(deepcopy(global_model.state_dict()))

    print("training completed. time: %.4f seconds" % (time.time() - start_time))

    attack_model = MIA_Model(global_model, combine_dataloader(client_loaders), clean_test_loader, num_classes=num_classes)
    MIA_Infer(attack_model, global_model, forget_train_loaders)

    if test_pretrain_model == False: 
        save_model_bydict(global_model, info=f"fedavg_{lab_desc}_{seed}", root=f"models/{dataset_name}/")
    
    # # step 3: retrain the model on the retained dataset
    # SimplyRetrain(client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, num_clients, network, 
    #               attack_model=attack_model, MIA_loader=forget_train_loaders)

    # step 4: unlearning

    # FedRapidRetrain(client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
    #                 num_clients, eraser_indices, network, attack_model=attack_model, MIA_loader=forget_train_loaders)

    # FedRecovery(global_model_trajectory, client_model_trajectory, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
    #             num_clients, eraser_indices, network, attack_model=attack_model, MIA_loader=forget_train_loaders)
    
    # FedPGD(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
    #        num_clients, eraser_indices, network, attack_model=attack_model, MIA_loader=forget_train_loaders)
    
    # FedAsc(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
    #        num_clients, eraser_indices, network, attack_model=attack_model, MIA_loader=forget_train_loaders)

    # FedPostTrain(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
    #              num_clients, eraser_indices, network, attack_model=attack_model, MIA_loader=forget_train_loaders)
    
    FedGradDiff(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
            num_clients, backdoor_rates, network, attack_model=attack_model, MIA_loader=forget_train_loaders, mask_threshold=0.25)
    
    FedGradDiff(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
            num_clients, backdoor_rates, network, attack_model=attack_model, MIA_loader=forget_train_loaders, mask_threshold=0.5)
    
    FedGradDiff(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
            num_clients, backdoor_rates, network, attack_model=attack_model, MIA_loader=forget_train_loaders, mask_threshold=0.75)
    
    FedGradDiff(global_model, client_models, client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, 
            num_clients, backdoor_rates, network, attack_model=attack_model, MIA_loader=forget_train_loaders, mask_threshold=1.)
    
    
    