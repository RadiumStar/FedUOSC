from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset


DATA_ROOT = "../../data/"


def dataloader2output(model, dataloader, true_label=False):
    """Get the output of model as predicted labels"""
    model.eval()
    outputs, predicted_labels = [], []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            logits = model(data)
            pred = F.softmax(logits, dim=1).cpu().numpy()  # Get probability distribution
            outputs.append(pred)

            if true_label: 
                predicted_labels.append(target)
            else: 
                predicted_label = np.argmax(pred, axis=1)  # Get predicted labels
                predicted_labels.append(predicted_label)

    return np.concatenate(outputs), np.concatenate(predicted_labels)


def dataloader2loss(model, dataloader): 
    """Get the output of model as loss value"""
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.cuda()
            logits = model(data)
            loss = criterion(logits, target).item()
            losses.append(loss)

    return losses


def add_backdoor_trigger(image, trigger_type="pattern", trigger_size=3, trigger_value=255, distance=2):
    """ Add a backdoor trigger to the image.

    :param image: Input image, either HWC or HW format.
    :param trigger_type: Trigger type ("pixel" or "pattern").
    :param trigger_size: Size of the backdoor trigger (default: 3x3).
    :param trigger_value: Value to fill the backdoor trigger (default: 255 for raw images).
    :param distance: Distance from bottom-right corner.

    :return: Image with backdoor trigger added.
    """
    # Determine if image has channels (HWC) or not (HW)
    has_channels = len(image.shape) == 3
    h, w = image.shape[:2]
    c = image.shape[2] if has_channels else 1

    # Boundary check
    if h < trigger_size + distance or w < trigger_size + distance:
        raise ValueError("Trigger size and distance exceed image boundaries.")

    if trigger_type == "pixel":
        if has_channels:
            # Multi-channel (HWC, e.g., CIFAR)
            image[-trigger_size - distance: -distance, -trigger_size - distance: -distance, :] = trigger_value
        else:
            # Single-channel (HW, e.g., MNIST)
            image[-trigger_size - distance: -distance, -trigger_size - distance: -distance] = trigger_value

    elif trigger_type == "pattern":
        pattern = np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]])
        if has_channels:
            # Multi-channel (HWC)
            for ch in range(c):
                trigger_area = np.where(pattern == 1, trigger_value, image[-trigger_size - distance: -distance, -trigger_size - distance: -distance, ch])
                image[-trigger_size - distance: -distance, -trigger_size - distance: -distance, ch] = trigger_area
        else:
            # Single-channel (HW)
            trigger_area = torch.tensor(np.where(pattern == 1, trigger_value, image[-trigger_size - distance: -distance, -trigger_size - distance: -distance]), dtype=image.dtype)
            image[-trigger_size - distance: -distance, -trigger_size - distance: -distance] = trigger_area

    else:
        raise ValueError("Invalid trigger type. Choose 'pixel' or 'pattern'.")

    return image


def split_non_iid(dataset, num_clients, alpha=0.5, num_classes=10): 
    """ Return indices of dataset for each client in non-iid scenario

    :param dataset: pytorch dataset
    :param num_clients: number of clients
    :param alpha: parameter of Dirichlet Distribution
    :param num_classes: number of classes

    :return: list of indices for each client in non-iid scenario
    """
    data_indices = list(range(len(dataset)))
    targets = np.array([dataset[i][1] for i in data_indices])
    class_indices = [np.where(targets == y)[0] for y in np.unique(targets)]
    client_indices = [[] for _ in range(num_clients)]

    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (len(class_idx) * proportions).astype(int)
        split_indices = np.split(class_idx, np.cumsum(proportions)[:-1])
        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)

    return client_indices


def dataprocessor(dataset_name="mnist", num_clients=10, batch_size=128, scenario="iid", backdoor_rates=None, target_label=None, alpha=0.5): 
    """ return processed dataset

    :param dataset_name: mnist, cifar-10 or cifar-100
    :param num_clients: number of clients
    :param batch_size: batch size for dataloader
    :param scenario: iid or non-iid
    :param backdoor_rates: percentage of data to be poisoned for each client
    :param target_label: backdoor target label, default randomly selected
    :param alpha: non-iid dirichlet parameter

    :return: processed dataloader: poisoned_train_loaders, poisoned_test_loader, clean_train_loaders, clean_test_loader
    """

    # load dataset
    if dataset_name == 'mnist': 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize MNIST images
        ])
        train_dataset = datasets.MNIST(root=DATA_ROOT+'mnist/', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=DATA_ROOT+'mnist/', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar-10': 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize CIFAR-10 images
        ])
        train_dataset = datasets.CIFAR10(root=DATA_ROOT+'cifar-10/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=DATA_ROOT+'cifar-10/', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar-100': 
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                 std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        train_dataset = datasets.CIFAR100(root=DATA_ROOT+'cifar-100/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=DATA_ROOT+'cifar-100/', train=False, download=True, transform=test_transform)
        num_classes = 100
    else: 
        exit('Error: Invalid dataset name')

    if scenario == "iid": 
        client_indices = np.array_split(np.random.permutation(len(train_dataset)), num_clients)
    elif scenario == "non_iid": 
        client_indices = split_non_iid(train_dataset, num_clients, alpha, num_classes)
    else: 
        exit('Error: Invalid scenario, only iid and non_iid supported')

    if target_label is None: target_label = np.random.randint(0, num_classes)

    num_backdoors = [int(len(client_indices[i]) * backdoor_rates[i]) for i in range(num_clients)]
    backdoor_indices = [np.random.choice(client_indices[i], num_backdoors[i], replace=False) for i in range(num_clients)] 
    clean_indices = [list(set(client_indices[i]) - set(backdoor_indices[i])) for i in range(num_clients)]

    for backdoor_idx in backdoor_indices: 
        for idx in backdoor_idx: 
            train_dataset.data[idx] = add_backdoor_trigger(train_dataset.data[idx])
            train_dataset.targets[idx] = target_label  # Add backdoor trigger label to poisoned data

    client_loaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) 
        if len(indices) else DataLoader([], batch_size=batch_size)
        for indices in client_indices
    ]

    poisoned_loaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=int(batch_size * backdoor_rates[i]), shuffle=True) 
        if len(indices) else DataLoader([], batch_size=batch_size)
        for i, indices in enumerate(backdoor_indices)
    ]

    clean_loaders = [
        DataLoader(Subset(train_dataset, indices), batch_size=int(batch_size * (1 - backdoor_rates[i])), shuffle=True) 
        if len(indices) else DataLoader([], batch_size=batch_size)
        for i, indices in enumerate(clean_indices)
    ]

    clean_test_loader = DataLoader(deepcopy(test_dataset), shuffle=True, batch_size=batch_size)
    for i in range(len(test_dataset)): 
        if test_dataset.targets[i] == target_label: continue
        test_dataset.data[i] = add_backdoor_trigger(test_dataset.data[i])
        test_dataset.targets[i] = target_label
    poisoned_test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader

    
def train(model, train_loader, optimizer, criterion):
    """ model train function(one epoch)
    
    :param model: model to train
    :param train_loader: train data loader
    :param optimizer: optimizer to use
    :param criterion: loss function to use
    :return: accuracy of the model on the training data
    """
    model.train()
    correct = 0

    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
    accuracy = correct / len(train_loader.dataset)

    return accuracy


def validate(model, test_loader, criterion):
    """ model validate function
    
    :param model: model to train
    :param test_loader: test data loader
    :param criterion: loss function to use
    :return: validation loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def test(model, test_loader): 
    """ model test function
    
    :param model: model to train
    :param test_loader: test data loader
    :return: test accuracy
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


def FedAvg(global_model: nn.Module, client_models: list[nn.Module], weights: list[float]=None, random_cl=None):
    """ Federated Average Algorithm with optional weights

    :param global_model: global model to be averaged
    :param client_models: list of client models to average
    :param weights: optional list of weights for each client model; if provided, it will be normalized
    :param random_cl: percentage of clients to be used for random sampling
    """
    global_dict = global_model.state_dict()
    
    if weights is not None:
        assert len(weights) == len(client_models), "weights length must match the number of client models."
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    elif random_cl is not None: 
        total_weight = int(random_cl * len(client_models))
        probs = np.random.choice(range(len(client_models)), size=total_weight, replace=False)
        weights = [1. if i in probs else 0. for i in range(len(client_models))]
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(client_models) for _ in client_models]

    for k in global_dict.keys():
        global_dict[k] = sum(weights[i] * client_models[i].state_dict()[k].float() for i in range(len(client_models)))

    global_model.load_state_dict(global_dict)


def get_distance(model1: nn.Module, model2: nn.Module):
    """ return l2 distance between two models parameters

    :param model1: model to compare
    :param model2: model to compare
    :return: l2 distance
    """
    with torch.no_grad():
        model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
        model2_flattened = nn.utils.parameters_to_vector(model2.parameters())

        distance = torch.square(torch.norm(model1_flattened - model2_flattened))
    return distance.item()


def get_gradients(model: nn.Module):
    """ return gradients of neural network model
    :param model: model to get gradients
    :return: gradients dictionary with parameter names as keys and gradients as values
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach()
    return gradients


def get_model_size(model: nn.Module): 
    """ return the model size(MB) """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size() 
    param_size_mb = param_size / (1024 * 1024)  # tranfer to MB
    return param_size_mb


def save_model_bydict(global_model: nn.Module, info: str, root="models/"): 
    """ save state dict of pytorch model with information """
    torch.save(global_model.state_dict(), root + info + ".pth")
    print(">>> save %s successfully!" % info)


def load_model_bydict(global_model: nn.Module, info: str, root="models/") -> nn.Module: 
    """ load state dict of pytorch model from root folder """
    global_model.load_state_dict(torch.load(root + info + ".pth"))
    print(">>> load %s successfully!" % info)
    return global_model


def save_model(global_model, info: str, root="models/"): 
    """ save pytorch model with information """
    torch.save(global_model, root + info + ".pth")
    print(">>> save %s successfully!" % info)


def load_model(global_model, info: str, root="models/"): 
    """ load pytorch model from root folder """
    global_model = torch.load(root + info + ".pth")
    print(">>> load %s successfully!" % info)
    return global_model
    

def get_reference_model(global_model: nn.Module, target_model: nn.Module, num_clients: int) -> nn.Module: 
    """ return reference model which is the average of other clients' models except target model"""
    ref_model_vec = num_clients / (num_clients - 1) * nn.utils.parameters_to_vector(global_model.parameters()) - 1 / (num_clients - 1) * nn.utils.parameters_to_vector(target_model.parameters()) 
    ref_model = deepcopy(global_model)
    nn.utils.vector_to_parameters(ref_model_vec, ref_model.parameters())
    return ref_model


def combine_dataloader(loaders: list[DataLoader], batch_size=128) -> DataLoader: 
    """ combine several dataloader into one dataloader """
    datasets = [loader.dataset for loader in loaders]
    combined_dataset = ConcatDataset(datasets)
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return combined_dataloader


def calculate_saliency_mask(grads, k=None, threshold=0.5): 
    """ calculate saliency mask with compression

    :param grads: dict, key is layer name and value is gradient
    :param k: int, number of k gradients to consider
    :param threshold: if k is None, k = threshold * d, where d is the dimension of model

    :return: dict, boolean mask of parameters saliency per layer
    """
    saliency_mask = {}

    # calculate k
    all_grads = []
    for name, grad in grads.items():
        if grad is not None:
            all_grads.append(grad.view(-1))
    all_grads_flat = torch.cat(all_grads)

    if k is None:
        k = int(all_grads_flat.numel() * threshold)
    
    _, saliency_indices = torch.topk(all_grads_flat.abs(), k)
    
    mask = torch.zeros_like(all_grads_flat)
    mask[saliency_indices] = 1

    print(f"mask shape{mask.shape}; ones: {torch.nonzero(mask).shape}")
    
    start_idx = 0
    for name, grad in grads.items():
        if grad is not None:
            numel = grad.numel()
            saliency_mask[name] = mask[start_idx:start_idx + numel].view(grad.shape)
            start_idx += numel
    
    return saliency_mask


def saliency_mask(model: nn.Module, target_loader: DataLoader, threshold=0.5): 
    """ Calculate the saliency mask for a given model based on gradients from the target_loader
    
    :param model: The neural network model for which to calculate the saliency mask
    :param target_loader: DataLoader containing the target data to compute gradients
    :param threshold: Fraction of gradients to consider for the saliency mask (default is 0.5)
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    criterion = nn.CrossEntropyLoss()
    gradients = dict([(k, torch.zeros_like(p)) for k, p in model.named_parameters()])

    model.eval()
    for data, target in target_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = -criterion(output, target)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad

    return calculate_saliency_mask(gradients, threshold=threshold)
