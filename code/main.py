import torch
import argparse
import yaml

from utils import *
from unlearning import Pretraining, SimplyRetrain, FedGradBalance
from model import MODELS_DICT


def load_yaml_config(file_path, setting):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get(setting, {})


def get_config(): 
    parser = argparse.ArgumentParser(description="Load YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--setting", type=str, required=True, help="Configuration setting (e.g., mnist_cu_iid)")
    args = parser.parse_args()

    return load_yaml_config(args.config, args.setting)


if __name__ == '__main__':
    params = get_config()

    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)  

    device = torch.device('cuda:{}'.format(params["cuda"]) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    num_clients = params["num_clients"]
    dataset_name = params["dataset"]
    batch_size = params["batch_size"]
    num_classes = params["num_classes"]
    network = MODELS_DICT[params["network"]](num_classes=num_classes)
    eraser_indices = params["eraser_idx"]
    backdoor_rate = params["backdoor_rate"]
    backdoor_rates = [backdoor_rate if i in eraser_indices else 0. for i in range(num_clients)]
    target_label = params["target_label"]
    scenario = params["scenario"]
    dirichlet = params["dirichlet"]

    # step 1: load and process dataset
    print(">>> loading dataset %s" % dataset_name)
    
    client_loaders, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader = dataprocessor(dataset_name, num_clients, batch_size=batch_size, scenario=scenario, backdoor_rates=backdoor_rates, target_label=target_label, alpha=dirichlet)

    # step 2: pretraining the federated model on the whole dataset
    global_model, client_models = Pretraining(client_loaders, clean_test_loader, poisoned_test_loader, network, params)
    
    # step 3: retraining the model from scratch
    SimplyRetrain(clean_loaders, clean_test_loader, poisoned_test_loader, network, params)

    # step 4: unlearning
    FedGradBalance(global_model, client_models, poisoned_loaders, clean_loaders, clean_test_loader, poisoned_test_loader, params, mask_threshold=0.5)
    
    
    