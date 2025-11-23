# Federated Unlearning with Oriented Saliency Compression

🔗 paper link: [*Federated Unlearning with Oriented Saliency Compression*](https://ieeexplore.ieee.org/document/11228643/)

---

## Prerequisites

Before running the code, ensure you have the following libraries installed:

```bash
pip install torch torchvision numpy pyyaml
```

## File Struction

```
FedUOSC/
├── config/
│   ├── [dataset.yaml]      # configuration files
├── code/
│   ├── [model.py]          # Model definitions
│   ├── [utils.py]          # Utility functions
│   ├── [unlearning.py]     # Unlearning methods
│   ├── [main.py]           # Main script
├── [README.md]             
```

## How to Use
1. Configuration File: prepare a `.yaml` configuration file before running the program or you can use the default file in `./config`, where `cu` means client removal and `su` means sample removal in federated system. 
2. Run the Program: execute the following command to start the program:

    ```
    python code/main.py --config <path_to_config_file> --setting <configuration_name>
    ```

    for example: 

    ```
    python code/main.py --config config/cifar10.yaml --setting cu_iid
    ```

## Notes
- Datasets will be automatically downloaded to the specified path (change your dataset path in `./code/utils.py`).
- Adjust the parameters in the configuration file as needed.

