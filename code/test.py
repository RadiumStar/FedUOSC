import argparse
import yaml

def load_yaml_config(file_path, key):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get(key, {})

def main():
    parser = argparse.ArgumentParser(description="Load YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--key", type=str, required=True, help="Configuration key (e.g., mnist_cu_iid)")
    args = parser.parse_args()

    config = load_yaml_config(args.config, args.key)
    print(config)

if __name__ == "__main__":
    main()