import yaml

def parse_config(config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    return config