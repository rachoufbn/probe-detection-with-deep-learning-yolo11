import os, yaml
from types import MappingProxyType
from lib import get_base_dir

base_dir = get_base_dir()
config_path = os.path.join(base_dir, "config.yaml")

# Load the YAML config file
with open(config_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f) or {}

# Prevent config_data from being modified by mistake (read-only dictionary)
CONFIG = MappingProxyType(config_data)