import os # Import the os module
from omegaconf import OmegaConf, DictConfig

# --- 1. Define the resolver function ---
# This function takes a string (the path) and expands it.
def expand_user_path(path: str) -> str:
    return os.path.expanduser(path)

# --- 2. Register the resolver with a name ---
# We'll name it 'expanduser' so we can call it in YAML.
OmegaConf.register_new_resolver("expanduser", expand_user_path)
