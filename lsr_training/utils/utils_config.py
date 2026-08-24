import os
from pathlib import Path

import yaml

def load_config(config_path, save_path=None):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get('seed') is None:
        config['seed'] = None
    if save_path is None:
        save_path = Path('outputs') / 'lsr' / Path(config_path).stem
    save_path = os.fspath(Path(save_path).expanduser().resolve())
    config['save_path'] = save_path
    config['resume_path'] = os.path.join(save_path, 'iter_last.pth')
    return config
