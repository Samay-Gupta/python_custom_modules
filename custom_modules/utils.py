import json
import os

def get_base_dir():
    return os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

def get_default_config():
    base_dir = get_base_dir()
    config_path = os.path.join(base_dir, 'storage', 'config.json')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
        config['storageDir'] = os.path.join(base_dir, config['storageDir'])
    return config

def get_requirements(submodule=None, as_list=False):
    base_dir = get_base_dir()
    if submodule:
        req_file = os.path.join(base_dir, 'custom_modules', submodule, 'requirements.txt')
    else:
        req_file = os.path.join(base_dir, 'custom_modules', 'requirements.txt')
    if not as_list:
        return req_file
    with open(req_file, 'r') as fp:
        return fp.readlines()

    

