import subprocess
import json
import sys
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

def create_cv_shortcut(envname: str='venv'):
    shell = os.environ.get('SHELL', '')
    if 'bash' in shell:
        profile_file = os.path.expanduser('~/.bashrc')
    elif 'zsh' in shell:
        profile_file = os.path.expanduser('~/.zshrc')
    else:
        print("Unsupported shell. Manual alias setup required.")
        return
    base_dir = get_base_dir()
    envpath = os.path.join(base_dir, envname)
    subprocess.check_call([sys.executable, '-m', 'venv', envpath])
    pip_executable = os.path.join(envpath, 'bin', 'pip')
    req_file = get_requirements('computer_vision', as_list=False)
    subprocess.check_call([pip_executable, 'install', '-r', req_file])
    alias_cmd = "\nexport cv() { " + \
        f"source {os.path.join(envpath, 'bin', 'activate')}; " + \
        f"python {os.path.join(base_dir, 'custom_modules', 'computer_vision', 'cli.py')} $@; " + \
        "deactivate }\n"
    with open(profile_file, 'a') as profile:
        profile.write(alias_cmd)

    print(f"Alias 'cv' created. Please restart the terminal or source the profile file.")

