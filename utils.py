import yaml


def read_yaml_file(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_file


def save_yaml_file(file_path: str, content):
    with open(file_path, encoding="utf-8", mode="w") as file_open:
        yaml_file = yaml.dump(content, file_open)