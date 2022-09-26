import yaml
import dotmap


class Configuration:
    _BASE_CONFIG_PATH = "./configs/base.yaml"

    def __init__(self):
        self.base = Configuration.load_base()

    @staticmethod
    def load_base() -> dotmap.DotMap:
        try:
            with open(Configuration._BASE_CONFIG_PATH, mode='r') as file:
                content = yaml.load(stream=file, Loader=yaml.FullLoader)
                return dotmap.DotMap(content)
        except IOError:
            raise IOError

