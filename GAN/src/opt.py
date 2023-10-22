import yaml
import argparse
import os

class Parser():
    def __init__(self, config_file='config.yml'):
        if not os.path.isfile(config_file):
            raise FileNotFoundError('Config File:', config_file, 'does not exist.')
        self._config = {}
        with open(config_file, 'r') as fh:
            config_yml = yaml.load(fh, yaml.FullLoader)
            self.load(config_yml)
        self.parse()
    
    def load(self, config_yml):
        for big_cat, val in config_yml.items():
            for key, val in val.items():
                self._config[key] = val

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gen', action='store_true', help='generate pictures')
        parser.add_argument('--no_crawl', action='store_true')
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--gan', type=str, default='dcgan')
        self._args = parser.parse_args()

    @property
    def config(self):
        return self._config
    
    @property
    def args(self):
        return self._args