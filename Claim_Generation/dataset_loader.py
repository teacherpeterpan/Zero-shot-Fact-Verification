"""
Tools for loading FEVER data
"""
import json
import os
from tqdm import tqdm

class FEVER_Dataset():
    def __init__(self, config):
        print('Loading FEVER >>>>>>>>')
        # load data
        with open(config.train_path, 'r') as f:
            self.FEVER_train = json.load(f)
        with open(config.dev_path, 'r') as f:
            self.FEVER_dev = json.load(f)

        print('Data loaded. ')
        print('Number of train examples: {}'.format(len(self.FEVER_train)))
        print('Number of dev examples: {}'.format(len(self.FEVER_dev)))