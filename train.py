import os
import yaml
import torch
import shutil
import random
import argparse
import numpy as np
from easydict import EasyDict

from modules.trainer import Trainer

DATASET_PATH = './Building3D_entry_level/Entry-level/' # ohter option "Tallinn/"
CONFIG_PATH = './Building3D/datasets/dataset_config.yaml'

def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    print(f'Using seed: {seed}')

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg

if __name__ == '__main__':
    seed_everything()

    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        default=DATASET_PATH,
        help='Path to the dataset',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default=CONFIG_PATH,
        help='Path to dataset config file',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.getcwd() + '/log/log' + '/',
        help='Directory to put the log data. Default: ./log/log'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("config", FLAGS.config)
    print("log", FLAGS.log)
    print("----------\n")
    
    # open config file
    dataset_config = cfg_from_yaml_file(FLAGS.config)
    dataset_config['Building3D']['root_dir'] = FLAGS.dataset

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    trainer = Trainer(dataset_config=dataset_config, logdir=FLAGS.log)
    trainer.train()