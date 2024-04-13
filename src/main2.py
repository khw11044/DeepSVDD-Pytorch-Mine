import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from collections import defaultdict

from train import train_and_vaild_save_model

import easydict 
args = easydict.EasyDict({
        'dataset_name':'cifar10',
        'net_name':'cifar10_LeNet_ELU',  # cifar10_LeNet, cifar10_LeNet_ELU,    'mnist_LeNet'      
        'xp_path' : './log/cifar10_test',   # './log/mnist_test' 
        'data_path' : './data',
        'objective' : 'one-class',     # soft-boundary
        'lr':0.0001,
        'ae_lr':0.0001,
        'ae_n_epochs':100,
        'n_epochs':100,
        'weight_decay':5e-7,
        'ae_weight_decay':5e-3,
        'lr_milestone':[90],
        'ae_lr_milestone': [90], # 250
        'ae_batch_size':2048,
        'batch_size':2048,
        'nu' : 0.1,
        'seed' : -1,
        'n_jobs_dataloader' : 0,
        'load_model' : None,
        'ae_optimizer_name': 'adam',  # amsgrad
        'optimizer_name': 'adam', # amsgrad
        'pretrain':True,         # 항상 True
        'normal_class': 0,        # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        'AutoEncoder_path':'./weights/pretrained_AE.pth',
        'save_path':'./weights/'
                })

if __name__ == '__main__':

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = args.xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    
    # Set seed
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info('Set seed to %d.' % args.seed)

    # Default device to 'cpu' if cuda is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = defaultdict()
    for i in range(10):
        args.normal_class = i
        auc = train_and_vaild_save_model(args, device)
        print(classes[i] , 'Test set AUC: {:.2f}%'.format(auc))
        results[classes[i]] = '{:.2f}%'.format(auc)
        print(results)
    
    print(results)

'''
airplane : 57.82%
automobile : 60.60%
bird : 45.53% 
cat : 60.26%
deer : 54.63%
dog : 64.41%
frog : 57.58%
horse : 56.78%
ship : 76.06%
truck : 67.50%

'''