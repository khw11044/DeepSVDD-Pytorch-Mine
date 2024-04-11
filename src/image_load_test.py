import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset



if __name__ == '__main__':
    # Get configuration
    normal_class = 5
    dataset_name = 'cifar10'
    net_name = 'cifar10_LeNet'
    xp_path = './log/cifar10_test'
    data_path = './data'
    objective = 'one-class'     # soft-boundary
    lr = 0.0001
    ae_n_epochs = 5 # 350 
    n_epochs = 5 # 150
    lr_milestone = [25,50]
    batch_size = 512 # 200
    weight_decay = 0.5e-6
    pretrain = True
    ae_lr = 0.0001
    ae_lr_milestone = [25,50] # 250
    ae_batch_size = 512 # 200
    ae_weight_decay = 0.5e-6
    load_config = None
    nu = 0.1
    seed = -1
    n_jobs_dataloader = 0
    load_model = None
    ae_optimizer_name = 'adam'  # amsgrad
    optimizer_name = 'adam' # amsgrad
    

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)
    train_loader, _ = dataset.loaders(batch_size=100, num_workers=0)
    _, test_loader = dataset.loaders(batch_size=100, num_workers=0)

    idx_sorted = list(range(100))
    X_normals = torch.tensor(np.transpose(train_loader.dataset.dataset.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
    X_outliers = torch.tensor(np.transpose(test_loader.dataset.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

    plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
    plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)


