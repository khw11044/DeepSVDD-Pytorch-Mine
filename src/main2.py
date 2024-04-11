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
    normal_class = 1
    dataset_name = 'cifar10'        # mnist, cifar10
    net_name = 'cifar10_LeNet_ELU'      # mnist_LeNet, cifar10_LeNet, cifar10_LeNet_ELU
    xp_path = './log/mnist_test'  # ./log/mnist_test   ./log/cifar10_test
    data_path = './data'
    objective = 'one-class'     # soft-boundary
    lr = 0.0001
    ae_n_epochs = 500 # 350     # mnist 100으로 해도 충분함 
    n_epochs = 500 # 150
    ae_lr_milestone = [350, 450] # 250
    lr_milestone = [350, 450]
    ae_batch_size = 2048 # 200
    batch_size = 2048 # 200
    weight_decay = 0.5e-6
    pretrain = True
    ae_lr = 0.0001
    ae_weight_decay = 0.5e-6
    load_config = None
    nu = 0.1
    seed = -1
    n_jobs_dataloader = 0
    load_model = None
    ae_optimizer_name = 'adam'  # amsgrad
    optimizer_name = 'adam' # amsgrad
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    
    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info('Set seed to %d.' % seed)

    # Default device to 'cpu' if cuda is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(objective, nu)
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    if pretrain:
        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=ae_optimizer_name,
                           lr=ae_lr,
                           n_epochs=ae_n_epochs,
                           lr_milestones=ae_lr_milestone,
                           batch_size=ae_batch_size,
                           weight_decay=ae_weight_decay,
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)


    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=optimizer_name,
                    lr=lr,
                    n_epochs=n_epochs,
                    lr_milestones=lr_milestone,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
    ab_idx_sorted = indices[labels == 1][np.argsort(scores[labels == 1])]  # sorted from lowest to highest anomaly score
    # normal로 평가한 
    if dataset_name in ('mnist', 'cifar10'):

        if dataset_name == 'mnist':
            X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.test_data[ab_idx_sorted[-32:], ...].unsqueeze(1)
            X_train = dataset.train_set.dataset.data[:32].unsqueeze(1)

        if dataset_name == 'cifar10':           # dataset.train_set.dataset.data
            X_normals = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.test_set.data[ab_idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_train = torch.tensor(np.transpose(dataset.train_set.dataset.data[:32], (0, 3, 1, 2)))

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)
        plot_images_grid(X_train, export_img=xp_path + '/X_train', title='Most anomalous examples', padding=2)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar')
    # cfg.save_config(export_json=xp_path + '/config.json')
