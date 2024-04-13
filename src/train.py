import torch
import numpy as np
import logging

from datasets.main import load_dataset
from deepSVDD import DeepSVDD
from utils.visualization.plot_images_grid import plot_images_grid

# Default device to 'cpu' if cuda is not available
def train_and_vaild_save_model(args, device):
    dataset_name = args.dataset_name
    data_path = args.data_path
    normal_class = args.normal_class
    objective = args.objective
    nu = args.nu
    net_name = args.net_name
    load_model = args.load_model
    pretrain = args.pretrain
    ae_optimizer_name = args.ae_optimizer_name
    ae_lr = args.ae_lr
    ae_n_epochs = args.ae_n_epochs
    ae_lr_milestone = args.ae_lr_milestone
    ae_batch_size = args.ae_batch_size
    ae_weight_decay = args.ae_weight_decay
    n_jobs_dataloader = args.n_jobs_dataloader
    optimizer_name = args.optimizer_name
    lr = args.lr
    n_epochs = args.n_epochs
    lr_milestone = args.lr_milestone
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    xp_path = args.xp_path
    
    
    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(objective, nu)
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        print('Loading model from %s.' % load_model)

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
    deep_SVDD.save_results(export_json=xp_path + '/{}_results.json'.format(normal_class))
    deep_SVDD.save_model(export_model=xp_path + '/{}_model.tar'.format(normal_class))
    # cfg.save_config(export_json=xp_path + '/config.json')
    
    return deep_SVDD.results['test_auc'] * 100