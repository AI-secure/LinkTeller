from __future__ import division
from __future__ import print_function

import argparse
import datetime
import logging
import numpy as np
import random

import torch

from worker import Worker
from gcn_trainer import GCNTrainer
from utils import init_logger


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num-epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden1', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--hidden2', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset: cora; cora.sample')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--mode', type=str, default='vanilla-clean', 
                        help='[ vanilla | vanilla-clean | clusteradj | clusteradj-clean ] ')
    parser.add_argument('--init-method', type=str, default='knn',
                        help='[ naive | voting | knn | gt ]')
    parser.add_argument('--cluster-method', type=str, default='hierarchical',
                        help='[ label | random | kmeans | sskmeans ]')
    parser.add_argument('--scale', type=str, default='small',
                        help='[ large | small ]')
    parser.add_argument('--break-method', type=str, default='kmeans',
                        help='[ kmeans | dp ]')
    parser.add_argument('--norm', type=str, default='AugNormAdj',
                        choices=['AugNormAdj', 'FirstOrderGCN', 'BingGeNormAdj', 'NormAdj', 'RWalk', 'AugRWalk'])
    parser.add_argument('--sample-type', type=str, default='balanced',
                        choices=['balanced', 'unbalanced', 'unbalanced-lo', 'unbalanced-hi', 'bfs', 'balanced-full'])

    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--influence', type=float, default=0.0001)
    parser.add_argument('--train-ratio', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--n-test', type=int, default=100)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--break-ratio', type=float, default=1)
    parser.add_argument('--feature-size', type=int, default=-1)
    parser.add_argument('--k', type=float, default=1)

    parser.add_argument('--approx', action='store_true', default=False)
    parser.add_argument('--attack', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--break-down', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--same-size', action='store_true', default=False)
    parser.add_argument('--eval-degree', action='store_true', default=False)
    parser.add_argument('--trainable', action='store_true', default=False)
    parser.add_argument('--early', action='store_true', default=False)
    parser.add_argument('--fnormalize', action='store_true', default=False)

    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--cluster-seed', type=int, default=42)
    parser.add_argument('--knn', type=int, default=-1)
    parser.add_argument('--noise-type', type=str, default='laplace')
    parser.add_argument('--perturb-type', type=str, default='discrete', 
                        choices=[ 'discrete', 'continuous' ])
    parser.add_argument('--attack-mode', type=str, default='efficient',
                        choices=['efficient', 'naive', 'baseline', 'baseline-feat'])

    parser.add_argument('--coeff', type=float, default=1)
    parser.add_argument('--degree', type=int, default=2)
    parser.set_defaults(assign_seed=42)

    return parser.parse_args()


def main():
    args = get_arguments()
    print(str(args))
    logging.info(str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.test:
        worker = Worker(args, dataset=args.dataset, mode=args.mode)

        trainer = GCNTrainer(args, worker=worker)
        trainer.init_model(model_path=args.model_path)
        trainer.test(args.eval_degree)

    else:
        cur_time = datetime.datetime.now().strftime("%m-%d-%H:%M:%S.%f")

        if args.mode in ( 'sgc-clean', 'sgc' ):
            subdir = 'mode-{}_lr-{}_{}'.format(args.mode, args.lr, cur_time)

        elif args.mode in ( 'vanilla-clean', 'cs' ):
            subdir = 'mode-{}_hidden-{}_lr-{}_decay-{}_dropout-{}_norm-{}_{}'.format(\
                args.mode, args.hidden, args.lr, args.weight_decay, args.dropout, args.norm, cur_time)

        elif args.mode == 'clusteradj-clean':
            if not args.scale:
                subdir = 'mode-clusteradj-clean_{}_{}'.format(\
                    args.cluster_method, cur_time)
            else:
                subdir = 'mode-clean_small_n-clusters-{}_{}'.format(\
                    args.n_clusters, cur_time)

        elif args.mode == 'vanilla':
            subdir = 'mode-global_perturb-{}_eps-{}_{}'.format(\
                args.perturb_type, args.epsilon, cur_time)

        elif args.mode == 'clusteradj':
            if not args.scale:
                subdir = 'mode-clusteradj_ratio-{}_eps-{}_train-{}_{}_{}'.format(\
                    args.train_ratio, args.epsilon, args.trainable, args.cluster_method, cur_time)
            elif args.scale == 'small':
                subdir = 'mode-clusteradj_small_eps-{}_n-clusters-{}_{}'.format(\
                    args.epsilon, args.n_clusters, cur_time)

        elif args.mode in ( 'degree_mlp', 'basic_mlp' ):
            subdir = 'mode-{}_{}'.format(args.mode, cur_time)

        elif args.mode in ( 'degcn-clean', 'degcn' ):
            subdir = 'mode-{}_eps-{}_{}'.format(args.mode, args.epsilon, cur_time)

        else:
            print('mode={} not implemented!'.format(args.mode))
            raise NotImplementedError

        print('subdir = {}'.format(subdir))
        init_logger('./logs_{}'.format(args.dataset), subdir, print_log=False)

        worker = Worker(args, dataset=args.dataset, mode=args.mode)

        trainer = GCNTrainer(args, subdir=subdir, worker=worker)
        trainer.init_model()
        trainer.train()
        trainer.test(args.eval_degree)


if __name__ == "__main__":
    main()
