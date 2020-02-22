from __future__ import division
from __future__ import print_function

import argparse
import datetime
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from cluster_worker import ClusterWorker
from gcn_trainer import GCNTrainer
from labeler import Labeler
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
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset: cora; cora.sample')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--mode', type=str, default='vanilla-clean', 
                        help='[ raw | global | jl | clusteradj | clusteradj-clean ] ')
    parser.add_argument('--init-method', type=str, default='knn',
                        help='[ naive | voting | knn | gt ]')
    parser.add_argument('--cluster-method', type=str, default='hierarchical',
                        help='[ label | random | kmeans | sskmeans ]')
    parser.add_argument('--scale', type=str, default='',
                        help='[ large | small ]')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--r', type=int, default=1,
                        help='parameter for JL')
    parser.add_argument('--train-ratio', type=float, default=0.5)
    parser.add_argument('--stop-num', type=int, default=50)
    parser.add_argument('--n-clusters', type=int, default=10)
    parser.add_argument('--break-ratio', type=float, default=1)
    parser.add_argument('--feature-size', type=int, default=-1)

    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--break-down', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--same-size', action='store_true', default=False)
    parser.add_argument('--eval-degree', action='store_true', default=False)

    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--cluster-seed', type=int, default=42)
    parser.add_argument('--knn', type=int, default=-1)
    parser.add_argument('--noise-type', type=str, default='laplace')
    parser.set_defaults(assign_seed=42)

    return parser.parse_args()


def main():
    args = get_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.test:
        worker = ClusterWorker(args, dataset=args.dataset, mode=args.mode)

        trainer = GCNTrainer(args, worker=worker)
        trainer.init_model(model_path=args.model_path)
        trainer.test(args.eval_degree)

    else:
        cur_time = datetime.datetime.now().strftime("%m-%d-%H:%M:%S.%f")[:-3]

        if args.mode == 'vanilla-clean':
            subdir = 'mode-raw_ratio-{}_{}'.format(args.train_ratio, cur_time)

        elif args.mode == 'clusteradj-clean':
            subdir = 'mode-clusteradj-clean_ratio-{}_{}_{}'.format(\
                args.train_ratio, args.cluster_method, cur_time)

        elif args.mode == 'vanilla':
            subdir = 'mode-global_ratio-{}_eps-{}_{}'.format(\
                args.train_ratio, args.epsilon, cur_time)

        elif args.mode == 'clusteradj':
            subdir = 'mode-clusteradj_ratio-{}_eps-{}_{}_{}'.format(\
                args.train_ratio, args.epsilon, args.cluster_method, cur_time)

        elif args.mode in ( 'jl' ):
            subdir = 'mode-jl_eps-{}_del-{}_r-{}_{}'.format(\
                args.epsilon, args.delta, args.r, cur_time)

        else:
            print('mode={} not implemented!'.format(args.mode))
            raise NotImplementedError

        print('subdir = {}'.format(subdir))
        init_logger('./logs_{}'.format(args.dataset), subdir, print_log=False)
        logging.info(str(args))

        worker = ClusterWorker(args, dataset=args.dataset, mode=args.mode)

        trainer = GCNTrainer(args, subdir=subdir, worker=worker)
        trainer.init_model()
        trainer.train()
        trainer.test(args.eval_degree)


if __name__ == "__main__":
    main()
