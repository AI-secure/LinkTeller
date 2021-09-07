from __future__ import division
from __future__ import print_function

import argparse
import datetime
import logging
import numpy as np

import torch

from worker import Worker
from mlp_trainer import MLPTrainer
from utils import init_logger


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num-epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset: cora; cora.sample')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--mode', type=str, default='sgc', 
                        help='[ vanilla | vanilla-clean | clusteradj | clusteradj-clean ] ')
    parser.add_argument('--scale', type=str, default='small',
                        help='[ large | small ]')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['AugNormAdj'])

    parser.add_argument('--l2-norm-clip', type=float, default=1., help='upper bound on the l2 norm of gradient updates (default: 0.1)')
    parser.add_argument('--noise-multiplier', type=float, default=1.1, help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    parser.add_argument('--microbatch-size', type=int, default=1, help='input microbatch size for training (default: 1)')
    parser.add_argument('--minibatch-size', type=int, default=256, help='input minibatch size for training (default: 256)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')

    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--train-ratio', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--k', type=float, default=1)

    parser.add_argument('--batch', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--trainable', action='store_true', default=False)
    parser.add_argument('--early', action='store_true', default=False)

    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--cluster-seed', type=int, default=42)
    parser.add_argument('--noise-type', type=str, default='gaussian')
    parser.add_argument('--trainer-type', type=str, default='mlp', 
                        choices=['mlp', 'dp_mlp', 'lr', 'dp_lr', 'multi_mlp', 'dp_multi_mlp'])
    parser.add_argument('--feature-size', type=int, default=-1)
    parser.add_argument('--iterations', type=int, default=14000, help='number of iterations to train (default: 14000)')
    parser.add_argument('--coeff', type=float, default=1)
    parser.add_argument('--degree', type=int, default=2)
    parser.set_defaults(assign_seed=42)

    return parser.parse_args()


def main():
    args = get_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.test:
        worker = Worker(args, dataset=args.dataset, mode=args.mode)

        trainer = MLPTrainer(args, worker=worker, train=False)
        trainer.init_model(model_path=args.model_path)
        trainer.test()

    else:
        cur_time = datetime.datetime.now().strftime("%m-%d-%H:%M:%S.%f")

        subdir = 'mode-{}_hidden-{}_lr-{}_decay-{}_dropout-{}_bs-{}_{}'.format(args.mode, args.hidden, args.lr, args.weight_decay, args.dropout, args.batch_size, cur_time)

        print('subdir = {}'.format(subdir))
        init_logger('./logs_{}'.format(args.dataset), subdir, print_log=False)
        logging.info(str(args))

        worker = Worker(args, dataset=args.dataset, mode=args.mode)

        trainer = MLPTrainer(args, subdir=subdir, worker=worker, train=True)
        trainer.init_model()
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    main()
