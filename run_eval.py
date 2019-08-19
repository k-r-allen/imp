from __future__ import (absolute_import, division, print_function,
                                                unicode_literals)

import datetime
import json
import os
import numpy as np
import six

import argparse
import os
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.configs import get_config
from fewshot.configs.mini_imagenet_config import *
from fewshot.configs.omniglot_config import *
from fewshot.configs.tiered_imagenet_config import *
from fewshot.data.data_factory import get_dataset
from fewshot.data.episode import Episode
from fewshot.data.mini_imagenet import MiniImageNetDataset
from fewshot.data.omniglot import OmniglotDataset
from fewshot.data.tiered_imagenet import TieredImageNetDataset

from fewshot.models.basic import Protonet

from fewshot.models.kmeans_refine import KMeansRefine
from fewshot.models.imp import IMPModel
from fewshot.models.map_dp import MapDPModel
from fewshot.models.softnn import SoftNN
from fewshot.models.crp import CRPModel
from fewshot.models.dp_means_hard import DPMeansHardModel
from fewshot.models.kmeans_distractor import KMeansDistractorModel
from fewshot.models.model_factory import get_model
from fewshot.utils.data_utils import *
from fewshot.utils.pytorch_utils import *
from fewshot.utils.experiment_logger import ExperimentLogger
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='Model for determining whether two faces belong to same identity')
parser.add_argument('--eval', action='store_true', default=False,
                                        help='whether to only run evaluation')
parser.add_argument('--use-test', action='store_true', default=False,
                                        help='Use the test set or not')
parser.add_argument('--disable-distractor', action='store_true', default=False,
                                        help='Whether or not to disable distractors')
parser.add_argument('--label-ratio', type=float, default=0.1, metavar='N',
                                        help='Portion of labeled images in the training set')
parser.add_argument('--mode-ratio', type=float, default=1.0, metavar='N',
                                        help='Portion of modes in the training set')
parser.add_argument('--nclasses-eval', type=int, default=5, metavar='N',
                                        help='Number of classes for testing')
parser.add_argument('--nclasses-train', type=int, default=20, metavar='N',
                                        help='Number of classes in an update')
parser.add_argument('--nsuperclassestrain', type=int, default=-1, metavar='N',
                                        help='Number of superclasses in an episode')
parser.add_argument('--nsuperclasseseval', type=int, default=-1, metavar='N',
                                        help='Number of superclasses for testing')
parser.add_argument('--nclasses-episode', type=int, default=5, metavar='N',
                                        help='Number of classes in an episode')
parser.add_argument('--accumulation-steps', type=int, default=1, metavar='N',
                                        help='Number of accumulation steps for an update')
parser.add_argument('--nshot', type=int, default=1, metavar='N',
                                        help='nshot')
parser.add_argument('--num-eval-episode', type=int, default=500, metavar='N',
                                        help='Number of evaluation episodes')
parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                                        help='Number of test images per episode')
parser.add_argument('--num-unlabel', type=int, default=5, metavar='N',
                                        help='Number of unlabeled for training')
parser.add_argument('--num-unlabel-test', type=int, default=5, metavar='N',
                                        help='Number of unlabeled for testing')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                                        help='random seed (default: 0)')
parser.add_argument('--data-root', default=None,
                                        help='Data root')
parser.add_argument('--dataset', default="omniglot",
                                        help='Dataset name')
parser.add_argument('--model', default="basic",
                                        help='Model name')
parser.add_argument('--results', default='./results',
                                        help='Checkpoint save path')
parser.add_argument('--super-classes', action='store_true', default=False,
                                        help='Use super-class labels')
parser.add_argument('--pretrain', default=None,
                                        help='folder of the model to load')
args = parser.parse_args()

free_gpu_id = get_free_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpu_id)

def _get_model(config):
    m = get_model(
                    args.model,
                    config,
                    args.dataset)
    return m.cuda()

def gen_id(config):
    return "{}_{}_{}_{}_{}-{:03d}".format(
            'label-ratio-'+str(args.label_ratio).replace(".","-"),
            'mode-ratio-'+str(args.mode_ratio).replace(".","-"),
            config.name,
            'num-unlabel-' + str(args.num_unlabel),
            datetime.datetime.now().isoformat(chr(ord("-"))).replace(":","-").replace( ".", "-"),
            int(np.random.rand() * 1000))

def evaluate(model, meta_dataset, num_episodes=500):
    all_acc = []
    model.eval()
    for neval in tqdm(six.moves.xrange(num_episodes), desc="evaluation", ncols=0):
        dataset = meta_dataset.next_episode(within_category=args.super_classes)

        batch = preprocess_batch(dataset)
 
        loss, output = model(batch, super_classes=args.super_classes)
        all_acc.append(output['acc'])   # [B, N, K]

    return {'acc': np.mean(all_acc), 'acc_ci': np.std(all_acc) * 1.96 / np.sqrt(num_episodes), 'hit': 1}

def train(config,
                    model,
                    optimizer,
                    meta_dataset,
                    meta_val_dataset=None,
                    log_results=True,
                    run_eval=True,
                    exp_id=None):

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_steps, gamma=0.5)
    if exp_id is None:
        exp_id = gen_id(config)

    save_folder = os.path.join(args.results, exp_id)
    save_config(config, save_folder)

    if args.super_classes:
        total_classes = args.nsuperclassestrain
    else:
        total_classes = args.nclasses_train

    #set up logging and printing
    if log_results:
        logs_folder = os.path.join("logs", exp_id)
        exp_logger = ExperimentLogger(logs_folder)
    it = tqdm(six.moves.xrange(args.accumulation_steps*config.max_train_steps), desc=exp_id, ncols=0)

    #Initialize for training loop
    model.train()
    time1 = time.time()
    lr = []
    clip = 1000 # for clipping loss
    best_acc = 0 # for saving best model

    #training loop
    for niter in it:

        if niter%args.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step()
            for param_group in optimizer.param_groups:
                lr += [param_group['lr']]

            dataset = meta_dataset.next_episode(within_category=args.super_classes)

        if args.accumulation_steps > 1:
            classes = np.random.choice(range(0, total_classes), args.nclasses_episode, replace=False)
            batch = dataset.next_batch_separate(classes, args.nclasses_episode)
        else:
            batch = dataset.next_batch()

        batch = preprocess_batch(batch)

        loss, output = model(batch, super_classes=args.super_classes)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(),clip)

        if (niter+1) % args.accumulation_steps == 0:
            optimizer.step()

        ##LOG and SAVE
        if (niter + 1) % (args.accumulation_steps*config.steps_per_valid) == 0 and run_eval:
            if log_results:
                exp_logger.log_learn_rate(niter, lr[-1])
            val_results = evaluate(model, meta_val_dataset, num_episodes=args.num_eval_episode)
            model.train()
            if log_results:
                exp_logger.log_valid_acc(niter, val_results['acc'])
                exp_logger.log_learn_rate(niter, lr[-1])
                val_acc = val_results['acc']
                it.set_postfix()
                meta_val_dataset.reset()

            if (niter + 1) % (args.accumulation_steps*config.steps_per_log) == 0 and log_results:
                exp_logger.log_train_ce(niter + 1, output['loss'])
                it.set_postfix(
                        ce='{:.3e}'.format(output['loss']),
                        val_acc='{:.3f}'.format(val_acc * 100.0),
                        lr='{:.3e}'.format(lr[-1]))
                print('\n')

        if (niter + 1) % (args.accumulation_steps*config.steps_per_save) == 0:
            if val_results['acc'] >= best_acc:
                best_acc = val_results['acc']
                save(model, "best", save_folder)

            save(model, niter, save_folder)

    return exp_id



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.num_test == -1 and (args.dataset == "tiered-imagenet" or
                                                             args.dataset == 'mini-imagenet'):
        num_test = 5 #to avoid too much computation
    else:
        num_test = args.num_test

    config = get_config(args.dataset, args.model)

    # Which testing split to use.
    train_split_name = 'train'
    if args.use_test:
        test_split_name = 'test'
    else:
        test_split_name = 'val'

    # Whether doing 90 degree augmentation.
    if 'omniglot' not in args.dataset:
        _aug_90 = False
    else:
        _aug_90 = True

    nshot = args.nshot
    meta_train_dataset = get_dataset(
          args,
          args.dataset,
          'train',
          args.nclasses_train,
          nshot,
          num_test=num_test,
          label_ratio=args.label_ratio,
          aug_90=_aug_90,
          num_unlabel=args.num_unlabel,
          seed=args.seed,
          mode_ratio=args.mode_ratio,
          cat_way=args.nsuperclassestrain)

    meta_test_dataset = get_dataset(
          args,
          args.dataset,
          test_split_name,
          args.nclasses_eval,
          nshot,
          num_test=num_test,
          aug_90=_aug_90,
          num_unlabel=args.num_unlabel_test,
          label_ratio=1,
          seed=args.seed,
          cat_way=args.nsuperclasseseval)

    m = _get_model(config)

    if args.eval:
        m = torch.load(
          os.path.join(args.results, args.pretrain))
    else:
        optimizer = optim.RMSprop(m.parameters(), lr=config.learn_rate, eps=1e-10, alpha=0.9, momentum=0.0)
        train(config, m, optimizer, meta_train_dataset, meta_val_dataset=meta_test_dataset)

    output = evaluate(m, meta_test_dataset, num_episodes=args.num_eval_episode)
    print(np.mean(output['acc']), (output['acc_ci']))


if __name__ == "__main__":
    print(args)
    main(args)
