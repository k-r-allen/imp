#!/bin/bash
#
# prototypicalNets.sbatch
#
#SBATCH -J prototypes        # A single job name for the array
#SBATCH -n 5              # one core
#SBATCH -N 1              # on one node
#SBATCH --mem=15G        # Memory request of 4 GB
#SBATCH --time=24:00:00 #time limit of 24 hours
#SBATCH --gres=gpu:titan-x:1

model=$1
dataset=$2
nsuper=$3
nsub=$4
nshot=$5

python run_eval.py --super-classes --nsuperclassestrain=$nsuper --nsuperclasseseval=$nsuper --data-root=/om/user/krallen/DATASETS/ --dataset=$dataset --label-ratio 1.0 --num-unlabel=0 --num-unlabel-test=0 --nclasses-train=$nsub --nclasses-episode=$nsub --nclasses-eval=$nsub --model $model --results "/om/user/krallen/github/imp/super_classes/"$dataset"/"$nshot"_"$nsub"/" --nshot=$nshot --seed=0 --mode-ratio 1.0
