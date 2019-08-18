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

results=$1
dataset=$2
model=$3
labelratio=$4
nclassestrain=$5
nshot=$6
numunlabel=$7
nclassesepisode=$8

python run_eval.py --data-root=/om/user/krallen/DATASETS/ --dataset=$dataset --label-ratio $labelratio --num-unlabel=$numunlabel --num-unlabel-test=$numunlabel --nclasses-train=$nclassestrain --nclasses-episode=$nclassesepisode --nclasses-eval=$nclassesepisode --model $model --results "/om/user/krallen/github/imp/"$results"/"$dataset"/"$nshot"_"$nclassesepisode"/" --nshot=$nshot --seed=0 --mode-ratio 1.0
