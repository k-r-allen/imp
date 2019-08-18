#!/bin/bash
dataset="omniglot"  #dataset name
labelratio=0.4 #fraction of  labeled data
numunlabel=5 #number of unlabeled examples
nclassestrain=5 #number of classes to sample from 
nclassesepisode=5 #n-way of episode
accumulationsteps=1 #number of gradient accumulation steps before propagating loss (refer to paper)
nclasseseval=5 #n-way of test episodes
nshot=1 #shot of episodes
model="kmeans-refine" #model name
seed=0 #seed
results="tests"

python run_eval.py --data-root=/om/user/krallen/DATASETS/ --dataset=$dataset --label-ratio $labelratio --num-unlabel-test=$numunlabel --num-unlabel=$numunlabel --nclasses-train=$nclassestrain --nclasses-episode=$nclassesepisode --nclasses-eval=$nclasseseval --model $model --results $results"/"$dataset"/"$nshot"_"$nclassesepisode"/" --nshot=$nshot --seed=$seed 

