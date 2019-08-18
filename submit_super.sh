#!/bin/bash
dataset="tiered-imagenet"
labelratio=1.0
numunlabel=0

nsub=5
nsuper=5
nshot=2

sbatch submit_super.sbatch 'kmeans-refine' $dataset $nsuper $nsub $nshot
sbatch submit_super.sbatch 'imp' $dataset $nsuper $nsub $nshot
sbatch submit_super.sbatch 'soft-nn' $dataset $nsuper $nsub $nshot

dataset="omniglot"

nsub=10
nsuper=10
nshot=1

sbatch submit_super.sbatch 'kmeans-refine' $dataset $nsuper $nsub $nshot
sbatch submit_super.sbatch 'imp' $dataset $nsuper $nsub $nshot
sbatch submit_super.sbatch 'soft-nn' $dataset $nsuper $nsub $nshot
