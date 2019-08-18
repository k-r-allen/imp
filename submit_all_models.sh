#!/bin/bash

dataset="omniglot"
results="results"
labelratio=0.4
numunlabel=5
classestotal=5

models=("imp")
# "soft-nn" "dp-means-hard" "crp" "kmeans-distractor" "kmeans-refine")
for j in ${!models[@]}; do
	sbatch submit_prototypes.sbatch $results $dataset ${models[$j]} $labelratio $classestotal 1 $numunlabel 20
	sbatch submit_prototypes.sbatch $results $dataset ${models[$j]} $labelratio $classestotal 5 $numunlabel 20
	sbatch submit_prototypes.sbatch $results $dataset ${models[$j]} $labelratio $classestotal 1 $numunlabel 5
	sbatch submit_prototypes.sbatch $results $dataset ${models[$j]} $labelratio $classestotal 5 $numunlabel 5
done
