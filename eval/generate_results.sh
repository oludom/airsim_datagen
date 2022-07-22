#!/bin/bash

# run "python3 generate_tum_trajectory_files.py" first to generate tum dataset files for the data set
# then you can evaluate the poses flown by the drone with the generated min snap trajectory as ground truth
# this script requires evo library to be installed:
# https://github.com/MichaelGrupp/evo


# dataset folder
DATASETS=/media/micha/eSSD/datasets
DFOLDER=/X4Gates_Circles
#DATASETS=/home/micha/dev/datasets/droneracing
#DFOLDER=/X4Gates_Circle_1
#DATASETS=/data/datasets
#DFOLDER=/X4Gates_Circle_2
DATASET="$DATASETS$DFOLDER"

# enable wildcards
shopt -s globstar
RESULTS="$DATASETS"/results"$DFOLDER"
mkdir $DATASETS/results
mkdir $RESULTS

for d in "$DATASET"/*/ ; do

  TRACKNAME=$(basename $d)

  evo_ape tum "$d"groundtruth_trajectory_"$TRACKNAME".tum "$d"actual_trajectory_"$TRACKNAME".tum --save_results "$RESULTS"/"$TRACKNAME".zip
done

# compare and save results
evo_res $RESULTS/*.zip -p --save_table "$RESULTS"/table.csv