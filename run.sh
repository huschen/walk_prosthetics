#!/bin/bash

#export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; mpirun --use-hwthread-cpus -bind-to core -n 6 \
python main.py --nb_epochs 700 --nb_epoch_cycles 20 --nb_rollout_steps 150 --batch_size 128 --no_sanity_run
