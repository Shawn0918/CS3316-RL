#!/bin/bash

# This script runs the experiment with two agents (TD3 and DDPG) on four MuJoCo environments (Hopper, Humanoid, HalfCheetah, and Ant).
# To run the script, just enter this directory and type "bash run.sh" in the terminal.

envs="Hopper-v2 Humanoid-v2 HalfCheetah-v2 Ant-v2"
model="TD3 DDPG"
for env in $envs
do
    for m in $model
    do
        python train_modified.py --env_name $env --seed 42 --start_timesteps 10000 --max_timesteps 2000000 --model_path ./models0607 --res_dir ./results0607 --agent $m
    done
done

