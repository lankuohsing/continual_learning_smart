#!/usr/bin/bash
MY_PYTHON="python"

# config
batchsize=50
num_iter=100
n_task=5
memory_size=100
n_sample_per_task=1000
learning_rate=0.0001
results="./results/Disjoint_Mnist_5/100/"
data_path="mnist_split.pt"

# number of seed to run
nb_seeds=3

seed=1
while [ $seed -le $nb_seeds ]
do
    echo $seed

    echo "***********************SMART***********************"
    $MY_PYTHON main.py --n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path $results --log_every 100 --samples_per_task $n_sample_per_task --data_file $data_path --cuda yes --tasks_to_preserve $n_task --model SMART --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize no --seed $seed --subselect 1 --age 0 --lr $learning_rate --n_sampled_memories $memory_size --memory_strength $batchsize --batch_size $batchsize --n_memories $batchsize --n_constraints $batchsize --n_iter $num_iter --reg_factor 0.005 --robust 0
    ((seed++))
done


