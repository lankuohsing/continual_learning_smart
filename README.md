## Schematic Memory Persistence and Transience for Efficient and Robust Continual Learning

Code for paper:
Schematic Memory Persistence and Transience for Efficient and Robust Continual Learning
Yuyang Gao, Giorgio A. Ascoli, Liang Zhao
Neural Network - Special Issue on Artificial Intelligence and Brain Science


## (key) Requirements 

- Python 2.8 or more.
- Pytorch 1.1.0

`pip install -r requirements.txt`
or
`conda install --file requirements.txt`


## How to run

For run the disjoint MNIST exerpiement please run ./Disjoint_Mnist.sh

or directly run the following command:
$ python main.py --n_layers 2 --n_hiddens 100 --data_path ./data/ --save_path ./results/Disjoint_Mnist_5/100/ --log_every 100 --samples_per_task 1000 --data_file mnist_split.pt --cuda yes --tasks_to_preserve 5 --model SMART --slack 0 --change_th 0.0 --repass 0 --eval_memory yes --normalize no --seed 0 --subselect 1 --age 0 --lr 0.0001 --n_sampled_memories 100 --memory_strength 50 --batch_size 50 --n_memories 50 --n_constraints 50 --n_iter 100 --reg_factor 0.005 --robust 0


## optional arguments for main.py

  -h, --help            show this help message and exit
  --model MODEL         choose the model to train
  --n_hiddens N_HIDDENS
                        number of hidden neurons at each layer
  --n_layers N_LAYERS   number of hidden layers
  --shared_head SHARED_HEAD
                        shared head between tasks
  --bias BIAS           add bias
  --reg_factor REG_FACTOR
                        The hyper-parameter for controlling of feature
                        sparsity regularization
  --robust ROBUST       whether to use the proposed robust regularization
  --nac_order NAC_ORDER
                        The neuronal correlation order for NAC regularization
  --n_memories N_MEMORIES
                        number of memories per task
  --n_sampled_memories N_SAMPLED_MEMORIES
                        number of sampled_memories per task
  --n_constraints N_CONSTRAINTS
                        number of constraints to use during online training
  --b_rehearse B_REHEARSE
                        if 1 use mini batch while rehearsing
  --tasks_to_preserve TASKS_TO_PRESERVE
                        number of tasks to preserve
  --change_th CHANGE_TH
                        gradients similarity change threshold for re-
                        estimating the constraints
  --slack SLACK         slack for small gradient norm
  --normalize NORMALIZE
                        normalize gradients before selection
  --memory_strength MEMORY_STRENGTH
                        memory strength (meaning depends on memory)
  --finetune FINETUNE   whether to initialize nets in indep. nets
  --n_epochs N_EPOCHS   Number of epochs per task
  --n_iter N_ITER       Number of iterations per batch
  --repass REPASS       make a repass over the previous data
  --batch_size BATCH_SIZE
                        batch size
  --mini_batch_size MINI_BATCH_SIZE
                        mini batch size
  --lr LR               SGD learning rate
  --cuda CUDA           Use GPU?
  --seed SEED           random seed
  --log_every LOG_EVERY
                        frequency of logs, in minibatches
  --save_path SAVE_PATH
                        save models at the end of training
  --output_name OUTPUT_NAME
                        special output name for the results?
  --data_path DATA_PATH
                        path where data is located
  --data_file DATA_FILE
                        data file
  --samples_per_task SAMPLES_PER_TASK
                        training samples per task (all if negative)
  --shuffle_tasks SHUFFLE_TASKS
                        present tasks in order
  --eval_memory EVAL_MEMORY
                        compute accuracy on memory
  --age AGE             consider age for sample selection
  --subselect SUBSELECT
                        first subsample from recent memories