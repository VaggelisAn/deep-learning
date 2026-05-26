#!/bin/bash
# conda activate dl

KERAS_HOME=/home/vag/deep_learning/keras.json

# Reduce noisy warnings from Theano/TensorFlow
export THEANO_FLAGS="cxx="
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"

# Parameters:
# Predictive factors J: 8/ 16, 8 default
PREDICTIVE_FACTORS=16
# MLP Layers: [J*4->J*3->J*2->J] (3 MLP layers), [J*3->J*2->J] (2 MLP layers), [J*2->J] (1 MLP layer)
LAYERS=[128,64,32,16]
# Negative sampling ratio: 1-10 (4 default)
NUM_NEG=4
# Batch size: 128/ 256 (default)
BATCH_SIZE=512

# Constants:
# epochs: not a lot needed
EPOCHS=15
# Learning rate: 0.001
LR=0.001
# Optimizer: adam
OPTIMIZER=adam 
PATH=Data/
DATASET=ml-100k
# Regularization for each MLP layer: must match number of layers above
REG_LAYERS=[0,0,0,0]

# NCF intialization: Gaussian
# a: 0.5

printf "\n\n\n------------------------------\nRunning NeuMF with the following parameters:\n"
printf "J = $PREDICTIVE_FACTORS\nMLP Layers = $LAYERS\nNegative samling ratio = $NUM_NEG\nBatch Size = $BATCH_SIZE\n"
printf "Epochs = $EPOCHS\nLearning Rate = $LR\nOptimizer = $OPTIMIZER\n"
/home/vag/miniconda3/envs/dl/bin/python -u MLP.py \
  --path="$PATH" \
  --dataset="$DATASET" \
  --epochs="$EPOCHS" \
  --batch_size="$BATCH_SIZE" \
  --layers="$LAYERS" \
  --reg_layers="$REG_LAYERS" \
  --num_neg="$NUM_NEG" \
  --lr="$LR" \
  --learner="$OPTIMIZER"
printf "finishdddd\n---------------------------------------\n"