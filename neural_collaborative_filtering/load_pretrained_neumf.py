import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import keras
from evaluate import evaluate_model
from Dataset import Dataset
from keras.optimizers import Adam
from NeuMF import get_model

# Dataset
dataset = Dataset("Data/ml-100k")
train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape

# Pre-trained NeuMF parameters
model_path = "Pretrain/ml-100k_NeuMF_16_[32,16]_best_run9.h5"
mf_dim = 16
layers = [32, 16]
reg_layers = [0, 0]
reg_mf = 0
learning_rate = 0.001

# Recreate model architecture
model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)

# Compile model
model.compile(optimizer=Adam(lr=learning_rate),
              loss='binary_crossentropy')

# Load saved weights
model.load_weights(model_path)
print("Weights loaded successfully")

# Evaluation params
topK = 10
evaluation_threads = 1
hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

# Evaluate top-K, for k=1..10:
topK_hrs = []
topK_ndcgs = []
for topK in range(1, 11):
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    topK_hrs.append(hr)
    topK_ndcgs.append(ndcg)

print(f"{topK_hrs}")
print(f"{topK_ndcgs}")