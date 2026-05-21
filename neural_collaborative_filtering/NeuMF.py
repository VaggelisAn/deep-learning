'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np

import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import json
import GMF, MLP
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='/Data',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of independent training runs to execute.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while train[u, j] != 0:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return np.array(user_input, dtype='int32'), np.array(item_input, dtype='int32'), np.array(labels, dtype='int32')

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1 # mp.cpu_count()
    print("NeuMF arguments: %s " %(args))

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    session_bests = []
    overall_best_hr = -1.0
    overall_best_ndcg = -1.0
    overall_best_iter = -1
    overall_best_run = -1
    overall_best_weights = None

    for run_id in range(1, args.runs + 1):
        print('\n===== Training Session %d/%d =====' % (run_id, args.runs))
        model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        if mf_pretrain != '' and mlp_pretrain != '':
            gmf_model = GMF.get_model(num_users,num_items,mf_dim)
            gmf_model.load_weights(mf_pretrain)
            mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
            mlp_model.load_weights(mlp_pretrain)
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
            print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))

        # Initial performance before training
        init_hits, init_ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        init_hr, init_ndcg = np.array(init_hits).mean(), np.array(init_ndcgs).mean()
        print('Session %d init: HR = %.4f, NDCG = %.4f' % (run_id, init_hr, init_ndcg))

        epoch_HRs = []
        epoch_NDCGs = []
        best_hr, best_ndcg, best_iter = init_hr, init_ndcg, -1

        for epoch in range(num_epochs):
            t1 = time()
            user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
            hist = model.fit([user_input, item_input],
                             labels,
                             batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
            t2 = time()

            if epoch % verbose == 0:
                hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                epoch_HRs.append(hr)
                epoch_NDCGs.append(ndcg)
                print('Session %d Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                      % (run_id, epoch, t2-t1, hr, ndcg, loss, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch

        print('Session %d HRs: %s' % (run_id, json.dumps(epoch_HRs)))
        print('Session %d NDCGs: %s' % (run_id, json.dumps(epoch_NDCGs)))
        print('Session %d Best HR = %.4f, Best NDCG = %.4f, Best Iteration = %d' % (run_id, best_hr, best_ndcg, best_iter))

        session_bests.append((best_hr, best_ndcg))
        if best_hr > overall_best_hr:
            overall_best_hr = best_hr
            overall_best_ndcg = best_ndcg
            overall_best_iter = best_iter
            overall_best_run = run_id
            overall_best_weights = model.get_weights()

    if session_bests:
        best_HRs = [x[0] for x in session_bests]
        best_NDCGs = [x[1] for x in session_bests]
        print('\n===== Aggregate Best Metrics =====')
        print('Best HRs: %s' % json.dumps(best_HRs))
        print('Best NDCGs: %s' % json.dumps(best_NDCGs))
        if len(best_HRs) > 1:
            print('Best HR mean: %.6f, std: %.6f' % (np.mean(best_HRs), np.std(best_HRs, ddof=0)))
            print('Best NDCG mean: %.6f, std: %.6f' % (np.mean(best_NDCGs), np.std(best_NDCGs, ddof=0)))
        else:
            print('Best HR mean: %.6f, std: %.6f' % (best_HRs[0], 0.0))
            print('Best NDCG mean: %.6f, std: %.6f' % (best_NDCGs[0], 0.0))

        if args.out > 0 and overall_best_weights is not None:
            best_model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
            if learner.lower() == "adagrad":
                best_model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "rmsprop":
                best_model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "adam":
                best_model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
            else:
                best_model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
            best_model.set_weights(overall_best_weights)
            model_out_file = 'Pretrain/%s_NeuMF_%d_%s_best_run%d.h5' % (args.dataset, mf_dim, args.layers, overall_best_run)
            best_model.save_weights(model_out_file, overwrite=True)
            print('Overall best model saved to %s' % model_out_file)
