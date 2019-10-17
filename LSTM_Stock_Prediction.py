#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fix random value
from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

# # 0. Load packages to be used
import keras
import numpy as np
from keras.utils import np_utils
from keras.utils import plot_model
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Concatenate
from keras.layers import Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.metrics import *
from sklearn.metrics import precision_recall_fscore_support
import datetime
import time
from keras.models import load_model
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

import os.path
from pandas import Series, DataFrame

import gc

# Packages related on plotting graphs
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

import matplotlib.dates as mdates
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
import PIL.Image as pilimg
import csv
import os
print(keras.__version__)

#####################################################################################################################
#####################################################################################################################
# Case applying only filtering in SVM, MLP 
def seq2filtered(seq, seq_label, seq_index, seq_true, window_size):
    x = []
    y = []
    index = []
    true_list = []
    for i in range(window_size - 1, seq.shape[0]):
        if (target_mode == 2 and Candidate_list[i] == 1) or (
                target_mode == 3 and Candidate_list[i] == -1) or (
                target_mode == 4 and Candidate_list[i] != 0):
            subset = seq[i]
            x.append(subset)
            y.append(seq_label[i])
            index.append(seq_index[i])
            true_list.append(seq_true[i])
    return np.array(x), np.array(y), index, true_list

def seq2dataset(seq, seq_label, window_size):
    x = []
    y = seq_label[window_size - 1:]
    for i in range(seq.shape[0] - window_size + 1):
        subset = seq[i:(i + window_size), :]
        x.append(subset)
    return np.array(x), y


def seq2dataset_filter(seq, seq_label, seq_index, seq_true, window_size):
    x = []
    y = []
    index = []
    true_list = []
    for i in range(seq.shape[0] - window_size + 1):
        if (target_mode == 2 and Candidate_list[i + window_size - 1] == 1) or (
                target_mode == 3 and Candidate_list[i + window_size - 1] == -1) or (
                target_mode == 4 and Candidate_list[i + window_size - 1] != 0):
            subset = seq[i:(i + window_size), :]
            x.append(subset)
            y.append(seq_label[i + window_size - 1])
            index.append(seq_index[i + window_size - 1])
            true_list.append(seq_true[i + window_size - 1])
    return np.array(x), np.array(y), index, true_list


#####################################################################################################################
#####################################################################################################################
## Set environment and parameters.
target_mode = 4 # Calculation mode for target value, 
                # 0: Daily stock prediction,  
                # 1: Prediction for upward trend, 
                # 2: Prediction for turning to downwards trend (MACD35, MACD50), 
                # 3: Turning points of unwards trend,  
                # 4: Turning points of upwards/downwards trends
                # 5: PLR， 6： Dump
if target_mode == 4:
    classes = 3
else:
    classes = 1
    
flatten = 0
filter_mode = 1
timesteps = 10
network_mode = 'LSTM'    #  MLP,  LSTM, CNN,  CNN_LSTM, Conv2, Conv2_LSTM
input_dim = 8
train_rate = 0.9  # --- The ratio between train and test data
val_rate = 1.0    # --- The ratio between train and validate, if it's 1, no validation
val_split = 0.1
num_epoch = 50
batch_size = 20
candle_size = 80
bar_size = 50
preprocesser = 4  # --- 0: No preprocess,  1: Normalize with Normal distribution
                  # 2: Minmax normalization[0,1], 
                  # 3: Abnormal value normalization, 
                  # 4: Maximum absolute value, 
                  # 5: User-defined

## Set data and log files
ticker = '600036.SS'
#data_file = 'E:/dev/CNN_LSTM/data/'+ ticker+ '_all.csv'
#log_file = 'E:/dev/CNN_LSTM/log/log_5d.csv'

data_file = 'data/'+ ticker+ '_all.csv'
log_file = 'log/log_5d.csv'

log_fields = ['Time', 'ticker', 'Confusion_Matrix', 'Data File', 'network_mode', 'feature_mode1', 'feature_mode2', 'Features1', 'Features2',
              'Accuacy', 'Precision', 'Recall', 'f1-score', 'target_mode', 'classes', 'filter_mode', 'timesteps', 'train_rate', 'val_rate', 
              'val_split', 'input_dim', 'node', 'preprocesser', 'num_epoch', 'batch_size', 'flatten']

if os.path.isfile(log_file) == False:
    with open(log_file, 'a') as log:
        writer = csv.DictWriter(log, fieldnames=log_fields)
        writer.writeheader()
    log.close()
    
baseData = pd.read_csv(data_file, index_col='Date')

Candidate_list = list(baseData['Candidate'])


Features = [
    ['Candidate', 'pctChange', 'pctMAVol', 'VR', 'PL', 'CCI', 'CCIS', 'RSI', 'pctK', 'pctD', 'pctR', 'EMACDR', 'ROCMA5', 'ROC5', 'Aratio', 'Bratio', 'ABratio'],
    ['Candidate', 'ROC5', 'pctChange', 'CCI', 'pctD', 'pctK', 'pctR', 'PL', 'ROCMA5', 'pctMAVol', 'RSI'],
    ['Candidate', 'pctChange', 'pctMAVol', 'VR', 'PL'],
    ['Candidate', 'pctChange', 'CCI', 'CCIS', 'RSI'],
    ['Candidate', 'pctChange', 'pctK', 'pctD', 'pctR'],
    ['Candidate', 'pctChange', 'EMACDR', 'ROCMA5', 'ROC5'],
    ['Candidate', 'pctChange', 'Aratio', 'Bratio', 'ABratio'],
    ['Candidate', 'bottomTail', 'topTail', 'Body', 'wideHL'],
    ['Candidate', 'ROC5', 'VR', 'PL', 'CCI', 'CCIS', 'RSI', 'pctK', 'pctD', 'pctR']
]

feature_mode1 = 8
feature_mode2 = 6

X1 = baseData[Features[feature_mode1]]
X2 = baseData[Features[feature_mode2]]

print('Network: ' + network_mode)

db_index = list(baseData.index)  # Date list
print(len(db_index))

X1 = X1.as_matrix()  # Matrix without date property
#print(X1.shape)
x1_feature_nums = X1.shape[1]
X2 = X2.as_matrix()  # Matrix without date property
#print(X2.shape)
x2_feature_nums = X2.shape[1]
 
# Classify of input and output data
dblbl = baseData['Target']
y_data = dblbl      # Compare with classification result after using model
color_data = dblbl  # Use when analyzing data
feature_nums = baseData.shape[1]  # --- Number of properties of data
# num_classes = len(set(dblbl))    


# Data preprocess 88888888888888
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

if preprocesser == 1:       # Normalize with Normal distribution
    X1 = scale(X1)
    X2 = scale(X2)
elif preprocesser == 2:     # Minmax normalization[0,1]
    X1 = minmax_scale(X1)
    X2 = minmax_scale(X2)
elif preprocesser == 3:     # Abnormal value normalization
    X1 = robust_scale(X1)
    X2 = robust_scale(X2)
elif preprocesser == 4:     # Maximum absolute value
    X1 = maxabs_scale(X1)
    X2 = maxabs_scale(X2)
elif preprocesser == 5:
    pass


if filter_mode == 1:
    xm1_dataset, ym1_dataset, xm1_index, ym1_data = seq2filtered(X1, dblbl, db_index, y_data, timesteps)
    xm2_dataset, ym2_dataset, xm2_index, ym2_data = seq2filtered(X2, dblbl, db_index, y_data, timesteps)
    xr1_dataset, yr1_dataset, xr1_index, yr1_data = seq2dataset_filter(X1, dblbl, db_index, y_data, timesteps)
    xr2_dataset, yr2_dataset, xr2_index, yr2_data = seq2dataset_filter(X2, dblbl, db_index, y_data, timesteps)
    print(xm1_dataset.shape)
    print(xr1_dataset.shape)
else:
    xm1_dataset = X1
    ym1_dataset= dblbl
    xm1_index = db_index
    ym1_data = y_data
    xm2_dataset = X2
    ym2_dataset= dblbl
    xm2_index = db_index
    ym2_data = y_data
    
    xr1_dataset, yr1_dataset = seq2dataset(X1, dblbl, timesteps)
    xr1_index = db_index
    yr1_data = y_data
    xr2_dataset, yr2_dataset = seq2dataset(X2, dblbl, timesteps)
    xr2_index = db_index
    yr2_data = y_data
    
    print(xm1_dataset.shape)
    print(xr1_dataset.shape)

    
if classes == 1:
    class_mode_func = 'binary'
    activation_func = 'sigmoid'
    loss_func = 'binary_crossentropy'
    optimizer_func = 'sgd'
    ym_lbl = ym1_dataset
    yr_lbl = yr1_dataset
else:
    class_mode_func = 'categorical'
    activation_func = 'softmax'
    loss_func = 'categorical_crossentropy'
    optimizer_func = 'adam'
    ym_lbl = np_utils.to_categorical(ym1_dataset) # One-hot encoding for labels
    yr_lbl = np_utils.to_categorical(yr1_dataset) # One-hot encoding for labels
    
    
# Split data
m_train_nums = int(len(ym1_dataset) * train_rate)
m_val_nums = int(m_train_nums * val_rate)

xm1_train = xm1_dataset[:m_val_nums, :]
xm1_val = xm1_dataset[m_val_nums:m_train_nums, :]
xm1_test = xm1_dataset[m_train_nums:, :]

xm2_train = xm2_dataset[:m_val_nums, :]
xm2_val = xm2_dataset[m_val_nums:m_train_nums, :]
xm2_test = xm2_dataset[m_train_nums:, :]


r_train_nums = int(len(yr1_dataset) * train_rate)
r_val_nums = int(r_train_nums * val_rate)

xr1_train = xr1_dataset[:r_val_nums, :, :]
xr1_val = xr1_dataset[r_val_nums:r_train_nums, :, :]
xr1_test = xr1_dataset[r_train_nums:, :, :]

xr2_train = xr2_dataset[:r_val_nums, :, :]
xr2_val = xr2_dataset[r_val_nums:r_train_nums, :, :]
xr2_test = xr2_dataset[r_train_nums:, :, :]

ym_train = ym_lbl[:m_val_nums]
ym_val = ym_lbl[m_val_nums:m_train_nums]
ym_test = ym_lbl[m_train_nums:]


yr_train = yr_lbl[:r_val_nums]
yr_val = yr_lbl[r_val_nums:r_train_nums]
yr_test = yr_lbl[r_train_nums:]
                        
#print(ym_test)

if classes == 1:
    ym_True = ym_test
    yr_True = yr_test
else:
    ym_True = [np.argmax(y, axis=None, out=None) for y in ym_test]
    yr_True = [np.argmax(y, axis=None, out=None) for y in yr_test]
    #print(ym_True)


#for input_dim in [64]:   #8, 10, 20, 60, 100,  128
#    for node in [8, 16, 32, 64, 128]:       # 1, 5, 10, 20, 30, 50，  1, 2, 8, 16, 32, 64, 128
#        for kernels1 in [8, 16, 32, 64, 128]:      # 30, 50 , 100
#            for kernels2 in [8, 16, 32, 64, 128]:  # 8, 16, 32, 64, 128

input_dim = 64    # 8, 10, 20, 60, 100,  128
node = 32         # 1, 5, 10, 20, 30, 50，  1, 2, 8, 16, 32, 64, 128
                
print(ticker + ', input_dim: ' + str(input_dim) + ', node: ' + str(node)) 
input1 = keras.layers.Input(shape=(timesteps, x1_feature_nums))
x1 = keras.layers.LSTM(input_dim)(input1)
x1 = keras.layers.Dense(node, activation='relu')(x1)

x1_train, y_train = xr1_train, yr_train
x1_val, y_val = xr1_val, yr_val
x1_test, y_test = xr1_test, yr_test

x1_train = np.concatenate((x1_train, x1_test))
y_train = np.concatenate((y_train, y_test))

y_True = yr_True


# concat = keras.layers.concatenate([x1])
concat = x1

out = keras.layers.Dense(classes, activation = activation_func)(concat)
model = keras.models.Model(inputs=[input1], outputs=out)
model.compile(loss = loss_func, optimizer = optimizer_func, metrics = ['accuracy'])
hist = model.fit([x1_train], y_train, batch_size = batch_size, nb_epoch = num_epoch, verbose = 1, validation_split = val_split, shuffle=False)
#hist = model.fit([x1_train, x2_train], y_train, batch_size = batch_size, nb_epoch = num_epoch, verbose = 1, validation_data = ([x1_val, x2_val], y_val), shuffle=True)

# 6. Evaluate model

#print("=" * 50)
print()
print(" -- Evaluate model --")
print()
#scores = model.evaluate([X1, X2], Y, steps=5)
scores = model.evaluate([x1_test], y_test, batch_size=batch_size)
print("%s: %.2f%% %s: %.2f%%" %(model.metrics_names[0], scores[0]*100, model.metrics_names[1], scores[1]*100))
print('loss_and_metrics : ' + str(scores))

# 7. Predict by model 
yhat = model.predict([x1_test], batch_size=batch_size)

if classes == 1:
    yhat = yhat
else:
    yhat = [np.argmax(y, axis=None, out=None) for y in yhat]
#print("=" * 50)
print(yhat)
print()
print(" -- Use model --")
print()
# print("Real Predict Result")
result = []
for i in range(len(y_test)):
    if y_True[i] == yhat[i]:
        result.append([y_True[i], yhat[i], 1])
    else:
        result.append([y_True[i], yhat[i], 0])
        # print('Real : ' + str(result[i][0]) + ', Predict : ' + str(result[i][1]) + ', Result: ' + str(result[i][2]))

result = np.array(result)
print(result.T)
acc = accuracy_score(y_True, yhat) * 100
print("Accuracy: %0.2f%%" % acc)    
print("-- Confusion Matrix --")
cf_matrix = confusion_matrix(y_True, yhat)
print(cf_matrix)
print("-- Precision and Recall --")
cls_report = classification_report(y_True, yhat)
print(cls_report)    

#######################################################################################################
####################################          Show graph       #######################################
get_ipython().run_line_magic('matplotlib', 'inline')

# 5. Monitor training process

#print("=" * 50)
#print()
#print("-- Training_Monitoring --")

#                fig, loss_ax = plt.subplots()
#
#                acc_ax = loss_ax.twinx()
#
#                loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#                loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#                #loss_ax.set_ylim([0, 1])
#
#                acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#                acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
#                #acc_ax.set_ylim([0, 1])
#
#                loss_ax.set_xlabel('epoch')
#                loss_ax.set_ylabel('loss')
#                acc_ax.set_ylabel('accuray')
#
#                loss_ax.legend(loc='upper left')
#                acc_ax.legend(loc='lower left')

plt.show()


now = datetime.datetime.now()
cls_report = precision_recall_fscore_support(y_True, yhat)
value1, value2, value3 = (str(cls_report[0]), str(cls_report[1]), str(cls_report[2]))
with open(log_file, 'a') as log:
    writer = csv.DictWriter(log, fieldnames= log_fields)
    writer.writerow({
        'Time': now,
        'ticker': ticker, 
        'Confusion_Matrix': str(cf_matrix),
        'Data File': data_file,
        'network_mode': network_mode,
        'feature_mode1': feature_mode1,
        'feature_mode2': feature_mode2,
        'Features1': Features[feature_mode1],
        'Features2': Features[feature_mode2],
        'Accuacy': acc, 
        'Precision': value1,
        'Recall': value2,
        'f1-score': value3,
        'target_mode': target_mode,
        'classes': classes,
        'filter_mode': filter_mode,
        'timesteps': timesteps,
        'train_rate': train_rate,
        'val_rate': val_rate,
        'val_split': val_split,
        'input_dim': input_dim,
        'node': node,
        'preprocesser': preprocesser,
        'num_epoch': num_epoch,
        'batch_size': batch_size,
        'flatten': flatten
    })

log.close()
gc.collect()