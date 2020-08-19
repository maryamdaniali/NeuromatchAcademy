
## Objective: This code uses neural activity of mice brain to detect the expected behavior and the actual behavior of each trail
##      1- We want to see if we can detect mouse response based on its neural activity for each trial. [response classification]
##      2- We want to see if we can detect the expected action for each trial, based on mouse's neural activity.[actual behavior classification]
## Data: Steinmetz (https://www.nature.com/articles/s41586-019-1787-x), Spiking neural activity of different brain areas of mice 
## Input: spiking rates for the visual and motor cortex of mice 
## Output: classification accuracy on two different problems: 1-expected action 2-actual behavior detection
## Method: 1-D convolutional neural network
## Author: Maryam Daniali, July, 2020.

import os, requests
import random
import time
import pickle
import numpy as np
import tensorflow as tf
import sys
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D, Dropout,Activation
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.model_selection import KFold




VALIDATION_SPLIT = 0.25 #amount of data for cross validation

'''Creating directories for results text files and plots'''
dir = os.path.dirname(__file__)
current_time = int(time.time())
RESULT_PATH = os.path.join(dir, 'Result/'+f"{current_time}/")


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    # Ref: https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory?noredirect=1&lq=1
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def load_data():
  '''Loads the Steinmetz data from three different files uploaded on osf.io '''
  fname = []
  for j in range(3):
    fname.append('steinmetz_part%d.npz'%j)
  url = ["https://osf.io/agvxh/download"]
  url.append("https://osf.io/uv3mw/download")
  url.append("https://osf.io/ehmw2/download")

  for j in range(len(url)):
    if not os.path.isfile(fname[j]):
      try:
        r = requests.get(url[j])
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          with open(fname[j], "wb") as fid:
            fid.write(r.content)

  alldata = np.array([])
  for j in range(len(fname)):
    alldata = np.hstack((alldata, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

  return alldata        

def retrieve_data_spikes(data_dict , session):
  ''' retrieve spiking data for desired brain regions
      input: data dictionary for all features, the corresponding session number (0:38)
      output: 3D matrix for spiking rate (#neurons, #trial, #timebins) for the desired regions
  '''    

  dat = data_dict[session]
  unique_brain_areas = np.unique(dat['brain_area'])

  dt = dat['bin_size'] # binning at 10 ms
  vis_right = dat['contrast_right'] # 0 - low - high
  vis_left = dat['contrast_left'] # 0 - low - high

  # groupings of brain regions
  regions = [  "vis ctx", "primary vis ctx" ,  "mot ctx", "primary mot ctx" ]
  brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                  ["VISp"], #primary visual cortex
                  ["MOp", "MOs"], #motor cortex
                  ["MOp"] # primary motor cortex
                  ]

  nareas = 4 # all regions 
  NN = len(dat['brain_area']) # number of neurons

  primary_vis_data = dat['spks'][np.isin(dat['brain_area'], brain_groups[1])]
  vis_data = dat['spks'][np.isin(dat['brain_area'], brain_groups[0])]
  primary_mo_data = dat['spks'][np.isin(dat['brain_area'], brain_groups[3])]
  mo_data = dat['spks'][np.isin(dat['brain_area'], brain_groups[2])]

  
  return primary_vis_data, vis_data, primary_mo_data, mo_data


def oneDconvModel(n_timesteps, n_features, n_outputs):
  ''' 1D-CNN model for classification
      input: length of sequences (n_timesteps), number of features in the data(1 here), number of classes(n_outputs)
      output: created model object
  '''    
  model = Sequential()
  model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(n_outputs, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model

def train_eval(model,X_train, y_train,X_test, y_test):
  ''' takes in the model, train and test set, and evaluates the modol accuracy
      input: model, train set, train labels, test set, test labels
      output: accuracy on the test set, the used model
  '''  
  verbose, epochs, batch_size = 0, 100, 5
  model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose) 
  # evaluate model
  _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
  return accuracy, model

def train(data, target):
  ''' Takes in the data, shuffles, one-hot code the labels, applies cross validation and calls other training methods
      input: whole data matrix, whole labels
      output: avraged accuracy and the standard deviation of that over the K cross validation cases
  '''  
  data = np.expand_dims(data,axis=2)
  label = to_categorical(target)
  data, label = shuffle(data, label)
  
  n_examples, n_timesteps, n_features = data.shape[0], data.shape[1], 1
  

  n_folds = 4
  accuracy = []
  kf = KFold( n_splits=n_folds, shuffle=False)
  for train, test in kf.split(data):
    model = None
    model = oneDconvModel(n_timesteps, n_features,len(np.unique(target)))
    acc, trained_model = train_eval(model, data[train], label[train], data[test], label[test])
    accuracy.append(acc)

  #print_intermadiate_output(trained_model,data[test], label[test]) ##optional
  vis_raw(data[test], 'test_seq' , data[test].min().min(),  data[test].max().max())
  return np.array(accuracy).mean(), np.array(accuracy).std() 


def print_intermadiate_output(trained_model,data, label):
  ''' Takes in the trained model, and represents intermadiate output (feature map) for each record in the data matrix
      input: model, data, labels
      output: no direct outputs - calls vis_features() that prints out the intemadiate feature maps
  '''  
  new_model = Sequential()
  # creating a temporary model with same weights which just has the first hiddel layer
  for layer in trained_model.layers[0:1]:
    new_model.add(layer)
  intermediate_output = new_model.predict(data) 
  for i in range(len(data)):
    vis_features(intermediate_output[i,:,:],f'_test_{i}_label_{label[i,:]}', (intermediate_output.min(), intermediate_output.max()))

def vis_features(data, title , _range = (0,1)):
    ''' Visualizes sequence over time in a three column plot, suitable for intermadiate features'''  
    # 3 columns
    min_y, max_y = _range
    curr_path = RESULT_PATH
    mkdir_p(curr_path)
    n_records, length = data.shape
    nplots = 10
    ncol = 3
    i = 0
    while( i < n_records): 
        fig, axs = plt.subplots(nrows=nplots, ncols=ncol, sharex=True)
        if i+(nplots*ncol)<= n_records:
            curr_chunk = data[i:i+(nplots*ncol)] 
        else:
            curr_chunk = data[i:n_records]       
        for j in range(nplots): 
            for k in range(ncol): 
                if ((j*ncol)+k) < curr_chunk.shape[1]:
                    curr_rec = np.transpose(curr_chunk[:,(j*ncol) + k])
                    axs[j,k].set_ylim([min_y,max_y])
                    axs[j,k].plot( curr_rec, label = 'f_' + str( i + (j*ncol) + k), color ="#a28e78")
                    axs[j,k].legend(prop={'size': 3})
                    plt.setp(axs[j,k].get_yticklabels(), visible=False)
    
                        
        plt.title(title)
        plt.savefig(curr_path + title + str(i)+'.pdf' , alpha = 0.5 , dpi=500)
        plt.clf()
        plt.close(fig)
        i += (nplots * ncol)

def vis_raw(data, title , range_min, range_max):
    ''' Visualizes sequence over time in one column plot with fix y axis range, suitable for raw sequences'''
    # raw input a single participant in 6 row
    curr_path = RESULT_PATH 
    mkdir_p(curr_path)
    nplots = 5

    num_sample = data.shape[0]
    i = 0
    while i< (num_sample-nplots):
      fig, axs = plt.subplots(nrows=nplots, ncols=1, sharex=True)
      for j in range(nplots): 
        axs[j].set_ylim([range_min,range_max])
        axs[j].plot( data[i,:], label = 'test'+str(i+j), color ="#a28e78")
        axs[j].legend(prop={'size': 3})
        plt.setp(axs[j].get_yticklabels())
              
      plt.savefig(curr_path + title+ str(i + nplots)+'.pdf', alpha = 0.5 , dpi=2000)
      plt.clf()
      plt.close(fig)
      i = i + nplots  



if __name__ == "__main__":

  '''in the main function, we take the data, and for each session we use our 1-D CNN network to classify two problems
      1- We want to see if we can detect mouse response based on its neural activity for each trial. [response classification]
      2- We want to see if we can detect the expected action for each trial, based on mouse's neural activity.[actual behavior classification]
      We print out the crossvalidated accuracy for each of the mentioned problems for three different cases:
      1-by looking at the whole experiment sequence (250 ms)
      2-by looking at the first 90 ms of the experiment (90 ms) which is mostly related to the visual cortex activities
      3-by looking at the sequence after the go-cue, when actually mice could move the wheel toward the target direction, which is moslty related to the motor activities
  '''
  
  alldata = load_data()
  dict_data_areas = {}
  for session in range(0,len(alldata)):

    print("_____________session:", session)
    print('mouse: ', alldata[session]['mouse_name'] ) #the particular mouse in this session (there are multiple sessions for some mice)
    print("brain areas: ", np.unique(alldata[session]['brain_area'])) #all brain area involved in that session
    res = alldata[session]['response'] # right - nogo - left (-1, 0, 1)
    dict_data_areas["pv"], dict_data_areas["v"], dict_data_areas["pm"], dict_data_areas["m"] = retrieve_data_spikes(alldata,session) #primary visual cortex(pv), vitual cortex (v), primary motor cortex(pm), motor cortex (m)
    

    vis_right = alldata[session]['contrast_right'] # value of the right monitor's contrast, 0 - low - high
    vis_left = alldata[session]['contrast_left'] # value of the left monitor's contrast,0 - low - high

    for key in dict_data_areas:

      key_ro = dict_data_areas[key][:,np.logical_and(vis_left==0, vis_right>0)] #when the right stimuli was stronger, expected behavior is turning to the left
      key_lo= dict_data_areas[key][:,np.logical_and(vis_left>0 , vis_right==0)] #when the left stimuli was stronger, expected behavior is turning to the right
      key_n = dict_data_areas[key][:,np.logical_and(vis_left==0 , vis_right==0)] #when both monitors are off, expected behavior is no-go
      key_b = dict_data_areas[key][:,np.logical_and(vis_left>0, vis_right>0)] #ignored - there is not correct behavior for this case based on the paper
      res_ro = res[np.logical_and(vis_left==0, vis_right>0)] # responses wihen the right stimuli was stronger
      res_lo = res[np.logical_and(vis_left>0 , vis_right==0)] # response when the left stimuli was stronger
      res_n = res[np.logical_and(vis_left==0 , vis_right==0)] # response when neither stimuli was on
      res_b = res[np.logical_and(vis_left>0, vis_right>0)] # response to both stimuli, ignored.


      if dict_data_areas[key].shape[0] > 0 :
        print("___current area:", key)

        ''' normalize averaged spike activities over the number of neurans based on the avrage activies before the stimulus onset'''
        key_ro_m = key_ro.mean(axis=0) - np.vstack(key_ro[:,:,:50].mean(axis=2).mean(axis=0))
        key_lo_m = key_lo.mean(axis=0) - np.vstack(key_lo[:,:,:50].mean(axis=2).mean(axis=0))
        key_n_m = key_n.mean(axis=0) - np.vstack(key_n[:,:,:50].mean(axis=2).mean(axis=0))

        ''' concatenate the nomalized spiking rate for all three types of stimuli'''
        key_all = np.concatenate((key_ro_m,key_lo_m, key_n_m))
        key_first90 = key_all[:,:90] #if we take spike rates just 90 first ms of the 250 ms trails, mostly related to visual activities
        key_actiontime = key_all[:,70:] #if we take spike rates just after the gocue, mostly related to motor activities

        ''' concatenate the actual mice responce for all three types of stimuli'''
        key_res_org = np.concatenate((res_ro, res_lo, res_n))
        key_res = np.where(key_res_org==-1, 2, key_res_org) 
        key_exp = np.concatenate((np.full_like(res_ro,2),np.full_like(res_lo, +1),np.full_like(res_n, 0)))

        '''train the model and find the cross validated accuracy'''
        accuracy_res_mean, accuracy_res_std = train(key_all, key_res)


        '''classification related to mice actual behavior'''
        # print('accuracy mouse response',accuracy_res)
        print(accuracy_res_mean , '+/-' , accuracy_res_std )
        accuracy_res_90_mean, accuracy_res_90_std = train(key_first90, key_res)
        # print('accuracy mouse response-90ms',accuracy_res_90)
        print(accuracy_res_90_mean , '+/-' , accuracy_res_90_std )
        accuracy_res_actiontime_mean, accuracy_res_actiontime_std = train(key_actiontime, key_res)
        # print('accuracy mouse action time',accuracy_res_actiontime)
        print(accuracy_res_actiontime_mean , '+/-' , accuracy_res_actiontime_std )

        '''classification related to mice expected behavior/action'''
        accuracy_exp_mean, accuracy_exp_std = train(key_all, key_exp)
        # print('accuracy expected behavior for the mouse', accuracy_exp)
        print(accuracy_exp_mean , '+/-' , accuracy_exp_std )
          
        accuracy_exp_90_mean, accuracy_exp_90_std = train(key_first90, key_exp)
        # print('accuracy expected behavior for the mouse-90ms', accuracy_exp_90)
        print(accuracy_exp_90_mean , '+/-' , accuracy_exp_90_std ) 
        accuracy_exp_actiontime_mean, accuracy_exp_actiontime_std = train(key_actiontime, key_exp)
        # print('accuracy mouse action time',accuracy_exp_actiontime) 
        print(accuracy_exp_actiontime_mean , '+/-' , accuracy_exp_actiontime_std )

      
          