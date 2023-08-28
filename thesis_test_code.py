import utils
import models
import time

from utils import dataset
from utils import timestep_entry
from utils import convert_data_to_json
from utils import convert_json_to_data
from utils import glitch
from utils import value_cap
from thesis_graphs import plotter

from models import Autoencoder
from test_sim import sim

import sklearn
import numpy as np
import tensorflow as tf
import os
import json
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow import keras

class data_entry:
  def __init__(self, data, target, label):
    self.data = data
    self.target = target
    self.label = label

class experiment_stats:
  def __init__(self, dataset_name, r_loss_list, clean_r_loss_list, glitched_r_loss_list, average_clean_r_loss, average_r_loss, glitched_average_r_loss,  fv_loss_list, clean_fv_loss_list, glitched_fv_loss_list, average_clean_fv_loss, average_fv_loss, glitched_average_fv_loss):
    self.dataset_name = dataset_name
    self.r_loss_list = r_loss_list
    self.clean_r_loss_list = clean_r_loss_list
    self.glitched_r_loss_list = glitched_r_loss_list
    self.average_clean_r_loss = average_clean_r_loss
    self.average_r_loss = average_r_loss
    self.glitched_average_r_loss = glitched_average_r_loss
    self.fv_loss_list = fv_loss_list
    self.clean_fv_loss_list = clean_fv_loss_list
    self.glitched_fv_loss_list = glitched_fv_loss_list
    self.average_clean_fv_loss = average_clean_fv_loss
    self.average_fv_loss = average_fv_loss
    self.glitched_average_fv_loss = glitched_average_fv_loss
  
  def return_stat_averages(self, labels = True, return_as_string = True):
    if return_as_string:
      output_string = ""
      if labels:
        output_string = "dataset name,average clean recreation loss,average glitched recreation loss, average overall recreation loss,average clean feature vector loss,average glitched feature vector loss, average overall feature vector loss\n"
      output_string += str(self.dataset_name) + "," + str(self.average_clean_r_loss) + "," + str(self.glitched_average_r_loss) + "," + str(self.average_r_loss) + "," + str(self.average_clean_fv_loss) + "," + str(self.glitched_average_fv_loss) + "," + str(self.average_fv_loss) + "\n"
      print(output_string)
      return(output_string)
    else:
      output = {}
      output["dataset name"] = self.dataset_name
      output["average clean recreation loss"] = self.average_clean_r_loss
      output["average glitched recreation loss"] = self.glitched_average_r_loss
      output["average overall recreation loss"] = self.average_r_loss
      output["average clean feature vector loss"] = self.average_clean_fv_loss
      output["average glitched feature vector loss"] = self.glitched_average_fv_loss
      output["average overall feature vector loss"] = self.average_fv_loss
      return(output)


class copier:
  def __init__(self, xdim, ydim):
    self.xdim = xdim
    self.ydim = ydim
    self.local_arr = np.zeros((xdim, ydim))
  
  def copy(self, input_arr, start_x = 0, start_y = 0):
    self.local_arr = input_arr[start_x:(start_x+self.xdim), start_y:(start_y+self.ydim)]
  
  def paste(self, input_arr, start_x = 0, start_y = 0):
    output_arr = input_arr
    output_arr[start_x:(start_x+self.xdim), start_y:(start_y+self.ydim)] = self.local_arr
    return(output_arr)
    
debug = False
save_mode = "full"
#save_mode = "partial"
#save_mode = "none"
range_testing = False
plot_feature_vector = False
include_glitched_data = True
save_plot = False
autoencoder_type = "both"
#autoencoder_type = "regular"
#autoencoder_type = "predictive"
#loss_function_type = "mse"
loss_function_type = "mae"
loss_function_range = ["mae"]#, "mse"]

training_epochs = 30
#learn_rate = 0.01
learn_rate = 0.001

random_sample_results = 60
#random_sample_results = 1
#random_sample_results = 0

num_glitched_entries = 20
#num_glitched_entries = -1

#data_type = "test"
#data_type = "test_sim"
data_type = "dataset"

anomaly_type = "blank"
#anomaly_type = "glitch"
#anomaly_type = "chunk"
#anomaly_type = "turbulent"

if not include_glitched_data:
  anomaly_type = "none"

regenerate_test = True
#regenerate_test = False

if anomaly_type == "blank":
  recreation_loss_threshold = 0.1
  feature_vector_loss_threshold = 0.2
elif anomaly_type == "glitch":
  recreation_loss_threshold = 0.03
  feature_vector_loss_threshold = 0.03
elif anomaly_type == "chunk":
  recreation_loss_threshold = 0.046
  feature_vector_loss_threshold = 0.047
elif anomaly_type == "turbulent":
  recreation_loss_threshold = 0.03
  feature_vector_loss_threshold = 0.05
elif anomaly_type == "none":
  recreation_loss_threshold = 0.03
  feature_vector_loss_threshold = 0.05
  
starting_timestep = 0
#starting_timestep = 10
periodic_save_interval = 60
#periodic_save_interval = 1

data_frame_01 = np.asarray([
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]])

data_frame_02 = np.asarray([
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,0,0,0,0,0,0,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]])

data_frame_03 = np.asarray([
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]])

data_frame_04 = np.asarray([
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0]])


data_frames = [data_frame_01, data_frame_02, data_frame_03, data_frame_04]

folder_path = "/home1/s4216768/Master's Thesis/Data/data_compiled/"
file_path = folder_path + "_compiled_data_reference_list.txt"

datasets = {}
glitch_datasets = {}

pad = True

if data_type == "dataset":
  max_timesteps = 54
  if pad == True:
  #  xdim=444
    xdim=448
  #  xdim=512
  #  ydim=84
    ydim=88
  #  ydim = 96  
  #  ydim=128
    zdim=1

  else:
    xdim=441
    ydim=84
    zdim=1

elif data_type == "test_sim":
  max_timesteps = 100
  xdim=128
  ydim=64
  zdim = 1
  filter_size = 11
  num_datasets = 300
  folder_path = "/home1/s4216768/Master's Thesis/Data/data_generated/"

else:
  max_timesteps = 100
  xdim = np.shape(data_frame_01)[0]
  ydim = np.shape(data_frame_01)[1]
  zdim = 1  
  num_datasets = 100

data_shape = np.asarray([xdim, ydim, zdim])
channels = 1


if data_type == "dataset":
  reference_file = open(file_path)
  reference_list = json.loads(reference_file.read())

  print("Loading data into local storage")

  for entry in reference_list:
    temp_list = []
    data_entries = convert_json_to_data(folder_path, (entry + "_compiled.txt"), return_dict = True)
    label = 0
    for timestep in data_entries:
      if int(timestep) > 10:
        #print(timestep)
        #print(type(data_entries[timestep].data))
        [xpad, ypad, zpad] = (data_shape - np.shape(data_entries[timestep].data))
        if data_entries[timestep].label == 1 and anomaly_type == "turbulent":
          label = 1
          #print("Turbulent!")
        if int(timestep) < (max_timesteps - 1):
          if pad == True:
            temp_entry = data_entry(np.pad(data_entries[timestep].data, ((0,xpad), (0,ypad), (0,zpad)), 'constant', constant_values=(0)),
                                    np.pad(data_entries[timestep+1].data, ((0,xpad), (0,ypad), (0,zpad)), 'constant', constant_values=(0)), label)
          else:
            temp_entry =data_entry(data_entries[timestep].data, data_entries[timestep+1].data, label)
          
          #with np.printoptions(threshold=np.inf):
            #print(temp_entry.data)
          temp_list.append(temp_entry)
    #quit()
    if anomaly_type == "turbulent" and label == 1:
      glitch_datasets["G_" + entry] = dataset(entry, temp_list, label = label)
      #print("Turbulent!")
    else:
      datasets[entry] = dataset(entry, temp_list, label = label)
    print("Local loading of " + entry)

  reference_file.close()
  
  if include_glitched_data and not (anomaly_type == "turbulent"):
    print("Generating bugged datasets")
    
    if anomaly_type == "chunk":
      chunk_copier = copier(200, 60)
    
    if num_glitched_entries > 0:
      sample_selection = random.choices(list(datasets), k = num_glitched_entries)
    else:
      sample_selection = list(datasets)
    #print(sample_selection)
    for selection in sample_selection:
      print(selection)
      if anomaly_type == "chunk":
        dataset_to_copy = random.choice(list(datasets))
      max_len = len(datasets[selection].data_entries)
      #print(max_len)
      rand_start_timestep = random.randint(20, max_len - 15)
      #quit()
      rand_end_timestep = random.randint(rand_start_timestep+5, rand_start_timestep+10)
      temp_list = []
      for timestep in range(len(datasets[selection].data_entries)):
        if timestep >= rand_start_timestep and timestep <= rand_end_timestep:
          if anomaly_type == "blank":
            temp_entry = data_entry(np.ones_like(datasets[selection].data_entries[timestep].data), datasets[selection].data_entries[timestep].target, 3)
          if anomaly_type == "glitch":
            temp_entry = data_entry(glitch(xdim, ydim, datasets[selection].data_entries[timestep].data, 400, num_glitches = 6, glitch_value = 1), datasets[selection].data_entries[timestep].target, 3)
          if anomaly_type == "chunk":
            chunk_copier.copy(datasets[dataset_to_copy].data_entries[timestep].data, start_x = 6, start_y = 20)
            temp_entry = data_entry(chunk_copier.paste(datasets[selection].data_entries[timestep].data, start_x = 6, start_y = 20), datasets[selection].data_entries[timestep].target, 3)
            plt.imsave("/home1/s4216768/Master's Thesis/Code/test_sim_results/sim_testing/test_" + str(timestep) + ".png", np.squeeze(temp_entry.data))
        else:
          temp_entry = data_entry(datasets[selection].data_entries[timestep].data, datasets[selection].data_entries[timestep].target, datasets[selection].data_entries[timestep].label)
        temp_list.append(temp_entry)
      entry = "G_"+selection
      #print(entry)
      glitch_datasets[entry] = dataset(entry, temp_list, label = 2)
    #quit()
  
  print("Loaded data into local storage")

elif data_type == "test_sim":
  datasets = {}
  print("Generating sim dataset")
  glitch_chance_params = list(range(-150, 100, 25))
  for element in range(len(glitch_chance_params)):
    if glitch_chance_params[element] < 0:
      glitch_chance_params[element] = 0
  glitch_size_params = range(1, 101, 10)
  num_glitches_params = range(1,4)
  k = 0
  l = 0
  for i in range(num_datasets):
    filler = ""
    if i < 10:
      filler = filler + "0"
    if i < 100:
      filler = filler + "0"
    if i >= 10 and i%10 == 0:
      k += 1
    if k >= 10 and k%10 == 0:
      l += 1
    simulator = sim(max_timesteps, xdim, ydim, filter_size = filter_size)
    print("glitch_chance")
    print(str(i%10) + "/" + str(len(glitch_chance_params)))
    print("glitch_size")
    print(str(k%10) + "/" + str(len(glitch_size_params)))
    print("num_glitches")
    print(str(l%3) + "/" + str(len(num_glitches_params)))
    simulator.run_sim(glitch_chance = glitch_chance_params[i%10], glitch_size = glitch_size_params[k%10], num_glitches = num_glitches_params[l%3])
    simulator.save_sim(folder_path + filler + str(i) + "/")
    temp_list = []
    if glitch_chance_params[i%10] > 0:
      label = 1
    else:
      label = 0
    print("Generated dataset " + filler + str(i))
    for j in range(simulator.duration):
      if j < simulator.duration:
        temp_entry = data_entry(simulator.timesteps[j], simulator.timesteps[j+1], label)
      temp_list.append(temp_entry)
    datasets[str(i)] = dataset(str(i), temp_list)
  print("Dataset generated")

else:
  print("Generating test dataset")
  datasets = {}
  for i in range(num_datasets-25):
    #print(i)
    temp_list = []
    state = random.randint(0,len(data_frames)-1)
    for j in range(max_timesteps):
      index = [*range(0,len(data_frames))]
      #print(index)
      dice_roll = random.randint(0, 10)
      if dice_roll <= 9:
        temp_entry = data_entry(data_frames[state], data_frames[state], state)
        state = state
      else:
        index.pop(state)
        dice_roll = random.randint(0,len(index)-1)
        temp_entry = data_entry(data_frames[state], data_frames[dice_roll], dice_roll)
        state = index[dice_roll]
      temp_list.append(temp_entry)
    datasets[str(i)] = dataset(str(i), temp_list, 0)
  for i in range(num_datasets-24, num_datasets):
    for j in range(max_timesteps):
      index = [*range(0,len(data_frames)-1)]
      #print(len(data_frames)-1)
      dice_roll = random.randint(0, 20)
      if dice_roll <= 9:
        temp_entry = data_entry(data_frames[state], data_frames[state], state)
        state = state
      elif dice_roll == 10:
        if state != len(data_frames)-1:
          index.pop(state)
        dice_roll = random.randint(0,len(index)-2)
        temp_entry = data_entry(data_frames[state], data_frames[dice_roll], dice_roll)
        state = index[dice_roll]
      else:
        temp_entry = data_entry(data_frames[state], data_frames[len(data_frames)-1], 4)
        state = len(data_frames)-1
      temp_list.append(temp_entry)
    datasets[str(i)] = dataset(str(i), temp_list, 1)
  print("Dataset generated")
    

print("Loading Data")

raw_data = []
raw_labels = []

train_all = False

for entry in datasets:
  print(entry)
  for timestep in datasets[entry].data_entries:
    if ((datasets[entry].label == 0) and (anomaly_type == "turbulent")) or anomaly_type != "turbulent":
      #print(datasets[entry].label)
      data = timestep
      raw_data.append(data)
      raw_labels.append(timestep.label)

print("Data Loaded")
#raw_data = np.array(raw_data)
raw_labels = np.array(raw_labels)
#quit()
print("Splitting Timesteps into Test and Train sets")

split_result = []
splitter = StratifiedShuffleSplit(n_splits=1, test_size=int(len(raw_data)*0.9))

print(int(len(raw_data)*0.9))

for train_index, test_index in splitter.split(raw_data, raw_labels):
    split_result.append([train_index, test_index])

train_data = []
train_targets = []

test_data = []
test_targets = []

all_data = []

for entry in split_result:
    test_index, train_index = entry

    for index in train_index:
      train_data.append(raw_data[index].data.data)
      train_targets.append(raw_data[index].target.data)
      all_data.append(raw_data[index].data.data)

    for index in test_index:
      test_data.append(raw_data[index].data.data)
      test_targets.append(raw_data[index].target.data)
      all_data.append(raw_data[index].data.data)

    break

train_data = np.asarray(train_data).astype('float')
train_targets = np.asarray(train_targets).astype('float')

test_data = np.asarray(test_data).astype('float')
test_targets = np.asarray(test_targets).astype('float')

all_data = np.asarray(all_data).astype('float')

print(np.shape(raw_data)) #.3-11130 .4-9540 .5-
print("split full data")

print("Training Autoencoder")

if data_type == "dataset":
  root = "/home1/s4216768/Master's Thesis/Results/"
else:
  root = "/home1/s4216768/Master's Thesis/Testing_Results/"

if range_testing:
  root = "/home1/s4216768/Master's Thesis/Range_Results/"


if autoencoder_type == "predictive" or autoencoder_type == "both":
  predictive_autoencoder = Autoencoder(data_shape)
if autoencoder_type == "regular" or autoencoder_type == "both":
  regular_autoencoder = Autoencoder(data_shape)
if autoencoder_type == "predictive" or autoencoder_type == "both":
  predictive_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), loss='mae')
if autoencoder_type == "regular" or autoencoder_type == "both":
  regular_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate), loss='mae')
if autoencoder_type == "predictive" or autoencoder_type == "both":

#learning_rate = tf.Variable(0.001, trainable=False)
#tf.keras.backend.set_value(learning_rate, learn_rate)

  start_time_train_pred = time.time()
  predictive_history = predictive_autoencoder.fit(train_data, train_targets, 
                                                  epochs=training_epochs,
                                                  batch_size=512,
                                                  validation_data=(test_data, test_targets),
                                                  shuffle=True)
  if save_plot:
    plt.plot(predictive_history.history["loss"], label="Training Loss")
    plt.plot(predictive_history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(root + "predictive_training_loss")
  end_time_train_pred = time.time()
  elapsed_time_train_pred = end_time_train_pred - start_time_train_pred
  print("Trained Predictive autoencoder, total time: " + str(elapsed_time_train_pred))
if autoencoder_type == "regular" or autoencoder_type == "both":
  start_time_train_reg = time.time()
  regular_history = regular_autoencoder.fit(train_data, train_data, 
                                            epochs=training_epochs,
                                            batch_size=512,
                                            validation_data=(test_data, test_data),
                                            shuffle=True)
  if save_plot:
    plt.plot(regular_history.history["loss"], label="Training Loss")
    plt.plot(regular_history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(root + "regular_training_loss")
  end_time_train_reg = time.time()
  elapsed_time_train_reg = end_time_train_reg - start_time_train_reg
  print("Trained Predictive autoencoder, total time: " + str(elapsed_time_train_reg))


autoencoders = {}
if autoencoder_type == "predictive" or autoencoder_type == "both":
  autoencoders["predictive"] = predictive_autoencoder
if autoencoder_type == "regular" or autoencoder_type == "both":
  autoencoders["regular"] = regular_autoencoder
print("Autoencoder Trained")

print("Data prepared, beginning test")

if loss_function_type == "mae":
  loss_function = tf.keras.losses.MeanAbsoluteError()
elif loss_function_type == "mse":
  loss_function = tf.keras.losses.MeanSquaredError()


dataset_range = {}

if random_sample_results > 0:
  sample_selection = random.choices(list(datasets), k = random_sample_results)
  for selection in sample_selection:
    dataset_range[selection] = datasets[selection]
  if include_glitched_data:
    glitched_selection = random.choices(list(glitch_datasets), k = num_glitched_entries)
    for selection in glitched_selection:
      dataset_range[selection] = glitch_datasets[selection]

else:
  for key, value in dict.items(datasets):
    dataset_range[key] = value
  if include_glitched_data:
    for key, value in dict.items(glitch_datasets):
      dataset_range[key] = value

if save_mode == "full":
  periodic_save_interval = 1

if range_testing:
  #recreation_loss_range = np.arange(0.0,1.0,0.05)
  recreation_loss_range = [0.0, 0.25, 1.0]
  #feature_vector_loss_range = np.arange(0.0,0.2, 0.05)
  feature_vector_loss_range = [0.0, 0.15, 1.0]
else:
  recreation_loss_range = [0.04]
  feature_vector_loss_range = [0.02]
  #feature_vector_loss_range = [1.0]

spreadsheet_data_path = root + "data_summary.txt"

autoencoder_experiment_summary = {}

stats_timestep_time_list = []

parameter = ""

for autoencoder_type in autoencoders:
  autoencoder_experiment_summary[autoencoder_type] = []
  autoencoder = autoencoders[autoencoder_type]
  print("Testing " + autoencoder_type + " autoencoder")
  root_path = root + autoencoder_type
  if not os.path.exists(root_path):
    os.mkdir(root_path)
  for rlf in loss_function_range:
    p1 = ""
    if rlf == "mae":
      recreation_loss_function = tf.keras.losses.MeanAbsoluteError()
      p1 = "r_mae"
    elif rlf == "mse":
      recreation_loss_function = tf.keras.losses.MeanSquaredError()
      p1 = "r_mse"
    for fvlf in loss_function_range:
      p2 = ""
      if fvlf == "mae":
        feature_vector_loss_function = tf.keras.losses.MeanAbsoluteError()
        p2 = "_fv_mae"
      elif fvlf == "mse":
        feature_vector_loss_function = tf.keras.losses.MeanSquaredError()
        p2 = "_fv_mse"
      parameter = p1+p2
      if parameter != "":
        root_path = root + autoencoder_type + "/" + parameter
        if not os.path.exists(root_path):
          os.mkdir(root_path)
      directory = "rt" + "{:1.2f}".format(recreation_loss_threshold) + "_fvt" + "{:1.2f}".format(feature_vector_loss_threshold) + "_" + anomaly_type
      root_path = root + autoencoder_type + "/" + parameter + "/" + directory
      if not os.path.exists(root_path):
        os.mkdir(root_path)
      spreadsheet_data_path = root_path + "/data_summary" + directory + ".txt"
      averages_data_path = root_path + "/experiment_results_averages.txt"
      accuracy_report_path = root_path + "/" + autoencoder_type + "_" + anomaly_type + "_accuracy_report.txt"
      if save_mode != "none":
        file_output = open(spreadsheet_data_path, "a")
        file_output.write("Autoencoder type: " + autoencoder_type + ",Anomaly type: " + anomaly_type + ",\n")
        file_output.write("Timestep,Label,Recreation Loss,Recreation Loss Threshold,Feature Vector Loss,Feature Vector Loss Threshold,saved?\n")
        file_output.close()
        spreadsheet_output = ""
      experiment_folder_path = root_path
      if not os.path.exists(root_path):
        os.mkdir(root_path)
      j = 0
      
      regeneration_losses = {}
      
      r_average_true_positive_list = []
      r_average_true_negative_list = []
      
      fv_average_true_positive_list = []
      fv_average_true_negative_list = []
      
      run_time_list = []
      
      for dataset in dataset_range:
      
        if save_mode != "none":
          file_output = open(spreadsheet_data_path, "a")
          file_output.write("\n")
          file_output.close()
        #print(dataset)
        directory = "/" + dataset + "/"
        path = root_path + "/" + directory
        #print(path)
        if not os.path.exists(path):
          os.mkdir(path)
        if not os.path.exists(path + "feature_vectors/"):
          os.mkdir(path + "feature_vectors/")
        if not os.path.exists(path + "graphs/"):
          os.mkdir(path + "graphs/")
        
        testing_data = []
        testing_targets = []
        testing_labels = []
        for entry in range(len(dataset_range[dataset].data_entries)):
          #print(entry)
          #print(type(datasets[dataset].data_entries[entry]))
          #print(type(datasets[dataset].data_entries[entry].data))
          data = dataset_range[dataset].data_entries[entry].data.data
          targets = dataset_range[dataset].data_entries[entry].target
          label = dataset_range[dataset].data_entries[entry].label
          testing_data.append(np.squeeze(data))
          testing_targets.append(np.squeeze(targets))
          if (label < 3 and anomaly_type != "turbulent") or label == 0:
            testing_labels.append(0)
          else:
            testing_labels.append(1)
        
        temp = testing_data.pop(len(testing_data)-1)
        test_result = []
        feature_vector = []
        console_output = []
        list_fv_loss = np.zeros(len(dataset_range[dataset].data_entries)-1)
        list_r_loss = np.zeros(len(dataset_range[dataset].data_entries)-1)
        list_glitched_r_loss = []
        list_glitched_fv_loss = []
        list_clean_r_loss = []
        list_clean_fv_loss = []
        list_saved = []
        list_labels = np.zeros(len(dataset_range[dataset].data_entries)-1)
        list_regen_loss = []
        
        stats_r_true_positives = 0
        stats_r_false_positives = 0
        stats_r_true_negatives = 0
        stats_r_false_negatives = 0
        
        r_true_positive_rate = 0
        r_true_negative_rate = 0
        
        stats_fv_true_positives = 0
        stats_fv_false_positives = 0
        stats_fv_true_negatives = 0
        stats_fv_false_negatives = 0
        
        fv_true_positive_rate = 0
        fv_true_negative_rate = 0
        
        stats_timesteps_saved = 0
        stats_anomalous_timesteps_detected = 0
        stats_high_change_timesteps_detected = 0
        stats_total_timesteps = 0
        regenerated_timestep = testing_data[0][None,:]
        
        stats_time_elapsed = 0
        stats_start_time = time.time()
        #print(np.shape(testing_data))
        i = 0
        temp_feature_vector = None
        
        
        graph_plotter = plotter(dataset, path + "graphs/", autoencoder_type + "_" + dataset + "_")
        
        for entry in range(len(testing_data)):
          saved = 0
          stats_total_timesteps += 1
          #print(np.shape(entry))
          timestep_time_start = time.time()
          feature_vector = autoencoder.encode(testing_data[entry][None,:])
          #print(np.shape(feature_vector))
          #quit()
          test_result = (np.squeeze(autoencoder.decode(feature_vector)))
          timestep_time_end = time.time()
          stats_timestep_time_list.append(timestep_time_end - timestep_time_start)
          if regenerate_test:
            regenerated_timestep = autoencoder.call(regenerated_timestep)
          feature_vector = np.squeeze(feature_vector[:,:,0])
          loss = recreation_loss_function(testing_targets[entry][None,:], test_result).numpy()
          list_r_loss[i] = (loss)
          if testing_labels[i] == 0:# and i >= 10:
            list_clean_r_loss.append(loss)
          elif testing_labels[i] > 0:
            list_glitched_r_loss.append(loss)
          
          #print(i)
          
          if i < 10:
            filler = "00"
          elif i < 100:
            filler = "0"
          else:
            filler = ""
          
          s_timestep = "Timestep " + str(i) + " Label: " + str(testing_labels[i]+1)
          
          if testing_labels[i]+1 == 4:
            s_glitch = "Glitched "
          else:
            s_glitch = ""
            
          feature_vector_loss = 0.0
          if i == 0:
            list_fv_loss[i] = feature_vector_loss
            list_clean_fv_loss.append(loss)
          
          if i != 0:
            feature_vector_loss = feature_vector_loss_function(feature_vector, temp_feature_vector).numpy()
            list_fv_loss[i] = feature_vector_loss
            
            if testing_labels[i] == 0:# and i >= 10:
              list_clean_fv_loss.append(loss)
            elif testing_labels[i] > 0:
              list_glitched_fv_loss.append(loss)
            
            if (loss > recreation_loss_threshold):
              s_state = " - High-loss (" + str(loss) + ")"
              stats_anomalous_timesteps_detected += 1
              if testing_labels[i] > 0:# and i >= 10:
                stats_r_true_positives += 1
              elif testing_labels[i] == 0:#  and i >= 10:
                stats_r_false_positives += 1
              
              if feature_vector_loss > feature_vector_loss_threshold:
                s_cause = ", feature vector difference: " + str(feature_vector_loss)
                s_effect = " - Saving to " + path + "/" +  filler + str(i) + "_result_high-Loss.png"
                stats_timesteps_saved += 1
                stats_high_change_timesteps_detected += 1
                #if i > 10:
                saved = i
                  #print(i)
                  
                if testing_labels[i] > 0:# and i >= 10:
                  stats_fv_true_positives += 1
                elif testing_labels[i] == 0:#  and i >= 10:
                  stats_fv_false_positives += 1
                
                if save_mode != "none":
                  img = plt.imsave(path + "/" +  filler + str(i) + "_result_high-Loss.png", test_result)
                  img = plt.imsave(path + "/" +  filler + str(i) + "_target.png", np.squeeze(testing_targets[entry][None,:]))
                  img = plt.imsave(path + "/" +  filler + str(i) + "_input.png", np.squeeze(testing_data[entry][None,:]))
                
              elif i % periodic_save_interval == 0 or (starting_timestep != 0 and i < starting_timestep):
                s_cause = ", periodic save"
                s_effect = " - Saving to " + path + "/" +  filler + str(i) + "_result_high-Loss.png"
                stats_timesteps_saved += 1
                #saved = 1
                
                
                if testing_labels[i] == 0:# and i >= 10:
                  stats_fv_true_negatives += 1
                elif testing_labels[i] > 0:# and i >= 10:
                  stats_fv_false_negatives += 1
                
                if regenerate_test:
                  regenerated_timestep = autoencoder.call(testing_data[entry][None,:])
                if save_mode != "none":
                  img = plt.imsave(path + "/" +  filler + str(i) + "_result_high-Loss.png", test_result)
                  img = plt.imsave(path + "/" +  filler + str(i) + "_target.png", np.squeeze(testing_targets[entry][None,:]))
                  img = plt.imsave(path + "/" +  filler + str(i) + "_input.png", np.squeeze(testing_data[entry][None,:]))
                
              else:
                s_cause = ", feature vector difference: " + str(feature_vector_loss)
                s_effect = " - Not saved"
                
                if testing_labels[i] == 0:# and i >= 10:
                  stats_fv_true_negatives += 1
                elif testing_labels[i] > 0:# and i >= 10:
                  stats_fv_false_negatives += 1
                
            elif loss <= recreation_loss_threshold:
              s_state = " - Low-loss (" + str(loss) + ")"
              if testing_labels[i] == 0:# and i >= 10:
                stats_r_true_negatives += 1
              else:
                stats_r_false_negatives += 1

              if feature_vector_loss > feature_vector_loss_threshold:
              
                if testing_labels[i] > 0:# and i >= 10:
                  stats_fv_true_positives += 1
                elif testing_labels[i] == 0:#  and i >= 10:
                  stats_fv_false_positives += 1
              '''
                s_cause = ", feature vector difference: " + str(feature_vector_loss)
                s_effect = " - Saving to " + path + filler + str(i) + "_result_low-Loss.png"
                stats_timesteps_saved += 1
                stats_high_change_timesteps_detected += 1
                saved = 1
                if save_mode != "none":
                  img = plt.imsave(path + filler + str(i) + "_result_low-Loss.png", test_result)
                  img = plt.imsave(path + filler + str(i) + "_original.png", np.squeeze(testing_targets[entry][None,:]))
              '''
              if i % periodic_save_interval == 0 or (starting_timestep != 0 and i < starting_timestep):
              #elif i % periodic_save_interval == 0:
                s_cause = ", periodic save"
                s_effect = " - Saving to " + path + "/" +  filler + str(i) + "_result_low-Loss.png"
                stats_timesteps_saved += 1
                #saved = 1
                if regenerate_test:
                  regenerated_timestep = autoencoder.call(testing_data[entry][None,:])
                if save_mode != "none":
                  img = plt.imsave(path + "/" + filler + str(i) + "_result_low-Loss.png", test_result)
                  img = plt.imsave(path + "/" +  filler + str(i) + "_original.png", np.squeeze(testing_targets[entry][None,:]))
                  
                if testing_labels[i] == 0:# and i >= 10:
                  stats_fv_true_negatives += 1
                elif testing_labels[i] > 0:# and i >= 10:
                  stats_fv_false_negatives += 1
                
              else:
                s_cause = ""
                s_effect = " - Not saved"
                
                if testing_labels[i] == 0:# and i >= 10:
                  stats_fv_true_negatives += 1
                elif testing_labels[i] > 0:# and i >= 10:
                  stats_fv_false_negatives += 1
              
          else:
            s_state = " - Low-loss (" + str(loss) + ")"
            s_cause = ", periodic save"
            s_effect = " - Saving to " + path + filler + str(i) + "_result_low-Loss.png"
            #stats_timesteps_saved += 1
            saved = 0
            if save_mode != "none":
              img = plt.imsave(path + filler + str(i) + "_result_low-Loss.png", test_result)
          
          if save_mode != "none":
            spreadsheet_output = str(i) + "," + str(testing_labels[i]) + "," + str(loss) + "," + "{:1.2f}".format(recreation_loss_threshold) + "," + str(feature_vector_loss) + "," + format(feature_vector_loss_threshold) + "," + str(saved) + "\n"
            file_output = open(spreadsheet_data_path, "a")
            file_output.write(spreadsheet_output)
            file_output.close()
          
          
          if regenerate_test:
            regen_loss = recreation_loss_function(testing_targets[entry][None,:], np.squeeze(regenerated_timestep)).numpy()
            list_regen_loss.append(regen_loss)
          
          if saved != 0:
            list_saved.append(saved)
          #print(list_saved)
          if anomaly_type != "turbulent":
            list_labels[i] = testing_labels[i]
          output_string = s_glitch + s_timestep + s_state + s_cause + s_effect
          console_output.append(output_string)
          print(output_string)
          temp_feature_vector = feature_vector

          if save_plot:
            graph_plotter.plot(feature_vector, regenerated_timestep, list_labels, list_saved, list_r_loss, list_fv_loss, recreation_loss_threshold, feature_vector_loss_threshold, testing_data, test_result, i, autoencoder_type, entry, save_plot = save_plot, plot_feature_vector = plot_feature_vector, regenerate_test = regenerate_test)
          
          i += 1
        
        #quit()
        
        testing_data.append(temp)
        
        stats_end_time = time.time()
        stats_time_elapsed = stats_end_time - stats_start_time
        run_time_list.append(stats_time_elapsed)
        
        print("ran dataset " + dataset + ". Elapsed time was: " + str(stats_time_elapsed))
        
        sum = 0
        for r_loss in list_r_loss:
          sum += r_loss
        average_r_loss = sum/len(list_r_loss)
        average_clean_r_loss = -1
        if len(list_clean_r_loss) > 0:
          sum = 0
          for r_loss in list_clean_r_loss:
            sum += r_loss
          average_clean_r_loss = sum/len(list_clean_r_loss)
        glitched_average_r_loss = -1
        if len(list_glitched_r_loss) > 0:
          sum = 0
          for r_loss in list_glitched_r_loss:
            sum += r_loss
          glitched_average_r_loss = sum/len(list_glitched_r_loss)
          sum = 0
        for fv_loss in list_fv_loss:
          sum += r_loss
        average_fv_loss = sum/len(list_fv_loss)
        average_clean_fv_loss = -1
        if len(list_clean_fv_loss) > 0:
          sum = 0
          for fv_loss in list_clean_fv_loss:
            sum += r_loss
          average_clean_fv_loss = sum/len(list_clean_fv_loss)
        sum = 0
        if len(list_glitched_fv_loss) > 0:
          for fv_loss in list_glitched_fv_loss:
            sum += r_loss
          glitched_average_fv_loss = sum/len(list_glitched_fv_loss)
        else:
          glitched_average_fv_loss = -1
        
        
        
        print("Calculating true positive and negative rates for dataset")
                        
        if len(list_glitched_r_loss) > 0 and len(list_clean_r_loss) > 0:
                              
          r_true_positive_rate = value_cap(stats_r_true_positives, 0, len(list_glitched_r_loss), error_catching = True) / len(list_glitched_r_loss)
          r_true_negative_rate = value_cap(stats_r_true_negatives, 0, len(list_clean_r_loss), error_catching = True) / len(list_clean_r_loss)
        
          fv_true_positive_rate = value_cap(stats_fv_true_positives, 0, len(list_glitched_fv_loss), error_catching = True) / len(list_glitched_fv_loss)
          fv_true_negative_rate = value_cap(stats_fv_true_negatives, 0, len(list_clean_fv_loss), error_catching = True) / len(list_clean_fv_loss)
        
          r_average_true_positive_list.append(r_true_positive_rate)
          r_average_true_negative_list.append(r_true_negative_rate)
      
          fv_average_true_positive_list.append(fv_true_positive_rate)
          fv_average_true_negative_list.append(fv_true_negative_rate)
        
        experiment_results = experiment_stats(dataset, list_r_loss, list_clean_r_loss, list_glitched_r_loss, average_clean_r_loss, average_r_loss, glitched_average_r_loss,  list_fv_loss, list_clean_fv_loss, list_glitched_fv_loss, average_clean_fv_loss, average_fv_loss, glitched_average_fv_loss)
        autoencoder_experiment_summary[autoencoder_type].append(experiment_results)#(dataset, list_r_loss, average_r_loss, list_fv_loss, average_fv_loss))
        output_string = "--------------------Dataset " + dataset + " Test Results--------------------\n" + \
                        "Average Recreation Loss: " + str(average_r_loss) + "\n" + \
                        "average Feature Vector Loss: " + str(average_fv_loss) + "\n" + \
                        "Recreation Loss Threshold: " + str(recreation_loss_threshold) + "\n" + \
                        "Feature Vector Loss Threshold: " + str(feature_vector_loss_threshold) + "\n" + \
                        "Total Timesteps Saved - " + str(stats_timesteps_saved) + "/" + str(stats_total_timesteps) + "\n" + \
                        "Anomalous Timesteps Detected - " + str(stats_anomalous_timesteps_detected) + "/" + str(stats_total_timesteps) + "\n" + \
                        "High-Change Timesteps Detected - " + str(stats_high_change_timesteps_detected) + "/" + str(stats_total_timesteps) + "\n" + \
                        "-----------------------------------------------------------------------------\n"
        
        print(output_string)
          
        if save_mode != "none":
          output_file_path = experiment_folder_path + "/experiment_summary.txt"
          file_output = open(output_file_path, "a")
          file_output.write(output_string)
          file_output.write("\n\n")
          file_output.close()
          
          output_file_path = path + "console_output.txt"
          file_output = open(output_file_path, "w")
          file_output.write(output_string)
          
          for string in console_output:
            file_output.write("\n")
            file_output.write(string)
            
          file_output.close()
          
          if j > 0:
            show_labels = False
          else:
            show_labels = True
          j+=1
          averages_output = experiment_results.return_stat_averages(labels = show_labels)
          file_output = open(averages_data_path, "a")
          file_output.write(averages_output)
          file_output.close()
        #print(np.shape(test_result))
        #print(np.shape(testing_data))
        #print("****")
        
        #break
      
        regeneration_losses[dataset] = list_regen_loss
      
      average_run_time = 0
      
      for entry in run_time_list:
        average_run_time += entry
        
      average_run_time = average_run_time/len(run_time_list)
      
      
      print("Calculating Average True Positives and Negatives")
      
      r_average_true_positive_rate = 0
      if len(r_average_true_positive_list) > 0:
        for entry in r_average_true_positive_list:
          r_average_true_positive_rate += entry
        r_average_true_positive_rate = r_average_true_positive_rate/len(r_average_true_positive_list)
      
      r_average_true_negative_rate = 0
      if len(r_average_true_negative_list) > 0:
        for entry in r_average_true_negative_list:
          r_average_true_negative_rate += entry
        r_average_true_negative_rate = r_average_true_negative_rate/len(r_average_true_negative_list)
      
      fv_average_true_positive_rate = 0
      if len(fv_average_true_positive_list) > 0:
        for entry in fv_average_true_positive_list:
          fv_average_true_positive_rate += entry
        fv_average_true_positive_rate = fv_average_true_positive_rate/len(fv_average_true_positive_list)
      
      fv_average_true_negative_rate = 0
      if len(fv_average_true_negative_list) > 0:
        for entry in fv_average_true_negative_list:
          fv_average_true_negative_rate += entry
        fv_average_true_negative_rate = fv_average_true_negative_rate/len(fv_average_true_negative_list)
      
      file_output = open(accuracy_report_path, "w")
      accuracy_report = "Recreation Average True Positive Rate,Recreation Average True Negative Rate,Feature Vector Average True Positive Rate,Feature Vector Average True Negative Rate\n" + str(r_average_true_positive_rate) + "," + str(r_average_true_negative_rate) + "," + str(fv_average_true_positive_rate) + "," + str(fv_average_true_negative_rate)
      file_output.write(accuracy_report)
      file_output.close()
      
      if regenerate_test:
        regeneration_loss_averages = []
        total_timesteps = len(regeneration_losses[dataset])
        for timestep in range(total_timesteps):
          timestep_regeneration_loss = []
          i = 0
          for dataset in regeneration_losses:
            timestep_regeneration_loss.append(regeneration_losses[dataset][timestep])
          
          sum = 0
          for entry in timestep_regeneration_loss:
            sum += entry
          timestep_regeneration_average = sum/len(timestep_regeneration_loss)
          regeneration_loss_averages.append(timestep_regeneration_average)
        
        #plt.figure(layout="constrained")
        plt.plot(range(total_timesteps), regeneration_loss_averages, label = autoencoder_type)
        plt.title('Image Reconstruction Loss Over Time')
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(root_path + "/" + "Recreation_loss_plot")
      print("Test results saved")

average_timestep_time = 0
for entry in stats_timestep_time_list:
  average_timestep_time += entry
average_timestep_time = average_timestep_time/len(stats_timestep_time_list)

print("Average dataset runtime: " + str(average_run_time))
print("Average timestep runtime: " + str(average_timestep_time))

'''

for autoencoder_type in autoencoder_experiment_summary:
  for dataset_stats in autoencoder_experiment_summary[autoencoder_type]:
    print(autoencoder_type + "     " + dataset_stats.dataset_name)
    print(str(len(dataset_stats.r_loss_list)) + "    " + str(len(dataset_stats.fv_loss_list)))
    print("Average Recreation Loss: " + str(dataset_stats.average_r_loss) + "    Average Feature Vector Loss: " + str(dataset_stats.average_fv_loss))

'''

'''

  print("Saving data for dataset " + directory)

  for i in range(len(test_result)):
    if debug:
      print("Calculating Loss for timestep " + str(i))
    if debug:
      print(loss)
    if i < 10:
      filler = "0"
    else:
      filler = ""
    if (loss <=0.25):
      if debug:
        print("timestep " + filler + str(i) + " has low loss")
      if (i % 5) == 0:
        #print(np.shape(test_result[i]))
        #print(np.shape(testing_data[i]))
        #print("--")
        img = plt.imsave(path + filler + str(i) + "_result_low-Loss.png", test_result[i])
    else:
      if debug:
        print("timestep " + filler + str(i) + " has high loss")
      #print(np.shape(test_result[i]))
      #print(np.shape(testing_data[i]))
      img = plt.imsave(path + filler + str(i) + "_result_high-Loss.png", test_result[i])
      img = plt.imsave(path + filler + str(i) + "_original.png", np.squeeze(testing_data[i]))
    if debug:
      print("---")

print("Test results saved")
'''
