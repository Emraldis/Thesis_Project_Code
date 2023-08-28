import sklearn
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.preprocessing import normalize

class Autoencoder(Model):
  def __init__(self, data_shape, debug = False, strides = 2, kernel_size = 3):
    super(Autoencoder, self).__init__()
    print(data_shape)
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(data_shape)),
      #layers.Conv2D(64, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same', strides=strides),
      #layers.Conv2D(32, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same', strides=strides),
      layers.Conv2D(16, kernel_size=(kernel_size, kernel_size), activation='tanh', padding='same', strides=strides),
      layers.Conv2D(8, kernel_size=(kernel_size, kernel_size), activation='tanh', padding='same', strides=strides),
      layers.Conv2D(4, kernel_size=(kernel_size, kernel_size), activation='tanh', padding='same', strides=strides)
      #layers.Dense(32, activation="relu"),
      #layers.Dense(16, activation="relu"),
      #layers.Dense(8, activation="relu")
    ])
    if debug:
        self.encoder.summary()
        
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(4, kernel_size=(kernel_size, kernel_size), strides=strides, activation='tanh', padding='same'),
      layers.Conv2DTranspose(8, kernel_size=(kernel_size, kernel_size), strides=strides, activation='tanh', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=(kernel_size, kernel_size), strides=strides, activation='tanh', padding='same'),
      #layers.Conv2DTranspose(32, kernel_size=(kernel_size, kernel_size), strides=strides, activation='relu', padding='same'),
      #layers.Conv2DTranspose(64, kernel_size=(kernel_size, kernel_size), strides=strides, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')
      #layers.Input(shape=(441,8)),
      #layers.Dense(16, activation="relu"),
      #layers.Dense(32, activation="relu"),
      #layers.Dense(84, activation="sigmoid")
    ])
    if debug:
        self.decoder.summary()

  def encode(self, x):
    return(self.encoder(x))
    
  def decode(self, x):
    return(self.decoder(x))

  def call(self, x):
    encoded = self.encode(x)
    #  encoded = np.squeeze(encoded[:,:,0])
    decoded = self.decode(encoded)
    return decoded

class test_model(Model):
  def __init__(self, data_shape, debug = False, strides = 1, kernel_size =1):
    super(test_model, self).__init__()
    #self.sequential_layers = tf.keras.Sequential([
    #layers.Dense()
    #])
    if debug:
      self.sequential_layers.summary()

class iterativeModel():
  def __init__(self, data_shape):
    self.autoencoder_array = []
    self.data_shape = data_shape
  
  def add_autoencoder_iteration(self):
    temp_autoencoder = Autoencoder(self.data_shape)
    #temp_autoencoder.compile(optimizer='adam', loss='mae')
    temp_autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    self.autoencoder_array.append(temp_autoencoder)
  
  def remove_autoencoder_iteration(self, index):
    self.autoencoder_array.pop(index)
  
  def train_autoencoder_iteration(self, index, train_data_start, train_data_result, test_data_start, test_data_result):
    self.autoencoder_array[index].fit(train_data_start, train_data_result,
                    epochs=10,
                    shuffle=True,
                    validation_data=(test_data_start, test_data_result))
    return()
  
  def call_autoencoder_iteration(self, index, data_in):
    data_out = self.autoencoder_array[index].call(data_in)
    return(data_out)
  
  def run_full(self, data_in_list, debug = False):
    data_in = np.zeros(self.data_shape)
    data_out_list = []
    if len(data_in_list) != len(self.autoencoder_array):
      print("timestep mismatch! \nlen(data_in_list) = " + str(len(data_in_list)) + "\nlen(self.autoencoder_array) = " + str(len(self.autoencoder_array)) + "\nAborting.")
      return()
      
    for i in range(len(self.autoencoder_array)):
      #print(type(list(np.shape(data_in_list[i]))))
      #print(type(self.data_shape.tolist()))
      if list(np.shape(data_in_list[i])) != self.data_shape.tolist():
        print("Missing data! Filling in.")
        if i != 0:
          data_in = data_out_list[i-1]
          if np.shape(data_in) != self.data_shape:
            print("Input shape mismatch, reshaping")
            data_in = np.reshape(data_in, self.data_shape)
      
      else:
          data_in = data_in_list[i]
      
      if debug:
          print(list(np.shape(data_in)))
          print(self.data_shape)
          print("---")
      
      #data_in[None, :]
      if debug:
        print(np.shape(data_in[None, :]))
      data_out = np.squeeze(self.call_autoencoder_iteration(i, data_in[None, :]))
      if debug:
        print(np.shape(data_in))
        print(np.shape(data_out))
        print("****")
      v_min = data_out.min()#[:,:,0:3].min()#axis=(0,1), keepdims=True)
      v_max = data_out.max()#[:,:,0:3].max()#axis=(0,1), keepdims=True)
      data_out = (data_out - v_min)/(v_max - v_min)
      
      #data_out = normalize(data_out[:,:,2])
      if False:
        print(np.shape(data_in))
        print(np.shape(data_out))
        print("vvv")
        print(data_in.min())
        print(data_in.max())
        print("---")
        print(data_out.min())
        print(data_out.max())
        print("^^^")
        print("####")
      data_out_list.append(data_out)#[:,:,0:3])
      if debug:
        print(np.shape(data_in))
        print(np.shape(data_out_list))
      
    return(data_out_list)