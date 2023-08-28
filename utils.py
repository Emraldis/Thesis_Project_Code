import numpy as np
import json
import random
import matplotlib.pyplot as plt

class dataset:
    def __init__(self, name, data_entries, label = 0):
        self.name = name
        self.data_entries = data_entries
        self.label = label

class timestep_entry:
    def __init__(self, timestep, timestep_data, timestep_label):
        self.timestep = timestep
        self.data = timestep_data.astype("float")
        self.label = timestep_label


        #sum = 0

        #for element in np.nditer(timestep_data):
        #    sum += element

        #sum /= timestep_data.size

        #if sum != 0:
        #    self.normal = timestep_data/sum
        #else:
        #    self.normal = timestep_data
        #self.normal = self.normal.astype(int)

    def convert_to_dict(self):
        return({"timestep":self.timestep, "data":self.data.tolist(), "label":self.label})

    def update_stats(self, timestep_average):
        self.stats = timestep_stats(self, timestep_average)

class timestep_stats:
    def __init__(self, timestep_data, timestep_average):
        self.diff = np.absolute(timestep_data.normal - timestep_average.normal)
        self.max_diff = np.amax(self.diff)


def convert_data_to_json(file_name, folder_path, root_path, xdim=441, ydim=84, zdim=1):

    data = {}
    label = 0

    for i in range(0,54):
        file_path = root_path + folder_path + file_name

        if i < 10:
            file_path = file_path + "0"

        full_path = file_path + str(i) + ".raw"
        print(full_path)
        file_input = open(full_path, "r")
        input_data = np.fromfile(file_input, dtype=np.uint8)
        input_data = np.reshape(input_data, (xdim, ydim, zdim))

        data[str(i)] = (timestep_entry(i, input_data, label).convert_to_dict())

        file_input.close()

    file_output = open(root_path + folder_path + "/" + folder_path + ".txt", "w")
    file_output.write(json.dumps(data))
    file_output.close()

def simple_convert_data_to_json(data, file_path, xdim=441, ydim=84, zdim=1):

    file_output = open(file_path, "w")

    file_output.write(json.dumps(data))

    file_output.close()

def convert_json_to_data(path, file_name, return_dict = False):

    if return_dict:
        data = {}

        input_file = open(path+file_name, "r")
        file_data = json.loads(input_file.read())

        for entry in file_data:
            data[file_data[entry]["timestep"]] = timestep_entry(file_data[entry]["timestep"], np.asarray(file_data[entry]["data"], dtype=np.float), file_data[entry]["label"])
    
    else:
        data = []

        input_file = open(path+file_name, "r")
        file_data = json.loads(input_file.read())

        for entry in file_data:
            data.append(timestep_entry(file_data[entry]["timestep"], np.asarray(file_data[entry]["data"], dtype=np.float), file_data[entry]["label"]))

    input_file.close()
    return(data)

def value_cap(value, lower_bound, upper_bound, error_catching = False):
  if value < lower_bound:
    if error_catching:
      print("ERROR! VALUE BELOW LOWER BOUND!")
      print(value)
      quit()
    return(lower_bound)
  elif value > upper_bound:
    if error_catching:
      print("ERROR! VALUE ABOVE UPPER BOUND!")
      print(value)
      quit()
    return(upper_bound)
  else:
    return(value)

def glitch(x_dim, y_dim, input_array, glitch_size, num_glitches = 1, glitch_value = -1, debug = True):
  #print("Glitching Timestep")
  for i in range(num_glitches):
    #print("i")
    current_coords = np.array([random.randint(0, x_dim-1),random.randint(0, y_dim-1)])
    #print(current_coords)
    for j in range(glitch_size):
      input_array[current_coords[0], current_coords[1]] = glitch_value
      coordinate_delta = np.array([random.randint(-1,1), random.randint(-1,1)])
      current_coords = current_coords + coordinate_delta
      current_coords[0] = value_cap(current_coords[0], 0, x_dim-1)
      current_coords[1] = value_cap(current_coords[1], 0, y_dim-1)
      if debug:
        #print(np.shape(input_array))
        filler = ""
        if j < 10:
          filler = filler + "0"
        if j < 100:
          filler = filler + "0"
        plt.imsave("/home1/s4216768/Master's Thesis/Code/test_sim_results/sim_testing/" + str(i) + "_" + filler + str(j) + ".png", np.squeeze(input_array))
  return(input_array)