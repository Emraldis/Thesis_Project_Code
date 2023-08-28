import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap

class plotter:
  def __init__(self, dataset_name, path, file_header):
    self.dataset_name = dataset_name
    
    self.path = path
    self.header = file_header
    print("saving files to " + self.header)
    
    
  def project_feature_vector(self, feature_vector):
    #pca = PCA(n_components=2)
    projected_feature_vector = TSNE(n_components=1, learning_rate='auto', init='random', perplexity=40).fit_transform(TSNE(n_components=2, learning_rate='auto', init='random', perplexity=40).fit_transform(feature_vector))
    #reducer = umap.UMAP()
    #projected_feature_vector = reducer.fit_transform(feature_vector)
    #pca.fit(feature_vector)
    #projected_feature_vector = pca.singular_values_
    #print(projected_feature_vector)
    #quit()
    return(projected_feature_vector)
  
  def plot(self, feature_vector, regenerated_timestep, list_labels, list_saved, list_r_loss, list_fv_loss, recreation_loss_threshold, feature_vector_loss_threshold, testing_data, test_result, i, autoencoder_type, entry, save_plot = True, plot_feature_vector = True, regenerate_test = False, split_graphs = True):
    if i < 10:
      filler = "00"
    elif i < 100:
      filler = "0"
    else:
      filler = ""
#    print("Plotting graph")
    if not split_graphs:
      if plot_feature_vector:
        
        projected_feature_vector = self.project_feature_vector(feature_vector)
        
        if regenerate_test:
          fig, axs = plt.subplot_mosaic([['top_plot', 'top_plot', 'top_plot', 'top_plot', 'top_plot'],['original_image', 'autoencoder_result', 'recreated_image', 'feature_vector_plot', 'feature_vector_plot'],['original_image', 'autoencoder_result', 'recreated_image', 'feature_vector_plot', 'feature_vector_plot']], figsize=(10.0, 10.0), layout="constrained")
          
          axs['recreated_image'].imshow(np.squeeze(regenerated_timestep))
          axs['recreated_image'].set_title('Recreation Results')
          axs['recreated_image'].axis('off')
          
        else:
          fig, axs = plt.subplot_mosaic([['top_plot', 'top_plot', 'top_plot', 'top_plot'],['original_image', 'autoencoder_result', 'feature_vector_plot', 'feature_vector_plot'],['original_image', 'autoencoder_result', 'feature_vector_plot', 'feature_vector_plot']], figsize=(10.0, 10.0), layout="constrained")
        
        axs['feature_vector_plot'].scatter(projected_feature_vector[:,0], projected_feature_vector[:,1])
        axs['feature_vector_plot'].set_title('Plotted Feature Vector')
      
      else:
      
        if regenerate_test:
          fig, axs = plt.subplot_mosaic([['top_plot', 'top_plot', 'top_plot'],['original_image', 'autoencoder_result', 'recreated_image'],['original_image', 'autoencoder_result', 'recreated_image']], figsize=(10.0, 10.0), layout="constrained")
          
          axs['recreated_image'].imshow(np.squeeze(regenerated_timestep))
          axs['recreated_image'].set_title('Recreation Results')
          axs['recreated_image'].axis('off')
        
        else:
          fig, axs = plt.subplot_mosaic([['top_plot', 'top_plot'],['original_image', 'autoencoder_result'],['original_image', 'autoencoder_result']], figsize=(10.0, 10.0), layout="constrained")
      
      fig.suptitle("Experimental results for timestep " + str(i) + " using " + autoencoder_type + " autoencoder")
      axs['top_plot'].plot(range(len(list_labels)), list_labels)
      axs['top_plot'].plot(range(len(list_r_loss)), list_r_loss, label = "Reconstruction Loss", color = 'r')
      axs['top_plot'].plot(range(len(list_fv_loss)), list_fv_loss, label = "Feature Vector Loss", color = 'b')
      axs['top_plot'].vlines(list_saved, 0, 0.5, color = 'g')
      axs['top_plot'].axhline(y = recreation_loss_threshold, color = 'r', linestyle = 'dashed', label = "Reconstruction Loss Threshold")
      axs['top_plot'].axhline(y = feature_vector_loss_threshold, color = 'b', linestyle = 'dashed', label = "Feature Vector Loss Threshold")
      axs['top_plot'].set_title('Reconstruction and Feature Vector Loss Over Time')
      axs['top_plot'].xlabel("Time")
      axs['top_plot'].ylabel("Loss")
      axs['top_plot'].legend()
      axs['original_image'].imshow(np.squeeze(testing_data[entry][None,:]))
      axs['original_image'].set_title('Original Image')
      axs['original_image'].axis('off')
      axs['autoencoder_result'].imshow(test_result)
      axs['autoencoder_result'].set_title('Actual Output')
      axs['autoencoder_result'].axis('off')
    
      if save_plot:
        filename = self.header + filler + str(i) + "_results_plot"
        plt.savefig(self.path + filename)

    else:
      if plot_feature_vector:
        projected_feature_vector = self.project_feature_vector(feature_vector)
        plt.figure(layout="constrained")
        plt.scatter(projected_feature_vector[:,0], projected_feature_vector[:,1])
        plt.title('Plotted Feature Vector')
        if save_plot:
          filename = self.header + filler + str(i) + "_feature_vector"
          plt.savefig(self.path + filename)
        plt.close()
      
      if regenerate_test:
        plt.figure(layout="constrained")
        plt.imshow(np.squeeze(regenerated_timestep))
        plt.title('Recreation Results')
        plt.axis('off')
        if save_plot:
          filename = self.header + filler + str(i) + "_recreated_timestep"
          plt.savefig(self.path + filename)
        plt.close()
      
      plt.figure(layout="constrained")
      plt.imshow(np.squeeze(testing_data[entry][None,:]))
      plt.title('Original Image')
      plt.axis('off')
      if save_plot:
        filename = self.header + filler + str(i) + "_original_timestep"
        plt.savefig(self.path + filename)
      plt.close()
      
      plt.figure(layout="constrained")
      plt.imshow(np.squeeze(test_result))
      plt.title('Actual Output')
      plt.axis('off')
      if save_plot:
        filename = self.header + filler + str(i) + "_actual_output"
        plt.savefig(self.path + filename)
      plt.close()
      
      plt.figure(layout="constrained")
      plt.plot(range(len(list_labels)), list_labels)
      plt.plot(range(len(list_r_loss)), list_r_loss, label = "Reconstruction Loss", color = 'r')
      plt.plot(range(len(list_fv_loss)), list_fv_loss, label = "Feature Vector Loss", color = 'b')
      plt.vlines(list_saved, 0, 0.5, color = 'g')
      plt.axhline(y = recreation_loss_threshold, color = 'r', linestyle = 'dashed', label = "Reconstruction Loss Threshold")
      plt.axhline(y = feature_vector_loss_threshold, color = 'b', linestyle = 'dashed', label = "Feature Vector Loss Threshold")
      plt.title('Reconstruction and Feature Vector Loss Over Time')
      plt.xlabel("Time")
      plt.ylabel("Loss")
      plt.legend()
      plt.title('Actual Output')
      #plt.axis('off')
      if save_plot:
        filename = self.header + filler + str(i) + "_timestep_plot"
        plt.savefig(self.path + filename)
      plt.close()