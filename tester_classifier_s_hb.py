# For system.
import sys
sys.path.append('./')
import numpy as np
import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import importlib
from urllib.request import urlopen
import argparse
import json
from pathlib import Path

# Others in the mean time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

# For torch.
import torch
import torch.nn as nn
from torch.nn import functional as F

import models.model_transformer as wits
importlib.reload(wits)
from models import logger
importlib.reload(logger)
from models.ChannelTransformationModule import ChannelTransformationModule as channel_transforms

# @title [RUN] Validate splits and locations with Plotting

import matplotlib.pyplot as plt
from matplotlib import container
import pandas as pd
import seaborn as sns
sns.reset_orig()


def plot_positions(positions, positions_new_final):
    """
    Plots old and new positions for sanity check.

    Parameters
    ----------
    positions : numpy.ndarray

    Returns
    -------
    None

    """
    plt.plot(positions[:,0], positions[:,1], '.b', markersize = 2.0, label='Actual')
    plt.plot(positions_new_final[:,0], positions_new_final[:,1], '.r', markersize = 2.0, label='Actual')

    plt.xlabel('x [mm]', fontsize=13)
    plt.ylabel('y [mm]', fontsize=13)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, loc='best', prop={'size': 12}, numpoints=1, fancybox=False)

    plt.show()


def prep_data_load(args):
  """
  Selected datasets for creating train, test and val sets.
  
  Parameters:
      args:
          - dataset_to_download (str): Dataset to 'download' (DIS_lab_LoS, ULA_lab_LoS, URA_lab_LoS, URA_lab_nLoS).
          - saved_dataset_path (str): Path to where datasets are saved.
          - sub_dataset_file_csi (str): CSI file.
          - sub_dataset_file_loc (str): Locations file.
  
  Returns:
          - train_dataset (data (not) loader torch): Training.
          - val_dataset (data (not) loader torch): Validation.
          - test_dataset (data (not) loader torch): Testing.
  """
  # Define dataset related paths and file names
  dataset_to_download = args.dataset_to_download
  if dataset_to_download == "DIS_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/DIS_lab_LoS'
      channel_file_name = 'ultra_dense/DIS_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/DIS_lab_LoS"
  elif dataset_to_download == "ULA_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/ULA_lab_LoS'
      channel_file_name = 'ultra_dense/ULA_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/ULA_lab_LoS"
  elif dataset_to_download == "URA_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/URA_lab_LoS'
      channel_file_name = 'ultra_dense/URA_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/URA_lab_LoS"
  elif dataset_to_download == "URA_lab_nLoS":
      download_dataset_sub_path = 'ultra_dense/URA_lab_nLoS'
      channel_file_name = 'ultra_dense/URA_lab_nLoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/URA_lab_nLoS"
  elif dataset_to_download == "S-200":
      print('Note that for this case we use a smaller sample size.')
  elif dataset_to_download == "HB-200":
      print('Note that for this case we use a smaller sample size.')        
  else:
      raise ValueError("This dataset is not used. Check the configuration of dataset name!")

  print(f'Dataset main path is {os.path.dirname(os.path.realpath(args.saved_dataset_path))}')
  print(f'\n\n******** Dataset Selected is {dataset_to_download}************\n\n')

  '''
  Here, you load the data (or a sample from the dataset). Otherwise, below (commented)
  I have put the pre-processing of the KUL datasets.
  '''
  with open(Path(args.saved_dataset_path)/args.sub_dataset_file_csi, 'rb') as f1:
    csi2 = np.load(f1)
    f1.close()
  with open(Path(args.saved_dataset_path)/args.sub_dataset_file_loc, 'rb') as f2:
      location_data_and_classes = np.load(f2)    
      f2.close()

  '''
  import sys
  import re
  import zipfile
  #loadingData = loadCSI.LoadCsiMat(saved_dataset_path+"/"+saved_dataset_file_name,dataset_size)   

  KU_dataset_npz = np.load(saved_dataset_path+'/ultra_dense.zip')


  # define the type of dataset from:
  # 1. DIS_lab_LoS; 2. ULA_lab_LoS; 3. URA_lab_LoS; 4. URA_lab_nLoSeeeecmd

  # locations data
  positions = KU_dataset_npz["ultra_dense/URA_lab_nLoS/user_positions.npy"]
  print("shape of user locations: ", positions.shape)

  # define location ids to consider (every third is enough.)
  locIDs = np.arange(0,len(positions),3) # Note that 3 in here is to sample every third location.

  # Define parameters useful for reading the csi measuremenets and location tags.
  start_measurement = 0
  stop_measurement = len(locIDs)#positions.shape[0] # all locations/measurements.
  antennas = 64
  subcarriers = 100

  # select only relevant positions ids
  positions = positions[start_measurement:stop_measurement,:]

  # read channel measurements
  zip = zipfile.ZipFile(saved_dataset_path+'/ultra_dense.zip')

  # regex all filenames from the dataset type, i.e., from any of the KUL_lab_* datasets.
  reg = re.compile('ultra_dense/URA_lab_nLoS/samples/channel_measurement_.*')

  #csi_filenames = list(filter(reg.match, zip.namelist()))
  #csi_filenames.sort() # sort them to start from 0, so we can match with positions.
  #print(csi_filenames) #comment this later.

  # Now read the CSI based on selected locations 

  csi = np.empty((stop_measurement-start_measurement, antennas, subcarriers), dtype=complex)

  #for i, ID in enumerate(locIDs):
      # Store sample
      #sample = KU_dataset_npz['ultra_dense/URA_lab_nLoS/samples/channel_measurement_' + str(ID).zfill(6) + '.npy']
      # print(sample.real.shape)
      # csi[i, :, :] = KU_dataset_npz['ultra_dense/URA_lab_nLoS/samples/channel_measurement_' + str(ID).zfill(6) + '.npy']

  #np.save(dataset_path+"kuluwen_URA_lab_nLoS_75k.npy", csi)

  KU_dataset_npz = np.load(saved_dataset_path+'/ultra_dense.zip')


  # define the type of dataset from:
  # 1. DIS_lab_LoS; 2. ULA_lab_LoS; 3. URA_lab_LoS; 4. URA_lab_nLoS

  # locations data
  positions_pre = KU_dataset_npz["ultra_dense/URA_lab_nLoS/user_positions.npy"]
  print("shape of user locations: ", positions_pre.shape)

  # define location ids to consider (every third for example.)
  locIDs = np.arange(0,len(positions_pre),3)

  # Define parameters useful for reading the csi measuremenets and location tags.
  start_measurement = 0
  stop_measurement = len(locIDs)#positions.shape[0] # all locations/measurements.
  antennas = 64
  subcarriers = 100

  # select only relevant positions ids
  positions_pre = positions_pre[locIDs,0:2]


  # Now load csi from the saved data (every third, based on locIDs)
  csi = np.load(saved_dataset_path+'/kuluwen_URA_lab_nLoS_75k.npy')
  csi.shape, positions.shape

  # Concat position IDs to postitions
  position_IDs = np.arange(0,len(positions_pre),1)
  positions = np.concatenate((positions_pre,np.expand_dims(position_IDs, axis=1)), axis=1)
  print(csi.shape, positions.shape)

  a = np.arange(0,len(positions),1)
  positions_new = positions[a]
  csi_new = csi[a]

  ### Now, add a label to indicate the spot for the purpose of classification.
  # Block 1 (<0 and <2500)
  b = np.where(positions_new[:,0:1]<0) 
  positions_block1 = positions_new[b[0]]
  csi_block1 = csi_new[b[0]]

  b = np.where(positions_block1[:,1:2]<2500) 
  positions_block1 = positions_block1[b[0]]
  csi_block1 = csi_block1[b[0]]

  ID_block_1 = np.ones(len(positions_block1),dtype = int)

  # Block 2 (>0 and <2500)
  b = np.where(positions_new[:,0:1]>0) 
  positions_block2 = positions_new[b[0]]
  csi_block2 = csi_new[b[0]]

  b = np.where(positions_block2[:,1:2]<2500) 
  positions_block2 = positions_block2[b[0]]
  csi_block2 = csi_block2[b[0]]

  ID_block_2 = 2*np.ones(len(positions_block2),dtype = int)

  # Block 3 (<0 and >2500)
  b = np.where(positions_new[:,0:1]<0) 
  positions_block3 = positions_new[b[0]]
  csi_block3 = csi_new[b[0]]

  b = np.where(positions_block3[:,1:2]>2500) 
  positions_block3 = positions_block3[b[0]]
  csi_block3 = csi_block3[b[0]]

  ID_block_3 = 3*np.ones(len(positions_block3),dtype = int)

  # Block 4 (>0 and >2500)
  b = np.where(positions_new[:,0:1]>0) 
  positions_block4 = positions_new[b[0]]
  csi_block4 = csi_new[b[0]]

  b = np.where(positions_block4[:,1:2]>2500) 
  positions_block4 = positions_block4[b[0]]
  csi_block4 = csi_block4[b[0]]

  ID_block_4 = 4*np.ones(len(positions_block4),dtype = int)

  # ** I check whether the split is correctly done. **.
  # split_bar = '='*20
  # print(f"{split_bar} CSI OLD values {split_bar} ")
  # print(pd.DataFrame(csi[:,0,0]))
  # print(f"{split_bar} CSI NEW values {split_bar} ")
  # print(pd.DataFrame(csi_new_final[:,0,0]))

  # print(f"{split_bar} POSITIONS OLD values {split_bar} ")
  # print(pd.DataFrame(positions))
  # print(f"{split_bar} POSITIONS NEW values {split_bar} ")
  # print(pd.DataFrame(positions_new_final))


  csi = np.concatenate([csi_block1, csi_block2, csi_block3, csi_block4], axis=0)
  positions_new_final = np.concatenate([positions_block1[:,0:2], positions_block2[:,0:2], positions_block3[:,0:2], positions_block4[:,0:2]], axis=0)
  ID_blocks_new = np.expand_dims(np.concatenate([ID_block_1, ID_block_2, ID_block_3, ID_block_4]),1)
  
  # @title [RUN] Process Dataset

  # KU Luwen Version
  location_data = positions_new_final

  print("Shape after einsum: ", csi.shape, location_data.shape)

  # Select number of subcarriers we want to process (INstead of 100 that are in the dataset)
  every_which = 1 #@param {type:"integer"}
  listSubs = np.arange(0,100,every_which)
  csi_new = csi[:,:,listSubs]
  num_subs = len(listSubs)
                
  # Get real, imag and abs values. First vectorize and then reshape back
  csi_real_pre = np.real(csi_new.view('complex')) #/ np.max(np.real(csi_new.view('complex')))                  
  csi_real_pre_shaped = np.reshape(csi_real_pre,[csi_real_pre.shape[0], 64, num_subs])
  csi_imag_pre = np.imag(csi_new.view('complex')) #/ np.max(np.imag(csi_new.view('complex')))
  csi_imag_pre_shaped = np.reshape(csi_imag_pre,[csi_imag_pre.shape[0], 64, num_subs])
  csi_abs_pre = np.abs(csi_new.view('complex')) #/ np.max(np.abs(csi_new.view('complex')))                     
  csi_abs_pre_shaped = np.reshape(csi_abs_pre,[csi_abs_pre.shape[0], 64, num_subs])

  print(csi_real_pre_shaped.shape,csi_imag_pre_shaped.shape, csi_abs_pre_shaped.shape)                

  del csi_real_pre
  del csi_imag_pre
  del csi_abs_pre
  # Reshape back and normalize
  csi = np.stack((csi_real_pre_shaped,csi_imag_pre_shaped, csi_abs_pre_shaped), axis=3, out=None)       

  print(csi.shape, location_data.shape)

  scalar = MinMaxScaler()
  scalar = scalar.fit(location_data)

  print("\nValues for Transformations\n. Means: {},{},{}, Stds: {},{},{}".format(np.mean(csi_real_pre_shaped), np.mean(csi_imag_pre_shaped), np.mean(csi_abs_pre_shaped),np.std(csi_real_pre_shaped),np.std(csi_imag_pre_shaped),np.std(csi_abs_pre_shaped)))
  print("\nValues for Transformations\n. Mins: {},{},{}, Max: {},{},{}".format(np.min(csi_real_pre_shaped), np.min(csi_imag_pre_shaped), np.min(csi_abs_pre_shaped),np.max(csi_real_pre_shaped),np.max(csi_imag_pre_shaped),np.max(csi_abs_pre_shaped)))

  '''

  scalar = MinMaxScaler()
  scalar = scalar.fit(location_data_and_classes[:,0:2])

  #tx_transform = scalar.fit_transform(np.expand_dims(tx_for_normalization[:,0], axis =1))
  tx_transform = scalar.fit_transform(location_data_and_classes[:,0:2])
  # Concat location IDs after scaling
  tx_transform = np.concatenate((tx_transform,location_data_and_classes[:,2:3]), axis=1) 

  print(csi2.shape, tx_transform.shape)

  X_train, x_test, Y_train, y_test = train_test_split(csi2[:,:,0:100,:], tx_transform, stratify=tx_transform[:,2:3], test_size=5000) #locations_ID2 was replaced by tx...

  
  #test_size = 15000 used for ablations, otherwise 0.005:
  x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train[:,2:3], test_size=2000) #1k: 22k; 5k: 18k; 10k: 22k
  print(f"Shapes: {x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape}")
  print(f"Train Data: {len(x_train)}")
  print(f"Validation Data: {len(x_val)}")
  print(f"Test Data: {len(x_test)}")
  print("Unique spots for classification: ",np.unique(tx_transform[:,2:3]))


  #train_and_val_batchsize = args.train_val_batchsize #@param {type:"integer"}

  X_train2 = np.einsum('basc->bcas', x_train)
  X_test2 = np.einsum('basc->bcas', x_test)
  X_val2 = np.einsum('basc->bcas', x_val)


  tensor_x = torch.tensor(X_train2).float()
  tensor_test_x = torch.tensor(X_test2).float()
  tensor_val_x = torch.tensor(X_val2).float()

  tensor_y = torch.tensor(y_train).float()#.cuda()
  tensor_test_y = torch.tensor(y_test).float()#.cuda()
  tensor_val_y = torch.tensor(y_val).float()#.cuda()

  train_dataset = TensorDataset(tensor_x,tensor_y) 

  test_dataset = TensorDataset(tensor_test_x,tensor_test_y)# create your datset

  val_dataset = TensorDataset(tensor_val_x,tensor_val_y) # create your datset

  print('done')
  return train_dataset, test_dataset, val_dataset, scalar



# @title [RUN] Classifier

@torch.no_grad()
def classifier_knn(train_embeddings, train_spots, test_embeddings, test_spots, k=20, T=0.5, C_spots=360):
  """
  A classifier for spot estimation.
  Arguments:
      train_embeddings
      train_spots
      test_embeddings 
      test_spots
      k (int, optional): NNs (default = 20)
      T (float, optional): Weight (Default = 0.5).
      C_spots (int, optional): The number of classes in the classification task. Default is 360.
  
  Returns:
      tuple: Two values (In KUL datasets C_spot = 4. Hence, comment top_5):
          top_1 (float): Top-1 accuracy of the classifier.
          top_5 (float): Top-5 accuracy of the classifier.
  """
  # Initialize counters
  top_1, top_5, total = 0.0, 0.0, 0
  
  train_embeddings = train_embeddings.t()
  # Determine the number of channels and the number of chunks for evaluating 
  num_test_channels, num_correct = test_spots.shape[0], 100
  h_per_chunk = num_test_channels // num_correct
  
  # Initialize the retrieval one hot
  spot_retrieval_one_hot = torch.zeros(k, C_spots).to(train_embeddings.device)
  
  # Iterate over test channels
  for c_channel in range(0, num_test_channels, h_per_chunk):
      # Get the embeddings
      embeddings = test_embeddings[c_channel : min((c_channel + h_per_chunk), num_test_channels), :]
      targets = test_spots[c_channel : min((c_channel + h_per_chunk), num_test_channels)]
      batch_size = targets.shape[0]
      
      # Evaluate the NNs between the test and train embeddings
      nearest_channs = torch.mm(embeddings, train_embeddings)
      sim_score, sim_idxs = nearest_channs.topk(k, largest=True, sorted=True)
      
      # Retrieve k-NNs for each test channel
      candidates = train_spots.view(1, -1).expand(batch_size, -1)
      retrieved_neighbors = torch.gather(candidates, 1, sim_idxs)
      
      # Evaluate spot probabilities
      spot_retrieval_one_hot.resize_(batch_size * k, C_spots).zero_()
      spot_retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
      sim_score_transform = sim_score.clone().div_(T).exp_()
      probs = torch.sum(torch.mul(spot_retrieval_one_hot.view(batch_size, -1, C_spots), sim_score_transform.view(batch_size, -1, 1),),1,)
      _, predictions = probs.sort(1, True)

      # Find/Get predictions that match the target-spot
      correct = predictions.eq(targets.data.view(-1, 1))
      top_1 = top_1 + correct.narrow(1, 0, 1).sum().item()
      top_5 = top_5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top_5 does not make sense if k < 5
      total += targets.size(0)
  top_1 = top_1 * 100.0 / total
  top_5 = top_5 * 100.0 / total # For KUL no Top5
  return top_1, top_5

# To reset weights, lets initialize initially with xavier, latter we call defaul pytorch rand-normal or pre-trained.
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)     

### Get embeddings
def get_embeddings(model, h_channel, which_o='LID'):
    """
    Get a compressed representation of the channel (i.e., an embedding).
    
    Parameters
    ----------
    model : object
        Pre-trained model (or random).
    h_channel : tensor
        H tensor (3, N_r, N_c).
    which_o : str
        Options can be to get only the 'LID' and 'mean'. 
          However, you can select any, does not really matter for SWiT.
    
    Returns
    -------
    tensor
        Embedding (1, D).
    """
    # Get the intermediate layer output.
    o_r = model.get_intermediate_layers(h_channel.unsqueeze(0).cuda(), n=1)[0]
    dim = o_r.shape[-1]
    o_r = o_r.reshape(-1, dim)
    
    # Either 'LID' or 'mean'. However, you can select "any" of the "subcarrier" representations. All should give more or less similar results.
    if which_o == 'LID':
        o_r = o_r[0:1, ...]
    else:
        o_r = torch.mean(o_r, 0, True)
        
    return o_r


def plot_classifier_knn_results(results_path, experiment_name, plot_here = True):
    """
    Plot classifier results.
    
    Parameters:
    - results_path (str): path to results
    - experiment_name (str): name of the experiment

    Returns:
    None
    """
    with open(Path(results_path) / "knn_results.txt") as f:
        knn_results_js = json.loads(f.readlines()[-1])
    
    print("Knn results : ", type(knn_results_js))
    print(knn_results_js)
    
    knn_results_pd = pd.DataFrame.from_dict(knn_results_js)
    snsPlot = sns.lineplot(data=knn_results_pd, linestyle = 'dotted', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="b", markersize=16)
    snsPlot.set_xticks(range(len(knn_results_pd))) 
    snsPlot.set_xticklabels(knn_results_pd['Weights'], fontsize = 9)
    plt.title(experiment_name, fontsize=14)
    plt.xlabel("Random vs SWiT over Iterations (Top-5 not relevant)", fontsize = 13)
    plt.ylabel("Top-k Accuracy [%]", fontsize = 13)
    plt.minorticks_off()
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.savefig(results_path+'/'+'classifier_rand_vs_models_w_aug.png')
    plt.savefig(results_path+'/'+'classifier_rand_vs_models_w_aug.pdf', dpi=400)
    print('plt.savefig')
    if plot_here==True:
      plt.show()
    plt.close(1)


def get_args_values(args):
    args_dict = vars(args)
    args_strings = []
    for key, value in args_dict.items():
        args_strings.append("{}: {}".format(key, str(value)))
    return "\n".join(args_strings)


if __name__ == '__main__':
  # Load the the config file
  with open("config.json", "r") as f:
      config = json.load(f)

  results_path = config["project_path"]+"/results/"+config["experiment_name"]
  if not os.path.exists(results_path):
    print("There is no results path for this experiment. Thus, I will create a folder to store the results.\n")
    os.makedirs(results_path)
    print("results path created: ", results_path)

  sys.path.append(config["project_path"])

  CHECKPOINT_PATH = config["project_path"]+"/saved_models/"+config["experiment_name"]  
                  # Path to the folder where the pretrained models are saved
  if not os.path.exists(CHECKPOINT_PATH):
    print("There is no checkpoint path for this experiment. Thus, I will create a folder to store the checkpoints.\n")
    os.makedirs(CHECKPOINT_PATH)
    print("CHECKPOINT_PATH created: ", CHECKPOINT_PATH)
  parser = argparse.ArgumentParser(f'Evaluation on {config["dataset_to_download"]} dataset.')
  parser.add_argument('--experiment_name', type=str, default=config['experiment_name'], help='Name of this experiment.')
  parser.add_argument('--dataset_to_download', type=str, default=config['dataset_to_download'], help='Path to dataset to load.')
  parser.add_argument('--saved_dataset_path', type=str, default=config['saved_dataset_path'], help='Path to dataset to load.')
  parser.add_argument('--sub_dataset_file_csi',type=str, default=config['sub_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
  parser.add_argument('--sub_dataset_file_loc',type=str, default=config['sub_loc_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
  parser.add_argument('--realMax', type=float, default=config['realMax'], help='Max value of real part for the whole dataset')
  parser.add_argument('--imagMax', type=float, default=config['imagMax'], help='Max value of imag part for the whole dataset')
  parser.add_argument('--absMax', type=float, default=config['absMax'], help='Max value of abs part for the whole dataset')
  parser.add_argument('--model_name', type=str, default=config['arg2_model_name'], help='WiT-based transformer.')
  parser.add_argument("--encoder", type=str, default=config['arg1_encoder'], help='We use momentum target encoder.')
  parser.add_argument('--number_antennas', type=int, default=config['arg3_Nr'], help='Number of antenna elements per subcarrier.')
  parser.add_argument('--total_subcarriers', type=int, default=config['arg4_Nc'], help='Total number of subcarriers.')
  parser.add_argument('--eval_subcarriers', type=int, default=config['arg5_Nc_prime'], help='Selected number of subcarriers.')
  parser.add_argument('--weights_path', type=str, default=CHECKPOINT_PATH+'/checkpoint.pth', help="Path to load pre-trained weights.")
  parser.add_argument('--train_val_batchsize', type=int, default=config['train_and_val_batchsize'], help='Batch size for train and val. For test, we use 1.')
  parser.add_argument("--knn", type=int, default=config['arg_classifier_k'], help="k NNs, default 20.")
  parser.add_argument("--c_spots", type=int, default=config['arg_classifier_c_spots'], help="Number of spots/classes. For KUL: 4; For S: 360; For HB: 406")
  parser.add_argument("--pth_names_classifier", type = list, default = config['pth_names_classifier'], help= "A list of saved models (or epoch checkpoints) to evaluate. Include some random names, to understand the gain.")
  #parser.add_argument("--h_slice",type=tuple, default=(64,1), help="Top kNNs")
  args = parser.parse_args(args=[])

  args.h_slice = (64,1)

  print(f'***Configuration****\n',"\n",get_args_values(args))

  print('\n\n',args.saved_dataset_path)

  train_dataset, test_dataset, val_dataset, scalar = prep_data_load(args)

  # Get Features from pre-trained model.
  datasets = [train_dataset, test_dataset]

  global_transfo2_test = channel_transforms('classifier',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)

  # To save hyperparameter:
  if len(global_transfo2_test.transform) == 1:
    Transf = str(global_transfo2_test.transform[0])
  elif len(global_transfo2_test.transform) == 2:  
    Transf = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1])
  elif len(global_transfo2_test.transform) == 3:  
    Transf = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1]) ,str(global_transfo2_test.transform[2])  

  knn_results = {'Transformations':[],'Weights':[],'Top1': [],'Top5':[]}

  # Eval. over different models.
  epochs_to_eval = args.pth_names_classifier #['checkpoint_Rnd1','checkpoint0010','checkpoint_Rnd2','checkpoint0030','Rand3','checkpoint0040', 'Rand4','checkpoint0080']#,'checkpoint_nlos,checkpoint_nlos_new']#,'rand','checkpoint0070','checkpoint0090','rrr','checkpoint']

  model = wits.__dict__[args.model_name](h_slice=(args.number_antennas, 1), num_classes=2)

  for j in range(len(epochs_to_eval)):
    
    ### 1. Reset Weights to xavier (to avoid weight accum.) ####
    model.apply(weights_init) 

    ### 2. Load pre-trained weights ###
    args.weights_pth = Path(CHECKPOINT_PATH) / (epochs_to_eval[j]+'.pth') 

    print(f"Model {args.model_name} {args.number_antennas}x{1} built.")
    #model.get_intermediate_layers
    print(f"\nWeights that are supposed to find at : \n {args.weights_pth}\n")
    logger.load_pretrained_weights(model, args.weights_pth, args.encoder)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Avoid to get embeddings from a validation set. Only for train and test.
    for i in range(2):
      data_loader = DataLoader(
          datasets[i],
          batch_size=1,
          num_workers=1,
          pin_memory=True,
          drop_last=True,
      )    
      print(f"Data loaded: there are {len(train_dataset)} and {len(test_dataset)} CSI Samples.")

      x_lst = []
      embeddings = []
      spot_lst = []
      for tensor_channel, spot in data_loader:
        h_input = tensor_channel[0:1]
        h_input = global_transfo2_test(h_input)
        h_input = torch.squeeze(h_input, 0)
        h_input_feat = get_embeddings(model.to('cuda'), h_input.to('cuda'), which_o='LID').T
        embeddings.append(h_input_feat.flatten().unsqueeze(0).detach().cpu().numpy())
        x_lst.append(tensor_channel.detach().cpu().numpy())
        spot_lst.append(spot.detach().cpu().numpy())
      print(h_input_feat.shape, h_input.shape)

      if i==0:      
        print("x_lst_num",len(x_lst))
        train_embeddings = np.concatenate(embeddings, axis=0 )
        train_spots = np.concatenate(spot_lst, axis=0 )
      elif i==1:
        test_embeddings = np.concatenate(embeddings, axis=0)
        test_spots = np.concatenate(spot_lst, axis=0)

    print(train_embeddings.shape, train_spots.shape, test_embeddings.shape,test_spots.shape )
    
    #top_5 = 100 # Hard coded, since for KUL c_spots = 4.
    print('\n ***Be careful when using other datasets! Uncomment parts where top-5 is set to the value of 100.')
    k=20

    train_embeddings, test_embeddings, train_spots, test_spots = torch.from_numpy(train_embeddings), torch.from_numpy(test_embeddings), torch.from_numpy(train_spots), torch.from_numpy(test_spots)
    
    # Only top_1 return. Top_5 is commented. Careful when evaluated on the S and HB datasets.
    top_1, top_5 = classifier_knn(train_embeddings, train_spots[:,2:3].type(torch.long), test_embeddings, test_spots[:,2:3].type(torch.long), args.knn, 0.7, C_spots=args.c_spots)
    print(f'\nEvaluated Epoch is {epochs_to_eval[j]}.\n')
    print(f"{k}-NN classifier result: Top1: {top_1}, Top5: {top_5}")
    knn_results['Weights'].append(epochs_to_eval[j])
    knn_results['Top1'].append(top_1)
    knn_results['Top5'].append(top_5)
    knn_results['Transformations'].append(Transf)

  for i in range(len(knn_results['Weights'])):
    knn_results['Weights'][i] = knn_results['Weights'][i][-5:]

  if logger.is_main_process():
      with (Path(results_path) / "knn_results.txt").open("a") as f:
        f.write('\n')
        f.write(json.dumps(knn_results) + "\n")

  # Load saved data (dict) and plot and save
  plot_classifier_knn_results(results_path, args.experiment_name, plot_here = True)