# For system.
import sys
sys.path.append('./')
import numpy as np
from tqdm import tqdm
import os
from io import BytesIO
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import importlib
from urllib.request import urlopen
import argparse
import json
from pathlib import Path

# Reading file
from scipy.io import loadmat
import pandas as pd
import h5py as h5
import pickle as pkl

# Others in the mean time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

# For torch.
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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

from models.ChannelTransformationModule import ChannelTransformationModule as channel_transforms

import torch
from torch import nn
import pickle as pkl
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Warnings
import warnings

warnings.simplefilter('ignore', UserWarning)

def train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, project_path, device, global_transform, num_epochs=1800, best_vloss = 0.0009, CHECKPOINT_PATH='./', experiment_name='Test'):
    """"
    Train/evaluate the model for `num_epochs` epochs.
    Parameters:
    model = wit_model
    train_loader = training data loader
    val_loader = validation data loader
    loss_fn = loss function (criterion)
    optimizer = optimizer (optim.AdamW(wit_model.parameters(), lr=learning_rate))
    writer = writer (tensorboard SummaryWriter to write logs)
    project_path = main project path, where methods and log folders are located. './WiT' in this case. Check the notebook for the example.
    device = torch.device("cuda") 
    num_epochs = 1800 (in paper), epochs
    best_vloss = 0.0009 (when to start considering to save the model checkpoints)
    CHECKPOINT_PATH = Where to save models.
    experiment_name = experiment_name (name of experiment for saving logs)
    -----------
    """
    
    epoch_number = 0
    best_vloss = best_vloss
    training_stats = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_train_accuracy = 0
        '''when using custom dataloader:
        for batch, sample_batched in enumerate(train_loader):
             H = sample_batched['csi']
             u = sample_batched['label'][:,0:2]
        '''
        for H, u in train_loader: 
            H = global_transform(H) #with augmentation.
            H = H.float().to(device)
            u = u[:,0:2].float().to(device) # Here, I have u[:,0:2] because in the dataloader, the labels also have LID (num_samples x 3).
            u_hat = model(H,)
            loss = loss_fn(u_hat, u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            for H_val, u_val in val_loader:
                H_val = global_transform(H_val) #with augmentation.
                H_val = H_val.float().to(device)
                val_u = u_val[:,0:2].float().to(device)
                #val_u = u[:,2].type(torch.LongTensor).to(device)
                val_u_hat = model(H_val)
                val_loss = loss_fn(val_u_hat, val_u)

                epoch_val_loss += val_loss / len(val_loader)

        # Save every 100 epochs.                
        if epoch % 75 == 0:
            model_name_epoch = 'model_fine_tuner_{}_{}'.format(timestamp, epoch_number)
            model_path = CHECKPOINT_PATH+"/"+model_name_epoch
            if config['save_new_models']==True:
                torch.save(model.state_dict(), model_path) # Uncomment later.

        ''' When using a tensorboard writer to monitor the data
        if epoch_val_loss < best_vloss:
            best_vloss = epoch_val_loss
            model_name_epoch = 'model_{}_{}'.format(timestamp, epoch_number)
            model_path = CHECKPOINT_PATH+"/"+model_name_epoch
            torch.save(model.state_dict(), model_path)
            #"HERE?"

        writer.add_scalar("Training Accuracy", epoch_accuracy, epoch)
        writer.add_scalar("Validation Accuracy", epoch_val_accuracy, epoch)   
        writer.add_scalars('Training vs. Validation Accuracy',
                       { 'Training' : epoch_accuracy, 'Validation' : epoch_val_accuracy },
                       epoch)   
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : epoch_loss, 'Validation' : epoch_val_loss },
                        epoch)                             
        #Track best performance, and save the model's state
        '''
        epoch_number += 1
        #writer.flush()
        if epoch % 50 == 0:
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.8f} - acc : {epoch_train_accuracy:.8f} - val_loss : {epoch_val_loss:.8f} val_acc : {epoch_val_accuracy:.8f} \n"
            )
        epoch_stats = {
        'train_loss': epoch_loss.detach().cpu().numpy(), 'val_loss': epoch_val_loss.detach().cpu().numpy(), 'epoch': epoch+1
        }
        training_stats = update_stats(training_stats, epoch_stats)
        #writer.flush()

    with open(project_path+'/runs/trainig_stats_{}'.format(experiment_name), 'wb') as f:       #this will save the list as "results.pkl" which you can load in later 
                pkl.dump(training_stats, f)          #as a list to python            


def test_loop(model, test_loader, global_transform, loss_fn, device, scalar=None): # when global_transform is used.
    """"
    Test the model in the test data loader.
    Parameters:
    model = wit_model
    test_loader = test data loader
    loss_fn = loss function (criterion)
    device = torch.device("cuda") 
    scalar= None or scalar .If None, there is no transformation of ground-truth (i.e., normalization 0-1). Otherwise, check scaler function MinMax() defined when loading the dataset. For regression, we always perform normalization.    -----------
    """
    epoch = .0
    listRMSE = []
    listMAE = []
    x_coordinate_actual = []
    y_coordinate_actual = []
    x_coordinate_estimated = []
    y_coordinate_estimated = []
    criterion_MAE = nn.L1Loss()
    model.eval() # **** Must eval() when reporting the results (for consistency). Otherwise, it keeps the stochasticity of the model.
    with torch.no_grad():
        epoch_test_loss = 0
        for data, label in tqdm(test_loader):
            data = global_transform(data) #with channel transformations.
            data = data.float().to(device)
            label = label[:,0:2].float().to(device)
            test_output = model(data)
            test_loss = loss_fn(test_output, label)

            test_output_numpy = test_output.detach().cpu().numpy()
            labels_output_numpy = label.detach().cpu().numpy()

            if scalar==None:
                y_test_original = labels_output_numpy
                y_test_estimated = test_output_numpy
            else:
                y_test_original = scalar.inverse_transform(labels_output_numpy)
                y_test_estimated = scalar.inverse_transform(test_output_numpy)
                
            x_coordinate_actual.append(y_test_original.item(0))
            y_coordinate_actual.append(y_test_original.item(1))
            x_coordinate_estimated.append(y_test_estimated.item(0))
            y_coordinate_estimated.append(y_test_estimated.item(1))

            test_loss_rescaled_MAE = criterion_MAE(torch.Tensor(y_test_estimated).float(),torch.Tensor(y_test_original).float())
            test_loss_rescaled_RMSE = loss_fn(torch.Tensor(y_test_estimated).float(),torch.Tensor(y_test_original).float())

            listRMSE.append(torch.sqrt(test_loss_rescaled_RMSE).item())
            listMAE.append(test_loss_rescaled_MAE.item())
            epoch += 1
            epoch_test_loss += test_loss / len(test_loader)

    listRMSE = np.array(listRMSE)
    listMAE = np.array(listMAE)
    x_coordinate_actual = np.array(x_coordinate_actual)
    y_coordinate_actual = np.array(y_coordinate_actual)
    x_coordinate_estimated = np.array(x_coordinate_estimated)
    y_coordinate_estimated = np.array(y_coordinate_estimated)
    actual_coordinates = np.stack((x_coordinate_actual,y_coordinate_actual), axis = 1)
    estimated_coordinates = np.stack((x_coordinate_estimated,y_coordinate_estimated), axis = 1)

    # return RMSE (based on nn.MSE()), MAE (based on nn.L1()), actual position coordinates, and estimated position coordinates.
    return listRMSE, listMAE, actual_coordinates, estimated_coordinates


def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
    epoch_stats: dict of metrics for one epoch
    training_stats: dict of lists for metrics during the training
    Returns:
    updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats

def plot_results_ecdf(listMAE, results_path, experiment_name, save=True, error_type = "MAE"):
    ## ECDF ###
    x_N3 = np.sort(listMAE,axis=None)
    y_N3 = np.arange(1,len(x_N3)+1)/len(x_N3)
    plt.figure(1)
    plt.plot(x_N3,y_N3,"k")
    #plt.xscale('symlog', linthreshy=0.01)
    plt.xscale('log')
    plt.ylabel('ECDF',fontsize=13)
    plt.xlabel('MAE',fontsize=13)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.title(experiment_name, fontsize = 13)
    if save==True:
        plt.savefig(results_path+'/'+error_type+'_ecdf.png')
        plt.savefig(results_path+'/'+error_type+'_ecdf.pdf', dpi=400)
    plt.show() # comment this to avoid popping plots in a local machine.
    print('plt.savefig')
    plt.close(1)

def print_table_results(actual_coordinates, estimated_coordinates, experiment_name, listMAE, listRMSE, results_path, save=True):
    data = [[1, experiment_name, np.mean(listMAE), np.percentile(listMAE, 95), np.mean(listRMSE)]]
    print(tabulate(data, headers=["Experiment name (model)", "MAE (L1Loss)", "95-th percentile","RMSE (MSELoss)"]))
    if save==True:
        print("\n\n### Save Summarized results in .txt file.####")
        with open(results_path+'/summarized_res.txt', 'w') as f:
            f.write(tabulate(data, headers=["Experiment name (model)", "MAE (L1Loss)", "95-th percentile","RMSE (MSELoss)"]))


def print_table_model_hyperparameters(training_args, folders, model_args, results_path, save=True):
    data = [["Training args", training_args],["Folders", folders],["Model args", model_args]]
    if save==True:
        print("\n\n### Save model hyperparameter results in .txt file.####")
        with open(results_path+'/model_hyper.txt', 'w') as f:
            f.write(tabulate(data))

############ When Using SSL and Linear Eval or Fine Tune #################

def plot_results_actual_vs_estimate_ssl(actual_coordinates, estimated_coordinates, results_path, experiment_name, save=True, plot_name = 'Actual_vs_Estimated'):
    """ Create plot to show actual vs estimated points from testing data
    """
    plt.plot(actual_coordinates[:,0],actual_coordinates[:,1],'.b',markersize = 2.0, label = 'Actual')
    plt.plot(estimated_coordinates[:,0],estimated_coordinates[:,1],'.r', markersize = 0.2, label = "Estimated")
    plt.ylabel('x [m]',fontsize=13)
    plt.xlabel('y [m]',fontsize=13)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    from matplotlib import container
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    ax.legend(handles, labels, loc='best', prop={'size': 12}, numpoints=1, fancybox=False)
    plt.title(experiment_name, fontsize = 13)
    if save==True:
        plt.savefig(results_path+"/"+plot_name+".png")
        plt.savefig(results_path+"/"+plot_name+".pdf",dpi=400)
        #plt.savefig("test.png")        
    plt.show() # comment this to avoid popping plots in a local machine.
    plt.close()   


# For SSL, I am saving multiple results for the same dataset. Lets use the below method to save results. It only changes the name_summarized.
def print_table_results_ssl(ModelName, transformation_type, experiment_name, listMAE, listRMSE, results_path, save=True, name_summarized="summarized_res"):
    data = [[1, ModelName, transformation_type, experiment_name, np.mean(listMAE), np.percentile(listMAE, 95), np.mean(listRMSE)]]
    print(tabulate(data, headers=['ModelName',"Transformation","Experiment name (model)", "MAE (L1Loss)", "95-th percentile","RMSE (MSELoss)"]))
    if save==True:
        print("\n\n### Save Summarized results in .txt file.####")
        with open(results_path+'/'+name_summarized+'.txt', 'w') as f:
            f.write(tabulate(data, headers=['ModelName',"Transformation during Testing","Experiment name (model)", "MAE (L1Loss)", "95-th percentile","RMSE (MSELoss)"]))



def load_model(model, CHECKPOINT_PATH, name_model_saved, optimizer):
    print('\n\n Loading model from ', CHECKPOINT_PATH+"/"+name_model_saved)
    model.load_state_dict(torch.load(CHECKPOINT_PATH+"/"+name_model_saved))


def plot_cumulate_stats(global_training_stats, training_stats, figsize=(5, 5), name=""):
    """ Create a plot for metric (loss or acc) in training_stats
    """
    for key,val in training_stats.items():
      global_training_stats.update({f'{key}_{name}': val})
    stats_names = [key[6:] for key in global_training_stats.keys() if key.startswith('train_')]
    f, ax = plt.subplots(len(stats_names), 1, figsize=figsize)
    if len(stats_names)==1:
        ax = np.array([ax])
    for key, axx in zip(stats_names, ax.reshape(-1,)):
        axx.plot(
            global_training_stats[f'epoch_{name}'],
            global_training_stats[f'train_{key}'],
            label=f"Training {key}")
        axx.plot(
            global_training_stats[f'epoch_{name}'],
            global_training_stats[f'val_{key}'],
            label=f"Validation {key}")
        axx.set_xlabel("Training epoch")
        axx.set_ylabel(key)
        axx.legend()
    plt.title(name)
    return global_training_stats
    
def plot_stats(training_stats, results_path, save_results, name="", figsize=(5, 5)):
    """ Create a plot for metric in training_stats
    """
    stats_names = [key[6:] for key in training_stats.keys() if key.startswith('train_')]
    f, ax = plt.subplots(len(stats_names), 1, figsize=figsize)
    if len(stats_names)==1:
        ax = np.array([ax])
    for key, axx in zip(stats_names, ax.reshape(-1,)):
        axx.plot(
            training_stats['epoch'],
            training_stats[f'train_{key}'],
            label=f"Training {key}")
        axx.plot(
            training_stats['epoch'],
            training_stats[f'val_{key}'],
            label=f"Validation {key}")
        axx.set_xlabel("Training epoch")
        axx.set_ylabel(key)
        axx.set_ylim((-0.001, 0.110))
        #axx.set_xlim((500,1910))
        axx.legend()
    plt.title(name)
    plt.show()
    if save_results==True:
        plt.savefig(results_path+'/'+name+".png")
    plt.close()

def get_args_values(args):
    args_dict = vars(args)
    args_strings = []
    for key, value in args_dict.items():
        args_strings.append("{}: {}".format(key, str(value)))
    return "\n".join(args_strings)


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
    See test_classifier for other processing steps. Here we load only a sample.
    '''
    with open(Path(args.saved_dataset_path)/args.sub_dataset_file_csi, 'rb') as f1:
        csi2 = np.load(f1)
        f1.close()
    with open(Path(args.saved_dataset_path)/args.sub_dataset_file_loc, 'rb') as f2:
        location_data_and_classes = np.load(f2)    
        f2.close()

    # Initial split for test dataset and scaling coordinates. Training data-regimes are defined during the training loop.
    scalar = MinMaxScaler()
    scalar = scalar.fit(location_data_and_classes[:,0:2])

    tx_transform = scalar.fit_transform(location_data_and_classes[:,0:2])
    # Concat location IDs after scaling
    tx_transform = np.concatenate((tx_transform,location_data_and_classes[:,2:3]), axis=1) 

    print(csi2.shape, tx_transform.shape)

    X_train, x_test, Y_train, y_test = train_test_split(csi2[:,:,0:100:3,:], tx_transform, stratify=tx_transform[:,2:3], test_size=5000) #locations_ID2 was replaced by tx...

    return X_train, x_test, Y_train, y_test, scalar

# To reset weights, lets initialize initially, latter we call defaul pytorch rand-normal or pre-trained.
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


class LinearHead(nn.Module):
    def __init__(self, embed_size, dim_out=2):
        super().__init__()
        self.linear = nn.Linear(embed_size, dim_out)
    def forward(self, x):
        x = self.linear(x)
        return x

###### A wrapper for the regressor and backbone.
class FineTuneWrapper(nn.Module):
    ''' Use 'LID' or averaged over subcarrier embeddings.'''
    def __init__(self, backbone, linear_regressor):
        super(FineTuneWrapper, self).__init__()
        self.backbone = backbone
        self.linear_regressor = linear_regressor

    def forward(self, x):
        output1 = self.backbone(x)
        #output = torch.mean(output1[...,:], dim=1)
        output = torch.squeeze(output1[:,0:1,:], dim=1)
        output = self.linear_regressor(output)
        return output

# To reset weights, lets initialize initially with xavier.
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

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

    parser = argparse.ArgumentParser(f'Linear Regressor Evaluation on {config["dataset_to_download"]} dataset.')
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
    parser.add_argument('--weights_pth', type=str, default=CHECKPOINT_PATH+config['pth_name_linear_tuner'], help="Path to load pre-trained weights. Write the name of the model (checkpoint).")
    parser.add_argument('--train_val_batchsize', type=int, default=config['train_and_val_batchsize'], help='Batch size for train and val. For test, we use 1.')
    #parser.add_argument("--h_slice",type=tuple, default=(64,1), help="Top kNNs")
    parser.add_argument('--criterion', type=str, default=config['criterion'], help='Default (and the only one supported) MSE.')
    parser.add_argument('--device', type=str, default=config['device'], help='cuda or cpu.')
    parser.add_argument('--epochs_linear', type=int, default=config['epochs_linear'], help='Commonly for linear case we do 500 epochs.')
    parser.add_argument('--best_vloss', type=int, default=config['best_vloss'], help='Validation loss when to create a checkpoint of the model.')
    parser.add_argument('--data_regimes', type=list, default=config['data_regimes'], help='A list of strings with data regimes (Only three regimes implemented: ["1k", "5k", "10k"]).')
    parser.add_argument('--save_results', type=bool, default=config['save_results'], help='Save all results (inlcuding plots, tables,....).')
    parser.add_argument('--learning_rate_eval', type=float, default=config['learning_rate_eval'], help='Learning rate for fine-tuner or train from scratch.')

    args = parser.parse_args(args=[])

    args.h_slice = (64,1)

    print(f'***Configuration****\n',"\n",get_args_values(args))

    print('\n\n',args.saved_dataset_path)

    X_train, x_test, Y_train, y_test, scalar = prep_data_load(args)

    global_transfo2_test = channel_transforms('regressor',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)

    # To save hyperparameter:
    if len(global_transfo2_test.transform) == 1:
        transformation_type = str(global_transfo2_test.transform[0])
    elif len(global_transfo2_test.transform) == 2:  
        transformation_type = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1])
    elif len(global_transfo2_test.transform) == 3:  
        transformation_type = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1]) ,str(global_transfo2_test.transform[2])  

    print(transformation_type)

    #Define a Loss function and optimizer

    if args.criterion == "MSE":
        criterion = nn.MSELoss()
    device = args.device
    epochs = args.epochs_linear
    epoch_number = 0
    best_vloss = args.best_vloss 
    learning_rate = args.learning_rate_eval

    data_regimes = args.data_regimes    

    ###### 2. Instatiate the regressor
    linear_regressor = LinearHead(embed_size=384, dim_out=2)  

    ################## 0. Reset Pre-trained model ##################
    print(f'\n\n ***** 0. Reset Pre-trained model *****\n')
    model = wits.__dict__[args.model_name](h_slice=(args.number_antennas, 1), num_classes=2)
    print(f"Model {args.model_name} {args.number_antennas}x{args.number_antennas} built.")
    ### 0. Reset Weights to xavier ####
    model.apply(weights_init)  #Reset weights to xavier before calling the pretrained weights (to avoid weight accum. for random case)
    model.to(device)
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    #args.weights_pth = Path(CHECKPOINT_PATH) / ('checkpoint_26Jan2023.pth') 

    #model.get_intermediate_layers
    print(f"\nWeights are found at : \n {args.weights_pth}\n")

    ### 0. Now load the pre-trained weights. ####
    logger.load_pretrained_weights(model, args.weights_pth, args.encoder)

    ##### 5. Instatiate the FineTuneWrapper
    model_new = FineTuneWrapper(model,linear_regressor)     
    model_new.to(device)

    for param in model_new.parameters():
        param.requires_grad = True
        print(model_new.backbone.blocks[0].mlp.fc1.weight.grad)


    #Define a Loss function and optimizer

    if args.criterion == "MSE":
        criterion = nn.MSELoss()
    device = args.device
    epochs = args.epochs_linear
    epoch_number = 0
    best_vloss = args.best_vloss 

    data_regimes = args.data_regimes

    # @markdown After the test is complete, consider to save results in /results/"experiment_name"/
    save_results = args.save_results #@param {type:"boolean"}

    for ff in range(len(data_regimes)):
        ################## 0. Reset Pre-trained model ##################
        print(f'\n\n ***** 0. Reset Pre-trained model *****\n')
        model = wits.__dict__[args.model_name](h_slice=(args.number_antennas, 1), num_classes=2)
        print(f"Model {args.model_name} {args.number_antennas}x{args.number_antennas} built.")
        ### 0. Reset Weights to xavier ####
        model.apply(weights_init)  #Reset weights to xavier before calling the pretrained weights (to avoid weight accum. for random case)
        
        #model.get_intermediate_layers
        print(f"\nWeights that are supposed to find at : \n {args.weights_pth}\n")
        ### 0. Now load the pre-trained weights. ####
        logger.load_pretrained_weights(model, args.weights_pth, args.encoder)
        for param in model.parameters():
            param.requires_grad = True

        ##### 5. Instatiate the FineTuneWrapper
        model_new = FineTuneWrapper(model, linear_regressor)     
        model_new.to(device)
        optimizer = optim.AdamW(model_new.parameters(), lr=learning_rate)

        ##################################
        '''
        Be careful with the data regimes. Always check how many samples exactly you have. 
        In the below, I am assuming a dataset with 28k samples in case of KUL and whole dataset size in case of S and H.
        '''
        if data_regimes[ff].startswith('1k'):
            out_of_train_samples = 22001#3334 # 22985, (kul:22001) (s:63212)this is for validation data, so that x_train is 1k if you use 28k dataset
            train_and_val_batchsize = 128 # 
        elif data_regimes[ff].startswith('5k'):
            out_of_train_samples = 18001#1334 # (s:59212) (kul:18001) (h:71200) this is for validation data, so that x_train is 5k if you use 28k dataset
            train_and_val_batchsize = 512
        elif data_regimes[ff].startswith('10k'):
            out_of_train_samples = 13001#334 # (s:54212) (kul:13001) this is for validation data, so that x_train is 10k if you use 28k dataset
            train_and_val_batchsize = 512    

        ################## 1. Get Dataloaders ##################
        print(f'\n\n ***** 1. Get Dataloaders *****\n')

        # Now, do the splitting for train and val.
        x_train, X_val, y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train[:,2:3], test_size=out_of_train_samples)

        # To avoid using a large val sample size, get rid of most of the samples.
        x_valaa, x_val, y_valaa, y_val = train_test_split(X_val, Y_val, stratify=Y_val[:,2:3], test_size=600)

        print(f"Shapes: {x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape}")
        print(f"Train Data: {len(x_train)}")
        print(f"Validation Data: {len(x_val)}")
        print(f"Test Data: {len(x_test)}")  

        ''' Create data loaders.'''
        X_train2 = np.einsum('basc->bcas', x_train)
        X_test2 = np.einsum('basc->bcas', x_test)
        X_val2 = np.einsum('basc->bcas', x_val)
        tensor_x = torch.tensor(X_train2).float()#.cuda() # transform to torch tensor
        tensor_test_x = torch.tensor(X_test2).float()#.cuda() # transform to torch tensor
        tensor_val_x = torch.tensor(X_val2).float()#.cuda() # transform to torch tensor
        tensor_y = torch.tensor(y_train).float()#.cuda()
        tensor_test_y = torch.tensor(y_test).float()#.cuda()
        tensor_val_y = torch.tensor(y_val).float()#.cuda()

        train_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        train_loader = DataLoader(train_dataset, batch_size=train_and_val_batchsize, shuffle=True,drop_last=True) # create your dataloader

        test_dataset = TensorDataset(tensor_test_x,tensor_test_y)# create your datset
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False) # create your dataloader

        val_dataset = TensorDataset(tensor_val_x,tensor_val_y) # create your datset
        val_loader = DataLoader(val_dataset, batch_size=train_and_val_batchsize, shuffle=True,drop_last=True) # create your dataloader


        print(f"Data loaded: there are {len(train_dataset)} and {len(test_dataset)} and {len(val_dataset)} CSI Samples.")

        ################## 4. Training Backbone  ##################
        print(f'\n\n ***** 4. Training *****\n')

        #writer = SummaryWriter(project_path+'/runs/{}'.format(experiment_name))
        
        ### 4.1 Train
        train_eval_loop(model_new, train_loader, val_loader, criterion, optimizer, config['project_path'], device, global_transfo2_test, num_epochs=epochs, best_vloss=best_vloss, CHECKPOINT_PATH=CHECKPOINT_PATH, experiment_name=data_regimes[ff]+args.experiment_name)
        
        with open(config['project_path']+'/runs/trainig_stats_{}'.format(data_regimes[ff]+args.experiment_name), 'rb') as f:
            training_stats = pkl.load(f)   
            plot_stats(training_stats, results_path, save_results, name='FineTuner_'+data_regimes[ff]+"_loss_curve", figsize=(10, 6))

        ################## 5. Test Fine-tuned Backbone  ##################

        print("\n********Now testing on test dataset loader*******\n\n")

        listRMSE, listMAE, actual_coordinates, estimated_coordinates = test_loop(model_new,test_loader,global_transfo2_test,loss_fn=criterion, device = device, scalar = scalar)

        print("\n\nDONE: returning parameters are: listRMSE, listMAE, actual_coordinates, estimated_coordinates \n\n")

        print("\n\nPlotting (and saving) retuls and \n\n")
        plot_results_actual_vs_estimate_ssl(actual_coordinates, estimated_coordinates, results_path, args.experiment_name, save=save_results, plot_name = 'FineTuning_'+data_regimes[ff]+"_Actual_vs_Estimated")

        plot_results_ecdf(listMAE, results_path, args.experiment_name, save=save_results, error_type = 'FineTuning_'+data_regimes[ff]+"_MAE")

        print_table_results_ssl(str(args.weights_pth)[-19:], transformation_type, args.experiment_name, listMAE, listRMSE, results_path, save=save_results, name_summarized = 'FineTuning_'+data_regimes[ff]+'_summarized_res')

        if save_results == True:
            print("\n### Save Errors for ECDF at a later time.####")
            hf = h5.File(results_path+'/FineTuning_'+data_regimes[ff]+"_MAE_errors_logs.h5", 'w')
            hf.create_dataset('listMAE', data=listMAE)
            hf.close()

            print("\n### Save Errors for ECDF at a later time.####")
            hf = h5.File(results_path+'/FineTuning_'+data_regimes[ff]+"_RMSE_errors_logs.h5", 'w')
            hf.create_dataset('listRMSE', data=listRMSE)
            hf.close()