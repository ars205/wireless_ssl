# For system.
import sys
sys.path.append('./')
import numpy as np
from tqdm import tqdm
import os
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
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

from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

# For torch.
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

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


# @title [RUN] Build Regressor Head

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm,_ = x.max(dim = 1, keepdim = True)
        return x / norm

class LinearHead(nn.Module):
    def __init__(self, embed_size, dim_out=2):
        super().__init__()
        self.linear = nn.Linear(embed_size, dim_out)
    def forward(self, x):
        x = self.linear(x)
        return x


def test_loop(model, test_loader, loss_fn, device, scalar=None):    
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
    model.eval() 
    with torch.no_grad():
        epoch_test_loss = 0
        for data, label in tqdm(test_loader):
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
            f.write(tabulate(data, headers=['ModelName',"Transformation during testing","Experiment name (model)", "MAE (L1Loss)", "95-th percentile","RMSE (MSELoss)"]))


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
    parser.add_argument('--weights_pth', type=str, default=CHECKPOINT_PATH+config['pth_name_linear_tuner'], help="Path to load pre-trained weights. Write here the name of the model (checkpoint).")
    parser.add_argument('--train_val_batchsize', type=int, default=config['train_and_val_batchsize'], help='Batch size for train and val. For test, we use 1.')
    parser.add_argument('--criterion', type=str, default=config['criterion'], help='Default (and the only one supported) MSE.')
    parser.add_argument('--device', type=str, default=config['device'], help='cuda or cpu.')
    parser.add_argument('--epochs_linear', type=int, default=config['epochs_linear'], help='Commonly for linear case we do 500 epochs.')
    parser.add_argument('--best_vloss', type=int, default=config['best_vloss'], help='Validation loss when to create a checkpoint of the model.')
    parser.add_argument('--data_regimes', type=list, default=config['data_regimes'], help='A list of strings with data regimes (Default: ["1k", "5k", "10k"]).')
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

    data_regimes = args.data_regimes

    for ff in range(len(data_regimes)):
        print(f'\n\Training is for the data regime {data_regimes[ff]}')

        ################## 0. Reset Pre-trained model ##################
        print(f'\n\n ***** 0. Reset Pre-trained model *****\n')
        model = wits.__dict__[args.model_name](h_slice=(args.number_antennas, 1), num_classes=2)
        print(f"Model {args.model_name} {args.number_antennas}x{1} built.")
        ### 0. Reset Weights to xavier ####
        model.apply(weights_init)  #Reset weights to xavier before calling the pretrained weights (to avoid weight accum. for random case)
        
        #model.get_intermediate_layers
        print(f"\nWeights that are supposed to find at : \n {args.weights_pth}\n")

        ### 0. Now load the pre-trained weights. ####
        logger.load_pretrained_weights(model, args.weights_pth, args.encoder)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        ##################################
        '''
        Be careful with the data regimes. Always check how many samples exactly you have. 
        In the below, I am assuming a dataset with 28k samples in case of KUL and whole dataset size in case of S and H.
        '''
        if data_regimes[ff].startswith('1k'):
            out_of_train_samples = 22001 # 3334, (kul:22001) (s:63212)this is for validation data, so that x_train is 1k
            train_and_val_batchsize = 128 # 
        elif data_regimes[ff].startswith('5k'):
            out_of_train_samples = 18001 #1334 # (s:59212) (kul:18001) (h:71200) this is for validation data, so that x_train is 5k
            train_and_val_batchsize = 512
        elif data_regimes[ff].startswith('10k'):
            out_of_train_samples = 13001 #334 # (s:54212) (kul:13001) this is for validation data, so that x_train is 10k
            train_and_val_batchsize = 512    

        ################## 1. Get Dataloaders ##################
        print(f'\n\n ***** 1. Get Dataloaders *****\n')

        # Now, do the splitting for train and val.
        x_train, X_val, y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train[:,2:3], test_size=out_of_train_samples)

        # To avoid using a large val sample size, get rid of most of the samples.
        x_valaa, x_val, y_valaa, y_val = train_test_split(X_val, Y_val, stratify=Y_val[:,2:3], test_size=410)

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
        #train_loader = DataLoader(train_dataset, batch_size=train_and_val_batchsize, shuffle=True,drop_last=True) # create your dataloader

        test_dataset = TensorDataset(tensor_test_x,tensor_test_y)# create your datset
        #test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False) # create your dataloader

        val_dataset = TensorDataset(tensor_val_x,tensor_val_y) # create your datset
        #val_loader = DataLoader(val_dataset, batch_size=train_and_val_batchsize, shuffle=True,drop_last=True) # create your dataloader


        print(f"Data loaded: there are {len(train_dataset)} and {len(test_dataset)} CSI Samples.")

        ################## 2. Embeddings ##################
        print(f'\n\n ***** 2. Embeddings *****\n')
       
        # Get Embeddings from pre-trained model.
        datasets = [train_dataset, val_dataset, test_dataset]

        for i in range(3):
            data_loader = DataLoader(
                datasets[i],
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                drop_last=True,
            )    
            print(f"Data loaded: there are {len(datasets[i])} CSI Samples.")

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
                #print("features",len(features))
                train_features = np.concatenate(embeddings, axis=0 )
                train_labels = np.concatenate(spot_lst, axis=0 )
            elif i==1:
                val_features = np.concatenate(embeddings, axis=0 )
                val_labels = np.concatenate(spot_lst, axis=0 )
            elif i==2:
                test_features = np.concatenate(embeddings, axis=0)
                test_labels = np.concatenate(spot_lst, axis=0)

        print(train_features.shape, train_labels.shape, val_features.shape, val_labels.shape, test_features.shape, test_labels.shape )

        ################## 3. New Dataloaders ##################
        print(f'\n\n ***** 3. New Dataloaders *****\n')
        
        tensor_x2 = torch.Tensor(train_features).float() # transform to torch tensor
        tensor_y2 = torch.Tensor(train_labels).float()
        train_dataset2 = TensorDataset(tensor_x2,tensor_y2) # create your datset
        train_loader2 = DataLoader(train_dataset2, batch_size=train_and_val_batchsize, shuffle=True, drop_last=True) # create your dataloader

        tensor_val_x2 = torch.Tensor(val_features).float() # transform to torch tensor
        tensor_val_2 = torch.Tensor(val_labels).float()
        val_dataset_2 = TensorDataset(tensor_val_x2,tensor_val_2) # create your datset
        val_loader2 = DataLoader(val_dataset_2, batch_size=train_and_val_batchsize, shuffle=True, drop_last=True) # create your dataloader

        tensor_test_x2 = torch.Tensor(test_features).float() # transform to torch tensor
        tensor_test_y2 = torch.Tensor(test_labels).float()
        test_dataset2 = TensorDataset(tensor_test_x2,tensor_test_y2) # create your datset
        test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False, drop_last=False) # create your dataloader


        ################## 4. Linear MLP Train ##################
        print(f'\n\n ***** 4. Linear MLP Train *****\n')
        net = LinearHead(train_features.shape[1],dim_out=2).to('cuda')
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate_eval, weight_decay = 10e-6)
        #optimizer = optim.SGD(net.parameters(), lr=0.0003)
        print(net)

        # Train Linear Regressor
        for epoch in range(epochs):
            epoch_number += 1
            epoch_loss = 0
            epoch_accuracy = 0
            accuracy = 0.0
            total = 0.0
            epoch_accuracy=0.0

            for data, label in train_loader2:
                data = data.float().to(device)
                label = label[:,0:2].float().to(device)
                optimizer.zero_grad()

                output = net(data)
                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss / len(train_loader2)

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in test_loader2:
                    data = data.float().to(device)
                    label = label[:,0:2].float().to(device)
                    #label = label[:,2:3].to(dtype=torch.long).to(device)
                    #label = torch.squeeze(label)
                    val_output = net(data)
                    val_loss = criterion(val_output, label)
                    epoch_val_loss += val_loss / len(test_loader2)
            #Print loss for every 50 Epochs.
            if epoch % 50 == 0:
                print(
                    f"Epoch : {epoch+1} - loss : {epoch_loss:.8f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.8f} - val_acc: {epoch_val_accuracy:.4f}\n"
                )

        # @markdown After the test is complete, consider to save results in /results/"experiment_name"/
        save_results = args.save_results #@param {type:"boolean"}

        print("\n********Now testing on test dataset loader*******\n\n")

        listRMSE, listMAE, actual_coordinates, estimated_coordinates = test_loop(net,test_loader2,loss_fn=criterion,device = device, scalar = scalar)

        print("\n\nDONE: returning parameters are: listRMSE, listMAE, actual_coordinates, estimated_coordinates \n\n")

        print("\n\nPlotting (and saving) retuls and \n\n")
        
        plot_results_actual_vs_estimate_ssl(actual_coordinates, estimated_coordinates, results_path, args.experiment_name, save=save_results, plot_name = 'LinearEval_'+data_regimes[ff]+"_Actual_vs_Estimated")

        plot_results_ecdf(listMAE, results_path, args.experiment_name, save=save_results, error_type = 'LinearEval_'+data_regimes[ff]+"_MAE")

        print_table_results_ssl(str(args.weights_pth)[-19:], transformation_type, args.experiment_name, listMAE, listRMSE, results_path, save=save_results, name_summarized = 'LinearEval_'+data_regimes[ff]+'_summarized_res')

        if save_results == True:
            print("\n### Save Errors for ECDF at a later time.####")
            hf = h5.File(results_path+'/LinearEval_'+data_regimes[ff]+"_MAE_errors_logs.h5", 'w')
            hf.create_dataset('listMAE', data=listMAE)
            hf.close()

            print("\n### Save Errors for ECDF at a later time.####")
            hf = h5.File(results_path+'/LinearEval_'+data_regimes[ff]+"_RMSE_errors_logs.h5", 'w')
            hf.create_dataset('listRMSE', data=listRMSE)
            hf.close()