import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import json
import argparse
import sys
import os
from pathlib import Path
import datetime
import time
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

import models.model_transformer as wits
importlib.reload(wits)
from models import logger
importlib.reload(logger)
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

    X_train, x_test, Y_train, y_test = train_test_split(csi2[:,:,0:100,:], tx_transform, stratify=tx_transform[:,2:3], test_size=100) #locations_ID2 was replaced by tx...

    return X_train, x_test, Y_train, y_test, scalar


class ModelWrapper(nn.Module):
    def __init__(self, backbone, pool_head, glob_head, loc_head):
        super(ModelWrapper, self).__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.pool_head = pool_head
        self.glob_head = glob_head
        self.loc_head = loc_head

    def forward(self, x, head_only=False, loca=True):
        output = self.backbone(x)
        logits = self.pool_head(output, loca)
        return [self.glob_head(logits[:,:1]), self.loc_head(logits[:,1:])]


def train_model(args):
    logger.init_distributed_mode(args)
    logger.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print(f'\n****Final configuration****\n',"\n", get_args_values(args))

    dataset = train_dataset

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=True)    
    
    print(f"\nData loaded: there are {len(dataset)} CSI Samples.")

    if args.model_name in wits.__dict__.keys():
        online_encoder = wits.__dict__[args.model_name](
            h_slice = args.h_slice,
            drop_path_rate=0.1,  # stochastic depth
        )
        targ_encoder = wits.__dict__[args.model_name](h_slice=args.h_slice)
        embed_dim = online_encoder.embed_dim
        num_heads = online_encoder.num_heads
    else:
        print(f"Unknow model: {args.model_name}")

    print(targ_encoder)
    print('###########')
    print(online_encoder)   


    # multi-crop wrapper handles forward with inputs of different sizes as shown in DINO.
    online_encoder = ModelWrapper(
        online_encoder, 
        wits.TransformerPooling(embed_dim, num_heads, args.k_nns),  
        wits.Projectors(embed_dim,args.g_pro_out),  
        wits.Projectors(embed_dim,args.l_pro_out),  
    )
    targ_encoder = ModelWrapper(
        targ_encoder,
        wits.TransformerPooling(embed_dim, num_heads, args.k_nns),
        wits.Projectors(embed_dim,args.g_pro_out),
        wits.Projectors(embed_dim, args.l_pro_out),
    )

    print('\n\n\n### Print targ_encoder Complete')
    print(targ_encoder)
    print('\n\n\n### Print online_encoder Complete')
    print(online_encoder)  

    lr_schedule = logger.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * logger.get_world_size()) / 256., 
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = logger.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    # Commented parts are useful for multi-gpu training.
    online_encoder, targ_encoder = online_encoder.cuda(), targ_encoder.cuda()
    # synchronize batch norms (if any)
    if logger.has_batchnorms(online_encoder):
        online_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(online_encoder)
        targ_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(targ_encoder)

        
        # we need DDP wrapper to have synchro batch norms working...
        #targ_encoder = nn.parallel.DistributedDataParallel(targ_encoder, device_ids=[args.gpu])
        #targ_encoder_without_ddp = targ_encoder.module
    else:
        # targ_encoder_without_ddp and targ_encoder are the same thing
        targ_encoder_without_ddp = targ_encoder      
    #online_encoder = nn.parallel.DistributedDataParallel(online_encoder, device_ids=[args.gpu])
    # targ_encoder and online_encoder start with the same weights
    targ_encoder_without_ddp.load_state_dict(online_encoder.state_dict())
    #targ_encoder_without_ddp.load_state_dict(online_encoder.module.state_dict())
    # there is no backpropagation through the targ_encoder, so no need for gradients
    for p in targ_encoder.parameters():
        p.requires_grad = False
    print(f"online_encoder and targ_encoder are built: they are both {args.model_name} network.")
    #return online_encoder, targ_encoder

    loss_ssl = Loss(
        args.g_pro_out, args.l_pro_out,
        args.Ns + 2,
        args.warmup_temp,
        args.target_temp,
        args.warmup_temp_epochs,
        args.epochs,
    ).cuda()

    params_groups = logger.get_params_groups(online_encoder)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with wits
    else:
        "Optimizer not supported."
    # for mixed precision training
    fp16_scaler = None

    momentum_schedule = logger.cosine_scheduler(args.momentum_targ_encoder, 1, args.epochs, len(data_loader))

    to_restore = {"epoch": 0}
    logger.restart_from_checkpoint(
        #os.path.join(args.output_path, "checkpoint.pth.tar"),
        os.path.join(args.output_path, "checkpoint.pth"),
        run_variables=to_restore,
        online_encoder=online_encoder,
        targ_encoder=targ_encoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        loss_ssl=loss_ssl,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting self-supervised model training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(online_encoder, targ_encoder, targ_encoder_without_ddp, loss_ssl, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

        save_dict = {
            'online_encoder': online_encoder.state_dict(),
            'targ_encoder': targ_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss_ssl': loss_ssl.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        logger.save_on_master(save_dict, os.path.join(args.output_path, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            logger.save_on_master(save_dict, os.path.join(args.output_path, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if logger.is_main_process():
            with (Path(args.output_path) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

class Loss(nn.Module):
    def __init__(self, g_pro_out, l_pro_out, vviews, warmup_targ_encoder_temp, targ_encoder_temp,
                 warmup_targ_encoder_temp_epochs, nepochs, online_encoder_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.online_encoder_temp = online_encoder_temp
        self.center_momentum = center_momentum
        self.vviews = vviews
        self.register_buffer("center", torch.zeros(1, 1, g_pro_out))
        self.register_buffer("patch_center", torch.zeros(1, l_pro_out))

        self.targ_encoder_temp_schedule = np.concatenate((
            np.linspace(warmup_targ_encoder_temp,
                        targ_encoder_temp, warmup_targ_encoder_temp_epochs),
            np.ones(nepochs - warmup_targ_encoder_temp_epochs) * targ_encoder_temp
        ))
    def forward(self, targ_encoder, online_encoder, online_encoder_output, targ_encoder_output, epoch, it):
        online_encoder_cls = online_encoder_output[0][0].chunk(2) + online_encoder_output[1][0].chunk(self.vviews-2)
        online_encoder_loc = online_encoder_output[0][1].chunk(2) + online_encoder_output[1][1].chunk(self.vviews-2)

        targ_encoder_cls = targ_encoder_output[0][0].chunk(2) + targ_encoder_output[1][0].chunk(self.vviews-2)
        targ_encoder_loc = targ_encoder_output[0][1].chunk(2) + targ_encoder_output[1][1].chunk(self.vviews-2)
        temp = self.targ_encoder_temp_schedule[epoch]

        c_loss = 0
        s_loss = 0
        n_loss_terms = 0
        m_loss_terms = 0
        assert len(targ_encoder_cls) == self.vviews
        for iq in range(len(targ_encoder_cls)):
          q_cls = F.softmax((targ_encoder_cls[iq] - self.center)/ temp, dim=-1).detach()
          for v in range(self.vviews):
              if v == iq:
                  q_pat = F.softmax((targ_encoder_loc[iq] - self.patch_center)/ temp, dim=-1).detach()
                  p_pat = online_encoder_loc[v]
                  patch_loss = torch.sum(-q_pat * F.log_softmax(p_pat / self.online_encoder_temp, dim=-1), dim=-1)
                  s_loss += patch_loss.mean()
                  m_loss_terms += 1
              else:
                  if iq > 1:
                      continue
                  cls_loss = torch.sum(-q_cls * F.log_softmax(online_encoder_cls[v] / self.online_encoder_temp, dim=-1), dim=-1)
                  c_loss += cls_loss.mean()
                  n_loss_terms += 1
        c_loss /= n_loss_terms
        s_loss /= m_loss_terms
        
        self.update_center(torch.cat(targ_encoder_cls), it)
        self.update_patch_center(targ_encoder_loc, it)
        return (c_loss + s_loss*0.1), c_loss.item(), s_loss.item()

    @torch.no_grad()
    def update_center(self, targ_encoder_output, it):
        """
        Update center used for targ_encoder output.
        """
        batch_center = torch.sum(targ_encoder_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(targ_encoder_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_patch_center(self, targ_encoder_output, it):
        """
        Update center used for targ_encoder output.
        """
        patch_num = 0
        batch_center = 0
        for t_out in targ_encoder_output:
            patch_num += t_out.shape[1] * t_out.shape[0]
            batch_center += t_out.sum(1).sum(0, keepdim=True)

        dist.all_reduce(batch_center)
        batch_center = batch_center / (patch_num * dist.get_world_size())

        # ema update
        self.patch_center = self.patch_center * self.center_momentum + batch_center * (1 - self.center_momentum)

def get_args_values(args):
    args_dict = vars(args)
    args_strings = []
    for key, value in args_dict.items():
        args_strings.append("{}: {}".format(key, str(value)))
    return "\n".join(args_strings)



def train_one_epoch(online_encoder, targ_encoder, targ_encoder_without_ddp, loss_ssl, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = logger.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (H_samples, _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        v_views = []
        local_views = args.Ns
        v_views.append(transformation_global_1(H_samples))
        v_views.append(transformation_global_2(H_samples))
        for _ in range(local_views):
            v_views.append(transformation_ns(H_samples))
        H_samples = [h_view.cuda(non_blocking=True) for h_view in v_views]
        # targ_encoder and online_encoder forward microscopic + compute self-supervised model loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.no_grad():
                targ_encoder_output = [targ_encoder(torch.cat(H_samples[:2]), head_only=True, loca=True), targ_encoder(torch.cat(H_samples[2:]), head_only=True, loca=True)] 
            online_encoder_output = [online_encoder(torch.cat(H_samples[:2]), head_only=True, loca=False), online_encoder(torch.cat(H_samples[2:]), head_only=True, loca=False)]
            loss, c_loss, s_loss = loss_ssl(targ_encoder, online_encoder, online_encoder_output, targ_encoder_output, epoch, it)
            loss_value = loss.item()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # updates
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = logger.clip_gradients(online_encoder, args.clip_grad)
            logger.cancel_gradients_last_layer(epoch, online_encoder,args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = logger.clip_gradients(online_encoder, args.clip_grad)
            logger.cancel_gradients_last_layer(epoch, online_encoder,args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        # EMA
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(online_encoder.parameters(), targ_encoder_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        # logging
        torch.cuda.synchronize()
        metric_logger.update(ssl_loss=loss_value)
        metric_logger.update(macro_loss=c_loss)
        metric_logger.update(micro_loss=s_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # Stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=1)
    return output


if __name__ == '__main__':
    # Load the the config file
    with open("config_train.json", "r") as f:
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

    parser = argparse.ArgumentParser(f'Training on {config["dataset_to_download"]} dataset.')
    parser.add_argument('--experiment_name', type=str, default=config['experiment_name'], help='Name of this experiment.')
    parser.add_argument('--dataset_to_download', type=str, default=config['dataset_to_download'], help='Path to dataset to load.')
    parser.add_argument('--saved_dataset_path', type=str, default=config['saved_dataset_path'], help='Path to dataset to load.')
    parser.add_argument('--sub_dataset_file_csi',type=str, default=config['sub_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
    parser.add_argument('--sub_dataset_file_loc',type=str, default=config['sub_loc_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
    parser.add_argument('--realMax', type=float, default=config['realMax'], help='Max value of real part for the whole dataset')
    parser.add_argument('--imagMax', type=float, default=config['imagMax'], help='Max value of imag part for the whole dataset')
    parser.add_argument('--absMax', type=float, default=config['absMax'], help='Max value of abs part for the whole dataset')
    parser.add_argument('--Ns', type=float, default=config['Ns'], help='Number of local views.')
    parser.add_argument('--model_name', default=config['model_name'], type=str, choices=['wit', 'wit_l'], help="""Name of model encoders (backbones too).""")
    parser.add_argument('--h_slice', type=tuple, default=(64,1),  help=""" Only per-subcarrier slices supported.""")
    parser.add_argument('--g_pro_out', type=int, default=config['global_projector_out'],  help="""Number of nodes in last layer of global projector/expander.""")
    parser.add_argument('--l_pro_out', type=int, default=config['local_projector_out'],  help=""""Number of nodes in last layer of local projector/expander.""")
    parser.add_argument("--k_nns", type=int, default=config['k_nns'], help="top-k similar subcarriers to select.")
    parser.add_argument('--momentum_targ_encoder', default=0.996, type=float, help="""Base EMA.""")
    parser.add_argument('--warmup_temp', type=float, default=config['warmup_temp'], help="""Value for the target encoder temperature, weep 0.04.""")
    parser.add_argument('--target_temp', type=float, default=config['target_temp'], help="""Final value (after linear warmup). Never above 0.07 is recommended.""")
    parser.add_argument('--warmup_temp_epochs', type=int, default=config['warmup_temp_epochs'], help='Epochs for (linear) warmup.')
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Start value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of for the weight decay. Use cosine scheduler.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Gradient norm if using gradient cropping.""")
    parser.add_argument('--batch_size_per_gpu', type=int, default=config['batch_size_per_gpu'],  help='batch-size. Single-gpu.')
    parser.add_argument('--epochs', type=int, default=config['epochs'],  help='Epochs of training.')
    parser.add_argument('--freeze_last_layer', type=int, default=config['freeze_layer_num_epochs'],  help="""Epochs  to keep fixed the output layer.""")
    parser.add_argument("--lr", type=float, default=config['learning_rate'],  help="""Highest value of learning rate (after the linear warmup). Scaling depends on the batch-size.""")
    parser.add_argument("--warmup_epochs", type=int, default=config['warmup_epochs_learning_rate'],  help="Epochs learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=config['minimum_lr'], help="""Lowest value of learning-rate (end of training). Based on cosie scheduler.""")
    parser.add_argument('--optimizer',  type=str, default=config['optimizer'], help="""Adam with weights-decoupling.""")
    parser.add_argument('--output_path', default=CHECKPOINT_PATH + "/", type=str, help='Logs and model checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Frequency of saving the models.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU. Not useful when single-gpu.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    args = parser.parse_args(args=[])

    print(f'***Configuration****\n',"\n",get_args_values(args))

    print('\n\n',args.saved_dataset_path)

    x_train, _, y_train, _, scalar = prep_data_load(args)

    transformation_global_1 = channel_transforms('first',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)
    transformation_global_2 = channel_transforms('second',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)
    transformation_ns = channel_transforms('others',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)

    print(f'\n\n ***** 1. Get Dataloaders only for training set.*****\n')

    print(f"Shapes: {x_train.shape}")
    print(f"Train Data: {len(x_train)}")

    X_train2 = np.einsum('basc->bcas', x_train)
    tensor_x = torch.tensor(X_train2).float()
    tensor_y = torch.tensor(y_train).float()

    train_dataset = TensorDataset(tensor_x,tensor_y)

    train_model(args)