import pandas as pd
import numpy as np
from scipy import stats
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

def find_sp_type(path):
    init = path.find('Pur_')
    sp = int(path[init+4:])
    if sp < 5:
        return 0
    elif sp >= 5 and sp < 9:
        return 1
    elif sp >= 9:
        return 2

def read_dat(path):
    
    # ----------------- find sp_type ---------------------------
    
    sp_type = find_sp_type(path)
    path_file = os.path.join(path, "target.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    target = df.values
    if sp_type == 0 or sp_type == 2:
        target = target[:, 0]
    elif sp_type == 1:
        target = target[:, 1]
    target = target.reshape(-1,1)
    # ---------- load time series -------------------------------
    path_file = os.path.join(path, "LTS.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    left = df.values
    mask_l = np.isnan(left)
    left = np.nan_to_num(left)
    mask_l1 = ~mask_l

    path_file = os.path.join(path, "RTS.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    rigth = df.values

    mask_r = np.isnan(rigth)
    rigth = np.nan_to_num(rigth)
    mask_r1 = ~mask_r
    
    data = np.c_[left, rigth]
    mask1 = np.c_[mask_l1, mask_r1]
    # --------- load events and define mask ---------------------
    path_file = os.path.join(path, "LEventCorrected.dat")
    if not os.path.isfile(path_file):
        path_file = os.path.join(path, "LEvent.dat")
    df = pd.read_csv(path_file, delimiter=",", header=None)
    events = df.values
    events_l = events[events[:, 2] == 2, :]
    mask_l2 = np.ones_like(left)
    for i in range(events_l.shape[0]):
        mask_l2[events_l[i, 0] : events_l[i, 1], :] = 0

    path_file = os.path.join(path, "REventCorrected.dat")
    if not os.path.isfile(path_file):
        path_file = os.path.join(path, "REvent.dat")
    df = pd.read_csv(path_file, delimiter=",", header=None)
    events = df.values
    events_r = events[events[:, 2] == 2, :]
    mask_r2 = np.ones_like(rigth)
    for i in range(events_r.shape[0]):
        mask_r2[events_r[i, 0] : events_r[i, 1], :] = 0

    mask2 = np.c_[mask_l2, mask_r2]
    mask = np.logical_and(mask1, mask2.astype(bool))

    # ---------normalization ------------------------------------
    batch = stats.zscore(data, axis=0, nan_policy="omit")

    # --------- pad or crop to 15000 samples --------------------
    if batch.shape[0] < 15000:
        batch = np.pad(
            batch, ((0, 15000 - batch.shape[0]), (0, 0)), "constant", constant_values=0
        )
        mask = np.pad(
            mask, ((0, 15000 - mask.shape[0]), (0, 0)), "constant", constant_values=0
        )
        mask2 = np.pad(
            mask2, ((0, 15000 - mask2.shape[0]), (0, 0)), "constant", constant_values=0
        )
    elif batch.shape[0] > 15000:
        batch = batch[:15000, :]
        mask = mask[:15000, :]
        mask2 = mask2[:15000, :]
        
    if target.shape[0] < 15000:
        target = np.pad(
            target, ((0, 15000 - target.shape[0]), (0, 0)), "constant", constant_values=0
        )
    elif target.shape[0] > 15000:
        target = target[:15000]
    
    #---------------------------------------------------------------
    #SP signals can be shorter than 15000 samples, so it is better to first
    #pad the signals and then concatenate them.
    target = stats.zscore(target, axis=0, nan_policy="omit")
    #mask = np.c_[mask, np.ones((15000,1))]
    #batch = np.c_[batch, target]
    #---------------------------------------------------------------
    #Series of 15000 samples are too long for the GPU memory
    #We will split the series into 3 series of 5000 samples
    #and we will train the model on each of them
    i = np.random.randint(0, 3)
    batch = batch[i*5000:(i+1)*5000,:]
    mask = mask[i*5000:(i+1)*5000,:]
    return batch, mask.astype(int) #batch[i:i+5000,:], mask[i:i+5000,:].astype(int)

def read_dat_blink_labels(path,start,length):
    # ----------------- find sp_type ---------------------------
    sp_type = find_sp_type(path)
    path_file = os.path.join(path, "target.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    target = df.values
    if sp_type == 0 or sp_type == 2:
        target = target[:, 0]
    elif sp_type == 1:
        target = target[:, 1]
    target = target.reshape(-1,1)
    # ---------- load time series -------------------------------
    path_file = os.path.join(path, "LTS.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    left = df.values
    mask_l = np.isnan(left)
    left = np.nan_to_num(left)
    mask_l1 = ~mask_l

    path_file = os.path.join(path, "RTS.dat")
    df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
    rigth = df.values

    mask_r = np.isnan(rigth)
    rigth = np.nan_to_num(rigth)
    mask_r1 = ~mask_r
    
   
    mask1 = np.c_[mask_l1, mask_r1]
    
    #--------- Create simulated blink labels -------------------
    mask_l2 = np.ones_like(left)
    mask_l2[start[0]:start[0]+length[0],:] = 0
    #left[start[0]:start[0]+length[0],:] = 0
    mask_r2 = np.ones_like(rigth)
    mask_r2[start[1]:start[1]+length[1],:] = 0
    #rigth[start[1]:start[1]+length[1],:] = 0
    mask2 = np.c_[mask_l2, mask_r2]
    data = np.c_[left, rigth]
    
    mask = np.logical_and(mask1, mask2.astype(bool))

    # ---------normalization ------------------------------------
    batch = stats.zscore(data, axis=0, nan_policy="omit")
    mean = np.nanmean(data,axis=0)
    std = np.nanstd(data,axis=0)
    # --------- pad or crop to 15000 samples --------------------
    if batch.shape[0] < 15000:
        batch = np.pad(
            batch, ((0, 15000 - batch.shape[0]), (0, 0)), "constant", constant_values=0
        )
        mask = np.pad(
            mask, ((0, 15000 - mask.shape[0]), (0, 0)), "constant", constant_values=0
        )
        mask2 = np.pad(
            mask2, ((0, 15000 - mask2.shape[0]), (0, 0)), "constant", constant_values=0
        )
    elif batch.shape[0] > 15000:
        batch = batch[:15000, :]
        mask = mask[:15000, :]
        mask2 = mask2[:15000, :]
        
    if target.shape[0] < 15000:
        target = np.pad(
            target, ((0, 15000 - target.shape[0]), (0, 0)), "constant", constant_values=0
        )
    elif target.shape[0] > 15000:
        target = target[:15000]
    
    
    #---------------------------------------------------------------
    #SP signals can be shorter than 15000 samples, so it is better to first
    #pad the signals and then concatenate them.
    target = stats.zscore(target, axis=0, nan_policy="omit")
    #mask = np.c_[mask, np.ones((15000,1))]
    #mask2 = np.c_[mask2, np.ones((15000,1))]
    #batch = np.c_[batch, target]
    #---------------------------------------------------------------
    #Series of 15000 samples are too long for the GPU memory
    #We will split the series into 3 series of 5000 samples
    #and we will train the model on each of them
    #i = np.random.randint(0, 3)
    #batch = batch[i*5000:(i+1)*5000,:]
    #mask = mask[i*5000:(i+1)*5000,:]
    return batch, mask.astype(int), mask2.astype(int), [mean,std] #batch[i:i+5000,:], mask[i:i+5000,:].astype(int)

def read_filepaths(file):
    with open(file, "r") as f:
        lines = f.read().splitlines()
    return lines


class SP_Dataset(Dataset):
    """
    Code for reading the SP_Dataset
    """

    def __init__(self, dataset_path="./datasets", sp_path_list="no_blink_paths.txt"):
        self.root = dataset_path

        path_file = os.path.join(self.root, sp_path_list)
        self.paths = read_filepaths(path_file)
        print("Number of samples =  {}".format(len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sp_tensor, mask = self.load_sp(self.root + self.paths[index])
        return sp_tensor, mask

    def load_sp(self, sp_path):
        time_series, mask = read_dat(sp_path)
        return torch.from_numpy(time_series), torch.from_numpy(mask)

class SP_Dataset_Test(Dataset):
    """
    Code for reading the SP_Dataset
    """

    def __init__(self, dataset_path="./datasets", sp_path_list="no_blink_paths.txt", path_blink_start="BlinkStarts.txt", path_blink_length="BlinkLengths.txt"):
        self.root = dataset_path
        path_file = os.path.join(self.root, sp_path_list)
        self.paths = read_filepaths(path_file)
        self.blink_start = np.loadtxt(os.path.join(self.root, path_blink_start)).astype(int)
        self.blink_length = np.loadtxt(os.path.join(self.root, path_blink_length)).astype(int)
        print("Number of samples =  {}".format(len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sp_tensor, mask, mask_loss, norm_params = self.load_sp(index)
        return sp_tensor, mask, mask_loss, norm_params

    def load_sp(self, index):
        sp_path = self.root + self.paths[index]
        time_series, mask, mask_loss, norm_params = read_dat_blink_labels(sp_path,start=[self.blink_start[2*index]-1,self.blink_start[2*index+1]-1],length=[self.blink_length[2*index],self.blink_length[2*index+1]])
        return torch.from_numpy(time_series), torch.from_numpy(mask), mask_loss, norm_params

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()