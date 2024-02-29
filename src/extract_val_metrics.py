import numpy as np
import pandas as pd
from utils.util_sp import read_filepaths
import os

root = '/media/my_ftp/Oculography/RevisionJorge_Junio2023/'
results_path = '/home/jarias@gaps_domain.ssr.upm.es/SP_SSSD/results/sp/T200_beta00.0001_betaT0.02/'

def find_sp_type(path):
    init = path.find('Pur_')
    sp = int(path[init+4:])
    if sp < 5:
        return 0
    elif sp >= 5 and sp < 9:
        return 1
    elif sp >= 9:
        return 2


def get_mse():
    
    blink_start = np.loadtxt(os.path.join(root, "BlinkStarts.txt")).astype(int)
    blink_length = np.loadtxt(os.path.join(root, "BlinkLengths.txt")).astype(int)
    path_file = os.path.join(root,"no_blink_paths.txt")
    paths = read_filepaths(path_file)
    inc = 32
    MSE = []
    for i in range(17):
        imputation = np.load(os.path.join(results_path,"imputation{}.npy".format(i)))
        for j in range(0,inc):
            
            sp_type = find_sp_type(paths[i*inc+j])
            
            
            path_file = os.path.join(root, paths[i*inc+j], "LTS.dat")
            df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
            left_sp = df.values
            
            path_file = os.path.join(root, paths[i*inc+j], "RTS.dat")
            df = pd.read_csv(filepath_or_buffer=path_file, delimiter=",")
            right_sp = df.values
            
            ini_blink_left = blink_start[i*2*inc+2*j]
            length_blink_left = blink_length[i*2*inc+2*j]
            end_l = np.minimum(ini_blink_left+length_blink_left,15000)
            left_segment = left_sp[ini_blink_left:end_l,:]
            
            ini_blink_right = blink_start[i*2*inc+2*j+1]
            length_blink_right = blink_length[i*2*inc+2*j+1]
            end_r = np.minimum(ini_blink_right+length_blink_right,15000)
            right_segment = right_sp[ini_blink_right:end_r,:]
            
            
            
            if sp_type == 0: # Anlayse x axis
                
                imputation_left_sp = imputation[j,0,ini_blink_left:end_l]
                MSE.append(np.mean((left_segment[:,0]-imputation_left_sp)**2))
                
                imputation_right_sp = imputation[j,2,ini_blink_right:end_r]
                MSE.append(np.mean((right_segment[:,0]-imputation_right_sp)**2))
            elif sp_type == 1: #º Analyse y axis
                
                imputation_left_sp = imputation[j,1,ini_blink_left:end_l]
                MSE.append(np.mean((left_segment[:,1]-imputation_left_sp)**2))
                
                imputation_right_sp = imputation[j,3,ini_blink_right:end_r]
                MSE.append(np.mean((right_segment[:,1]-imputation_right_sp)**2))
            elif sp_type == 2: # Analyse x and y axis
                
                imputation_left_sp = imputation[j,0,ini_blink_left:end_l]
                MSE.append(np.mean((left_segment[:,0]-imputation_left_sp)**2))
                
                imputation_right_sp = imputation[j,2,ini_blink_right:end_r]
                MSE.append(np.mean((right_segment[:,0]-imputation_right_sp)**2))
                
                
                imputation_left_sp = imputation[j,1,ini_blink_left:end_l]
                MSE.append(np.mean((left_segment[:,1]-imputation_left_sp)**2))
               
                imputation_right_sp = imputation[j,3,ini_blink_right:end_r]
                MSE.append(np.mean((right_segment[:,1]-imputation_right_sp)**2))
                
    
    MSE = np.array(MSE)
    MSE = np.sqrt(MSE)
    return np.mean(MSE), np.std(MSE)
                
                    
if __name__ == "__main__":
    mse, std = get_mse()
    print('MSE:', mse)
    print('STD:', std)
    
    