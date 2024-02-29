import os
import argparse
import json
import numpy as np
import torch

from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from utils.util_sp import SP_Dataset_Test
from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from statistics import mean


def generate(output_directory,
             num_samples,
             ckpt_path,
             batch_size,
             ckpt_iter,
             use_model,
             only_generate_missing):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    torch.cuda.set_device(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

            
    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).to(device)
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).to(device)
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).to(device)
    else:
        print('Model chosen not available.')
    print_size(net)

    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

        
        
    ### Custom data loading and reshaping ###
    
    testing_loader = SP_Dataset_Test(
        dataset_path=trainset_config["train_data_path"], sp_path_list="no_blink_paths_org.txt"
    )
    
    # ------------------------------------------------------------------------------------
    train_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 4,
        "drop_last": True,
    }  #'sampler' : sampler
    # ------------------------------------------------------------------------------------------
    testing_data = DataLoader(testing_loader, **train_params)
    
    print('Data loaded')

    all_mse = []

    
    for i, batch_dl in enumerate(testing_data):
        
        batch, mask, mask_loss, norm_params = batch_dl
        mask = mask.permute(0, 2, 1).float().to(device)
        mask_loss = mask_loss.permute(0, 2, 1).float()
        batch = batch.permute(0,2,1)
        batch = batch.float().to(device)          
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                diffusion_hyperparams,
                                cond=batch,
                                mask=mask,
                                device=device,
                                only_generate_missing=only_generate_missing)

        end.record()
        torch.cuda.synchronize()

        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                            ckpt_iter,
                                                                                            int(start.elapsed_time(
                                                                                                end) / 1000)))

        
        generated_audio = generated_audio.detach().cpu().numpy()
        temp = generated_audio*norm_params[1].numpy()[:,:,np.newaxis] + norm_params[0].numpy()[:,:,np.newaxis]
        #temp = generated_audio[:,0:4,:]*norm_params[1].numpy()[:,:,np.newaxis] + norm_params[0].numpy()[:,:,np.newaxis]
        #np.save('Batch_generated.npy', temp)
        batch = batch.detach().cpu().numpy()
        batch = batch*norm_params[1].numpy()[:,:,np.newaxis] + norm_params[0].numpy()[:,:,np.newaxis]
        #batch = batch[:,0:4,:]*norm_params[1].numpy()[:,:,np.newaxis] + norm_params[0].numpy()[:,:,np.newaxis]
        mask = mask.detach().cpu().numpy()
        mask_loss = mask_loss.detach().cpu().numpy() 
        #mask_loss = mask_loss[:,0:4,:]
        
        outfile = f'imputation{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, temp)

        #outfile = f'original{i}.npy'
        #new_out = os.path.join(ckpt_path, outfile)
        #np.save(new_out, batch)

        #outfile = f'mask{i}.npy'
        #new_out = os.path.join(ckpt_path, outfile)
        #np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)
        
        mse = mean_squared_error(temp[~mask_loss.astype(bool)], batch[~mask_loss.astype(bool)])
        all_mse.append(mse)
    
    print('Total MSE:', mean(all_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="src/config/config_SSSDS4.json",
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=500,
                        help='Number of utterances to be generated')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=train_config["batch_size"],
             use_model=train_config["use_model"],
             batch_size=train_config["batch_size"],
             only_generate_missing=train_config["only_generate_missing"])
