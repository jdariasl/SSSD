import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from utils.util import (
    find_max_epoch,
    print_size,
    training_loss,
    calc_diffusion_hyperparams,
)
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util_sp import SP_Dataset
from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer


def train(
    output_directory,
    ckpt_iter,
    n_iters,
    iters_per_ckpt,
    iters_per_logging,
    learning_rate,
    batch_size,
    use_model,
    only_generate_missing,
    masking,
    missing_k,
):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """


    torch.cuda.set_device(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"], diffusion_config["beta_0"], diffusion_config["beta_T"]
    )

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).to(device)
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).to(device)
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).to(device)
    else:
        print("Model chosen not available.")
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, "{}.pkl".format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location="cpu")

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("Successfully loaded model at iteration {}".format(ckpt_iter))
        except:
            ckpt_iter = -1
            print(
                "No valid checkpoint model found, start training from initialization try."
            )
    else:
        ckpt_iter = -1
        print("No valid checkpoint model found, start training from initialization.")

    ### Custom data loading and reshaping ###

    # load data
    train_loader = SP_Dataset(
        dataset_path=trainset_config["train_data_path"], sp_path_list="blink_paths.txt"
    )
    # ------------------------------------------------------------------------------------
    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
    }  #'sampler' : sampler
    # ------------------------------------------------------------------------------------------
    training_data = DataLoader(train_loader, **train_params)
    print("Data loaded")

    # training
    loss_history = []
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch_dl in training_data:
            batch, mask_mvi = batch_dl

            if masking == "rm":
                mask_imp = get_mask_rm(batch[0], missing_k)
            elif masking == "mnr":
                mask_imp = get_mask_mnr(batch[0], missing_k)
            elif masking == "bm":
                mask_imp = get_mask_bm(batch[0], missing_k)

            mask_imp[:,-1] = 1
            mask_imp = mask_imp.repeat(batch.size()[0], 1, 1)
            mask = torch.logical_and(mask_imp, mask_mvi)
            mask = mask.permute(0, 2, 1).float().to(device)

            loss_mask = torch.logical_and(~mask_imp.bool(), mask_mvi.bool())
            loss_mask = loss_mask.permute(0, 2, 1).bool().to(device)

            batch = batch.permute(0, 2, 1)
            batch = batch.float().to(device)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(
                net,
                nn.MSELoss(),
                X,
                diffusion_hyperparams,
                only_generate_missing=only_generate_missing,
                device=device,
            )

            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = "{}.pkl".format(n_iter)
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(output_directory, checkpoint_name),
                )
                print("model at iteration %s is saved" % n_iter)

            n_iter += 1
        save_loss(loss_history, output_directory)

def save_loss(loss_history, output_directory):
    loss_path = os.path.join(output_directory, "loss.txt")
    with open(loss_path, "w") as f:
        for loss in loss_history:
            f.write(str(loss) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="src/config/config_SSSDS4.json",
        help="JSON file for configuration",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config
    )  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config["use_model"] == 0:
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    elif train_config["use_model"] == 2:
        model_config = config["wavenet_config"]

    train(**train_config)
