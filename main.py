#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

# import Unet and nnUnet
from UNet.unet_model import UNet
from nnUnet.nnUnet import nnUNet

# Import denoising filters
from preprocessing import apply_gaussian_filter, apply_median_filter, apply_non_local_means_denoising, apply_bilateral_filtering, apply_wavelet_transform_denoising

datasets_params: dict[str, dict[str, Any]] = {}

def initialize_datasets_params(model_name: str) -> None:
    """
    Initialize datasets_params based on the provided model name.
    """
    if model_name == 'UNet':
        datasets_params["SEGTHOR"] = {'K': 5, 'net': UNet, 'B': 8}
    elif model_name == 'nnUNet':
        datasets_params["SEGTHOR"] = {'K': 5, 'net': nnUNet, 'B': 8}
    else:  # Assuming 'enet' or other models
        datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}
    
    # Add more datasets if needed
    datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}

class ReScale:
    def __init__(self, K):
        self.scale = 1 / (255 / (K - 1)) if K != 5 else 1 / 63

    def __call__(self, img):
        return img * self.scale

class Class2OneHot:
    """
    Converts class indices in the ground truth masks to one-hot encoded tensors.
    """
    def __init__(self, K: int):
        self.K = K

    def __call__(self, seg: torch.Tensor) -> torch.Tensor:
        # seg shape: [1, H, W]
        b, *img_shape = seg.shape  # b should be 1
        device = seg.device

        # Remove the batch dimension
        seg = seg.squeeze(0)  # shape: [H, W]

        # One-hot encode
        res = torch.zeros((self.K, *img_shape), dtype=torch.int32, device=device)
        res.scatter_(0, seg.unsqueeze(0), 1)  # scatter along channel dimension

        return res


def get_filter_function(filter_name: str) -> Any:
    """
    Returns the appropriate filter function based on the filter_name.
    """
    if filter_name == 'gaussian':
        return lambda nd: apply_gaussian_filter(nd, sigma=1)
    elif filter_name == 'median':
        return lambda nd: apply_median_filter(nd, size=3)
    elif filter_name == 'non_local_means':
        return lambda nd: apply_non_local_means_denoising(nd, h=10)
    elif filter_name == 'bilateral':
        return lambda nd: apply_bilateral_filtering(nd, d=9, sigmaColor=75, sigmaSpace=75)
    elif filter_name == 'wavelet':
        return lambda nd: apply_wavelet_transform_denoising(nd, wavelet='db1')
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    initialize_datasets_params(args.model)
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = datasets_params[args.dataset]['net'](1, K)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    filter_func = get_filter_function(args.filter)

    img_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),  # Convert to grayscale
        transforms.Lambda(lambda img: np.array(img)[np.newaxis, ...]),  # Add channel dimension
        transforms.Lambda(filter_func),  # Apply selected filter
        transforms.Lambda(lambda nd: nd / 255.0),  # Scale to [0,1]
        transforms.Lambda(lambda nd: torch.tensor(nd, dtype=torch.float32))  # Convert to tensor
    ])

    gt_transform = transforms.Compose([
        transforms.Lambda(lambda img: np.array(img)[...]),  # Convert to NumPy array
        ReScale(K),  # Scale based on number of classes
        transforms.Lambda(lambda nd: torch.tensor(nd, dtype=torch.int64).unsqueeze(0)),  # Add channel dimension
        Class2OneHot(K)  # One-hot encode
    ])

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug)
    
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:            
            if m == 'train':
                net.train()
                opt = optimizer
                cm = Dcm
                desc = f">> Training   ({e: 4d})"
                loader = train_loader
                log_loss = log_loss_tra
                log_dice = log_dice_tra
            elif m == 'val':
                net.eval()
                opt = None
                cm = torch.no_grad
                desc = f">> Validation ({e: 4d})"
                loader = val_loader
                log_loss = log_loss_val
                log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', choices=['SEGTHOR', 'TOY2'], required=True, help='Dataset name')
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    parser.add_argument('--model', default='enet', help="Which model to use? [ENet, UNet, nnUNet]" )
    parser.add_argument('--filter', choices=['gaussian', 'median', 'non_local_means', 'bilateral', 'wavelet'],
                        required=True, help="Filter to apply for preprocessing.")
    
    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()