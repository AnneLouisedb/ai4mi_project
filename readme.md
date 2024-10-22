# AI for medical imaging — Semantic Segmentation for Organs at Risk
 **University of Amsterdam**

 By Vardan Narula, AnneLouise de Boer, Gina Luijkx, Julotte van der Beek, and Dur e Najaf Amjad 

## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.

## Model Architectures

### E-Net
![U-Net+DR architecture,](images/ENet-architecture.png)
C. He, L. Chen, L. Xu, C. Yang, X. Liu, and B. Yang, "IRLSOT: Inverse Reinforcement Learning for Scene-Oriented Trajectory Prediction," *IET Intelligent Transport Systems*, vol. 16, 2022. [doi:10.1049/itr2.12172](https://doi.org/10.1049/itr2.12172).

### U-Net
![U-Net architecture](images/u-net-architecture.png)
 O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, ser. LNCS, vol. 9351. Springer, 2015, pp. 234–241. Available: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) 

### 2D V-Net
![U-Net architecture](images/VNet_architecture.png)
 D. Rastogi, P. Johri, and V. Tiwari, "Brain Tumor Segmentation and Tumor Prediction Using 2D-VNet Deep Learning Architecture," in *2021 10th International Conference on System Modeling & Advancement in Research Trends (SMART)*, pp. 723-732, 2021. [doi:10.1109/SMART52563.2021.9676317](https://doi.org/10.1109/SMART52563.2021.9676317).

### sU-Net
![sU-Net architecture](images/sU-Net.png)

### U-Net+DR
![U-Net+DR architecture,](images/UNet+DR_architecture.png)


## Pre-processing
The preprocessing steps include resampling and intensity normalization, ensuring that the input data is consistently formatted across datasets. Data augmentation techniques such as random cropping are done on the fly during training.

### Heart label transformation
TO DO

### Gaussian Smoothing

### Median Filtering

## Training

### Regular Training
```
$ TODO
$ TOD

```

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--epochs` | 200 | - | Number of training epochs |
| `--dataset` | - | SEGTHOR, TOY2 | Dataset name (required) |
| `--mode` | full | partial, full, weighted | Training mode |
| `--optimizer` | adam | adam, adamW | Optimizer choice |
| `--lr` | 0.0005 | - | Learning rate |
| `--dest` | - | - | Destination directory for results (required) |
| `--num_workers` | 5 | - | Number of worker processes |
| `--gpu` | False | - | Enable GPU usage (flag) |
| `--debug` | False | - | Run in debug mode with reduced dataset (flag) |
| `--model` | enet |ENet, UNet, VNet, SUNet, UNetDR| Model architecture to use |
| `--filter` | None | gaussian, median, non_local_means, bilateral, wavelet | Preprocessing filter |
| `--loss` | CE | CE, Dice, DiceCE, generalised_dice, multiclass_dice, tversky | Loss function |
| `--random_crop_h` | 100 | - | Height for random crop |
| `--random_crop_w` | 100 | - | Width for random crop |
| `--resume` | None | - | Path to model for resuming training |
| `--best_dice` | 0 | - | Best dice value of old model (for resuming) |
| `--dropout_prob` | 0.2 | - | Dropout probability |

#### Multi-phase training
Example of multi-phase training, loading a model trained on *cross entropy* and further training on the *tversky loss*
```
$ python main.py --dataset SEGTHOR --mode full --epochs 50 --dest results/segthor/SUNet_Hyb_CE/ce --gpu --model 'SUNet' --loss 'tversky' --resume 'model/ce_model.pkl'

```

| Parameter | Description |
|-----------|-------------|
| `--dest results/...` | Defines the output directory for results |
| `--model 'SUNet'` | Specifies the model architecture |
| `--loss 'tversky'` | Sets Tversky loss for the fine-tuning phase |
| `--resume 'model/ce_model.pkl'` | Path to the pre-trained model (trained with cross-entropy) |

---------


### nnU-Net Training
Before training make sure to run 'jobs/nnUNet-Setup.job'. 
```
# Training XXX inside thennU-Net pipeline
$ nnUNetv2_train 1 2d 1

# Training XXX inside the nnU-Net pipeline
$ nnUNetv2_train 1 3d_fullres 1
$ nnUNetv2_train 1 3d_lowres 1


```
### Loss functions

## Visuals - patient 16
| Ground Truth | U-Net CE | U-Net DR | nnU-Net DR |
|--------------|---------------------|-------------------------------|----------------------------------|
| ![Ground Truth](images/GT.jpeg) | ![U-Net CE](images/unet_ce.jpeg) | ![U-Net DR](images/unet+dr.jpeg) | ![nnU-Net DR](images/nnunet+DR.jpeg) |


## Metrics Computation 
TO DO; link to file with HD95 computation and DSC (VARDAN?)

### Results on internal validation set (DSC)

| Model    | Loss | Esophagus | Heart     | Trachea   | Aorta     |
|----------|------|-----------|-----------|-----------|-----------|
| E-Net    | ce   | 0.704     | 0.933     | 0.908     | 0.841     |
| E-Net    | dl   | 0.751     | 0.900     | 0.918     | 0.802     |
| E-Net    | dlce | 0.730     | 0.917     | 0.892     | 0.845     |
| U-Net    | dl   | 0.653     | 0.633     | 0.847     | 0.689     |
| U-Net    | ce   | 0.780     | *0.950*   | 0.930     | 0.900     |
| U-Net    | dlce | 0.787     | 0.918     | 0.921     | 0.880     |
| U-Net    | wce  | 0.770     | **0.951** | 0.942     | 0.873     |
| sU-Net   | dl   | 0.613     | 0.641     | 0.701     | 0.754     |
| sU-Net   | ce   | *0.812*   | 0.846     | 0.940     | 0.919     |
| sU-Net   | dlce | **0.819** | 0.916     | *0.944*   | **0.923** |
| U-Net+DR | dl   | 0.545     | 0.458     | 0.760     | 0.743     |
| U-Net+DR | ce   | 0.805     | 0.882     | **0.945** | *0.922*   |
| U-Net+DR | dlce | 0.811     | 0.891     | 0.909     | 0.918     |
| 2D V-Net | ce   | 0.723     | 0.879     | 0.935     | 0.827     |

### Results on test set
## TO DO

[] how to run training description

[] inference?

[] metrics computation 

[] affine script fixing

[] storing the best saved model

[] predictions on internal validation set, labels of validations set 

[] metrics computed (Need to be 3d?)

## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`;
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files;
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

### Setting up the environment
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```

For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

### Viewing the data
The data can be viewed in different ways:
- looking directly at the `.` in the sliced folder (`data/SEGTHOR`);
- using the provided "viewer" to compare segmentations ([see below](#viewing-the-results));
- opening the Nifti files from `data/segthor_train` with [3D Slicer](https://www.slicer.org/) or [ITK Snap](http://www.itksnap.org).

### Training a base network
Running a training
```
$ python main.py --help
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--gpu] [--debug]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights).
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
$ python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

The codebase uses a lot of assertions for control and self-documentation, they can easily be disabled with the `-O` option (for faster training) once everything is known to be correct (for instance run the previous command for 1/2 epochs, then kill it and relaunch it):
```
$ python -O main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

### Viewing the results
#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back):
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](images/viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


```
$ python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
    --legend --class_names background esophagus heart trachea aorta
```
![Example of the viewer on SegTHOR](images/viewer_segthor.png)

#### 3D viewers
To look at the results in 3D, it is necessary to reconstruct the 3D volume from the individual 2D predictions saved as images.
To stitch the `.png` back to a nifti file:
```
$ python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
```

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](images/3dslicer_zoom.png)


### Plotting the metrics
There are some facilities to plot the metrics saved by [`main.py`](main.py):
```
$ python plot.py --help
usage: plot.py [-h] --metric_file METRIC_MODE.npy [--dest METRIC_MODE.png] [--headless]

Plot data over time

options:
  -h, --help            show this help message and exit
  --metric_file METRIC_MODE.npy
                        The metric file to plot.
  --dest METRIC_MODE.png
                        Optional: save the plot to a .png file
  --headless            Does not display the plot and save it directly (implies --dest to be provided.
$ python plot.py --metric_file results/segthor/ce/dice_val.npy --dest results/segthor/ce/dice_val.png
```
![Validation DSC](images/dice_val.png)

Plotting and visualization ressources:
* [Scientific visualization Python + Matplotlib](https://github.com/rougier/scientific-visualization-book)
* [Seaborn](https://seaborn.pydata.org/examples/index.html)
* [Plotly](https://github.com/plotly/plotly.py)

## Submission and scoring
Groups will have to submit:
* archive of the git repo with the whole project, which includes:
    * pre-processing;
    * training;
    * post-processing where applicable;
    * inference;
    * metrics computation;
    * script fixing the data using the matrix `AFF` from `affine.py` (or rather its inverse);
    * (bonus) any solution fixing patient27 without recourse to `affine.py`;
    * (bonus) any (even partial) solution fixing the whole dataset without recourse to `affine.py`;
* the best trained model;
* predictions on the [test set](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EWZH7ylUUFFCg3lEzzLzJqMBG7OrPw1K4M78wq9t5iBj_w?e=Yejv5d) (`sha256sum -c data/test.zip.sha256` as optional checksum);
* predictions on the group's internal validation set, the labels of their validation set, and the metrics they computed.

The main criteria for scoring will include:
* improvement of performances over baseline;
* code quality/clear [git use](git.md);
* the [choice of metrics](https://metrics-reloaded.dkfz.de/) (they need to be in 3D);
* correctness of the computed metrics (on the validation set);
* (part of the report) clear description of the method;
* (part of the report) clever use of visualization to report and interpret your results;
* report;
* presentation.

The `(bonus)` lines give extra points, that can ultimately compensate other parts of the project/quizz.


### Packing the code
`$ git bundle create group-XX.bundle master`

### Saving the best model
`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

### Archiving everything for submission
All files should be grouped in single folder with the following structure
```
group-XX/
    test/
        pred/
            Patient_41.nii.gz
            Patient_42.nii.gz
            ...
    val/
        pred/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        gt/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        metric01.npy
        metric02.npy
        ...
    group-XX.bundle
    bestmodel-group-XX.pkl
```
The metrics should be numpy `ndarray` with the shape `NxKxD`, with `N` the number of scan in the subset, `K` the number of classes (5, including background), and `D` the eventual dimensionality of the metric (can be simply 1).

The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:
```
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
$ tar cf group-XX.tar.gz - group-XX/
```


