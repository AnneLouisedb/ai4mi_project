#!/bin/bash
#SBATCH --job-name=my_gpu_job          # Name of your job
#SBATCH --output=output_%j.log         # Standard output log (%j will be replaced with the job ID)
#SBATCH --error=error_%j.log           # Error log (%j will be replaced with the job ID)
#SBATCH --partition=gpu                # The partition (queue) to run the job on (change if necessary)
#SBATCH --gres=gpu:1                   # Request 1 GPU (adjust if more are needed)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task (adjust as necessary)
#SBATCH --mem=16G                      # Amount of memory to allocate (adjust as necessary)
#SBATCH --time=04:00:00                # Time limit for the job (4 hours in this case)
#SBATCH --mail-type=END                # Get an email when the job finishes (optional)
#SBATCH --mail-user=your.email@domain.com  # Your email address (optional)


# Activate your pip-based virtual environment

source ~/ai4mi/bin/activate
pip install SimpleITK nibabel
python -m pip install -r requirements.txt
pip install torch torchvision nibabel numpy tqdm

python main.py --epochs 200 --dataset SEGTHOR --mode full --dest ./results --model vnet --gpu



