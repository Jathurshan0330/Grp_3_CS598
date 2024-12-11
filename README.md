# Group 3 Eval Reproductions

This repository contains code for to reproduce synthetic signal experiments and validation of byte-pair encoding.

## Installation
Clone this repository
```
git clone https://github.com/your_username/repo_name.git
cd repo_name
```

Create a conda environment and install the required dependencies
```
conda create --name <env_name> python=3.9.7
conda activate <env_name>
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name=<env_name>
pip install -r requirements.txt
```

Run the jupyter notebook to reproduce the experiment and visualize the results. The results figures will be saved at ./sythetic_signals path.


# EEG Tokenizer Reproduction

## Dataset Processing
Run the process.py under the dataset processing code. Specifically, run the file under the TUEV folder.

## EEG tokenizer training
Under the config folder and file vqvae_spec_pretraining.yaml set the paths to the datasets and results path to save the results. Then run the following to train the EEG tokenizer.
```
python freq_vqvae_tokenizer_pretraining.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --gpu '1'
```

## Masked Token Prediction Pretraining
Run the following to pretrain the transformer classifier with EEG tokens.
```
python freq_vqvae_tokens_masked_modeling.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path 'PATH_TO_VQVAE/vqvae_last_model.pth' --gpu '6' --restart_embedding

```


## Downstream Finetuning
Run the following script to finetune the model to downstream tasks.

```
python freq_vqvae_tokens_aft_masked_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path 'PATH_to_VQVAE/vqvae_last_model.pth'  --masked_pretrained_path 'PATH_TO_PRETRAINED_MODEL'   --gpu '6' --restart_embedding
```

# Byte-pair encoding
To learn to tokens using byte-pair encoding followed by masked token prediction pretraining and finetuning run the following:

```
python freq_vqvae_learn_bpe.py --dataset_name 'TUEV'  --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path 'PATH_TO_VQVAE/vqvae_last_model.pth'  --num_BPE_learn 512 --compress_noise --gpu '0'
```
```
python freq_vqvae_tokens_BPE_masked_modeling.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path 'PATH_TO_VQVAE/vqvae_last_model.pth' --num_BPE_learn 512 --gpu '6' --restart_embedding

```
```
python freq_vqvae_tokens_BPE_aft_masked_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path 'PATH_TO_VQVAE/vqvae_last_model.pth' --masked_pretrained_path 'PATH_TO_PRETRAINED_MODEL'   --num_BPE_learn 512 --gpu '6' --restart_embedding

```

Additionally, the baselines can be reproduced by running the codes in the Baseline_reproduction_scripts folder.


