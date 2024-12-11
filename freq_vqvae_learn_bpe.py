from operator import is_
import numpy as np 
import json
import torch
from torch import nn
import yaml
import os 
from tqdm import tqdm
import time
import argparse

from datasets.data_loaders import get_dataloaders
from models.vqvae_2 import Freq_VQVAE, Temporal_VQVAE

from utils.utils import seed_everything
from utils.bpe_freq_vqvae import learn_bpe

import neptune



def learn_BPE_script():
    parser = argparse.ArgumentParser()
    
    #tokenizer parameters
    parser.add_argument('--dataset_name', type=str, default='SHHS', help='Dataset name')
    parser.add_argument('--signal_transform', type=str, default=None, help='Signal transformation')
    parser.add_argument('--resampling_rate', type=int, default=256, help='Resampling rate')
    
    parser.add_argument('--gpu',type=str, default=None, help='GPU to use')
    parser.add_argument('--vqvae_pretrained_path',type=str, default='vqvae_pretrained.pth', help='Path to the pretrained vqvae model')
    
    parser.add_argument('--num_BPE_learn', type=int, default=100, help='Maximum vocabulary size for BPE')
    parser.add_argument('--compress_noise', action='store_true', help='Compress noise')
    
    
    
    args = parser.parse_args()
    
    # read config file
    with open("./configs/vqvae_spec_pretraining.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
        
    
    data_set_params = config['Dataset'][args.dataset_name]
    training_params = config['classifier_training']
    if args.signal_transform is None:
        tokenizer_params = config['temporal']['vqvae']
    else:
        tokenizer_params = config[args.signal_transform]['vqvae']
        
    
    # save path
    save_path = '/'.join(args.vqvae_pretrained_path.split('/')[:-1]) +'/BPE'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = 'cuda'
        
    run = neptune.init_run(
            project="jathurshan/BPE-EEG-VQVAE",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",
            tags=[args.dataset_name,str(args.signal_transform),str(args.num_BPE_learn),'compress_noise='+str(args.compress_noise),'BPE',args.vqvae_pretrained_path.split('/')[-2] ],
        ) 
    # logger = NeptuneLogger(
    #         api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",  # replace with your own
    #         project="jathurshan/EEG-BPE-Classifier-Experiments",  # format "workspace-name/project-name"
    #         tags=[args.dataset_name,str(args.signal_transform),str(args.num_BPE_learn),'compress_noise='+str(args.compress_noise),'BPE',args.vqvae_pretrained_path.split('/')[-2] ],
    #         log_model_checkpoints=True,  # save checkpoints
    #     )
    
    # Create the dataloaders
    seed_everything(5)
    train_loader = get_dataloaders(data_dir=data_set_params['data_dir'],
                                   dataset_name=args.dataset_name,
                                   train_val_test='train',
                                   batch_size=training_params['batch_size'], 
                                   channels=data_set_params['channels'],
                                   num_workers=training_params['num_workers'],
                                   signal_transform = args.signal_transform,
                                   resampling_rate = args.resampling_rate,
                                   is_bpe = True)
    print('Testing the dataloader')
    for i, (data, target) in enumerate(train_loader):
        print(data.shape)
        break
    
    # Load vqvae tokenizer
    if args.signal_transform:
        print('Using Freq VQVAE')
        tokenizer = Freq_VQVAE(in_channels=tokenizer_params['in_channels'],
                        n_freq = tokenizer_params['n_freq'],
                        emb_size= tokenizer_params['emb_size'],
                        code_book_size= tokenizer_params['code_book_size'],
                        beta = tokenizer_params['beta'])
        print('Loading pretrained VQVAE model :',args.vqvae_pretrained_path)
        tokenizer.load_state_dict(torch.load(args.vqvae_pretrained_path))
        tokenizer.to(device).eval()
    else:
        print('Using Temporal VQVAE')
        tokenizer = Temporal_VQVAE(in_channels=tokenizer_params['in_channels'],
                                    kernel_size = tokenizer_params['kernel_size'],
                                    stride = tokenizer_params['stride'],
                                    smallest_kernel_divider = tokenizer_params['smallest_kernel_divider'],
                                    emb_size = tokenizer_params['emb_size'],
                                    code_book_size = tokenizer_params['code_book_size'],
                                    beta = tokenizer_params['beta'])
        print('Loading pretrained VQVAE model :',args.vqvae_pretrained_path)
        tokenizer.load_state_dict(torch.load(args.vqvae_pretrained_path))
        tokenizer.to(device).eval()
        
        
        
    
        
    
    max_vocab_size = tokenizer_params['code_book_size'] + args.num_BPE_learn
    bpe_vocab = {}
        
    print('Learning BPE')
    start = time.time()
    
    learned_bpe_vocab,noise_token = learn_bpe(
        val_loader=train_loader,
        bpe_vocab=bpe_vocab,
        max_vocab_size=max_vocab_size,
        tokenizer_model=tokenizer,
        data_set_params=data_set_params,
        vqvae_params=tokenizer_params,
        device=device,
        compress_noise=args.compress_noise)
    end = time.time()
    print(f'Time taken to learn {args.num_BPE_learn} BPE:',end-start)
    
    with open(f"""{save_path}/time_to_learn_BPE_{args.num_BPE_learn}_init_vocab_{tokenizer_params['code_book_size']}_compressed_noise_{args.compress_noise}.txt""", 'w') as f:
        f.write(str(end-start))
    
    # save noise token
    noise_token_dict = {'noise_token':noise_token}
    with open(f"""{save_path}/noise_token.json""", 'w') as f:
        json.dump(noise_token_dict, f)
    
    # 512
    with open(f"""{save_path}/bpe_vocab_{args.num_BPE_learn}_init_vocab_{tokenizer_params['code_book_size']}_compressed_noise_{args.compress_noise}.json""", 'w') as f:
        json.dump(learned_bpe_vocab, f)
        
    # 64
    bpe_vocab_64 = {k: v for k,v in learned_bpe_vocab.items() if k < tokenizer_params['code_book_size']+64}
    with open(f"""{save_path}/bpe_vocab_{64}_init_vocab_{tokenizer_params['code_book_size']}_compressed_noise_{args.compress_noise}.json""", 'w') as f:
        json.dump(bpe_vocab_64, f)
        
    # 128
    bpe_vocab_128 = {k: v for k,v in learned_bpe_vocab.items() if k < tokenizer_params['code_book_size']+128}
    with open(f"""{save_path}/bpe_vocab_{128}_init_vocab_{tokenizer_params['code_book_size']}_compressed_noise_{args.compress_noise}.json""", 'w') as f:
        json.dump(bpe_vocab_128, f)
    
    #256
    bpe_vocab_256 = {k: v for k,v in learned_bpe_vocab.items() if k < tokenizer_params['code_book_size']+256}
    with open(f"""{save_path}/bpe_vocab_{256}_init_vocab_{tokenizer_params['code_book_size']}_compressed_noise_{args.compress_noise}.json""", 'w') as f:
        json.dump(bpe_vocab_256, f)
    
    #print the length of the learned BPE
    print('Length of learned BPE:',len(learned_bpe_vocab))
    print('Length of learned BPE 64:',len(bpe_vocab_64))
    print('Length of learned BPE 128:',len(bpe_vocab_128))
    print('Length of learned BPE 256:',len(bpe_vocab_256))
    print('BPE learned and saved')
    
        
if __name__ == '__main__':
    learn_BPE_script()
    

# TUAB
    # stft window 4s hoplen 0.5s
    # python freq_vqvae_learn_bpe.py --dataset_name 'TUAB'  --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth'  --num_BPE_learn 512 --compress_noise --gpu '7'
    # multitaper window 4s hoplen 0.5s
    # python freq_vqvae_learn_bpe.py --dataset_name 'TUAB'  --signal_transform 'multitaper' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_multitaper_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth'  --num_BPE_learn 512 --compress_noise --gpu '0'

# SHHS
    # stft window 4s hoplen 0.5s : (for 250 batches of 1024) # using best model checkpoint not last
    # python freq_vqvae_learn_bpe.py --dataset_name 'SHHS'  --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/SHHS_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_best_model.pth'  --num_BPE_learn 512 --compress_noise #--gpu '7'
    
    
# TUEV temporal separation
    # python freq_vqvae_learn_bpe.py --dataset_name 'TUEV'  --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/temporal_separate_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth'  --num_BPE_learn 512 --compress_noise --gpu '0'


## TUEV freq bin masking
    # python freq_vqvae_learn_bpe.py --dataset_name 'TUEV'  --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/freq_bin_masking_temporal_separate_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth'  --num_BPE_learn 512 --compress_noise --gpu '0'
