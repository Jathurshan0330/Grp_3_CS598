from cgi import test
import os
import argparse
import re
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
import json
import lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy



from pyhealth.metrics import multiclass_metrics_fn,binary_metrics_fn
from utils.utils import seed_everything, class_wise_f1_score, BCE, focal_loss
from datasets.data_loaders import get_dataloaders

from models.model_blocks import PositionalEncoding
from models.vqvae_2 import Freq_VQVAE, Temporal_VQVAE
from models.classifier import LAT_Classifier





class Pl_LAT_Masked_Modelig_Freq_VQVAE(pl.LightningModule):
    def __init__(self,
                args,
                tokenizer_params,
                data_set_params,
                training_params,
                save_path):
        super().__init__()
        
        self.args = args
        self.tokenizer_params = tokenizer_params
        self.data_set_params = data_set_params
        self.training_params = training_params
        self.save_path = save_path
        
        self.num_channels = data_set_params['num_channels']
        
        
        if self.args.restart_embedding:
                # create a new embedding layer
                print('Restarting the embedding layer')
                # additional token for masks
                self.eeg_token_embedding = nn.Embedding(self.tokenizer_params['code_book_size']+1, self.tokenizer_params['emb_size'])
                
        
        if args.signal_transform:
            print('Using Freq VQVAE')
            self.tokenizer = Freq_VQVAE(in_channels=self.tokenizer_params['in_channels'],
                            n_freq = self.tokenizer_params['n_freq'],
                            emb_size= self.tokenizer_params['emb_size'],
                            code_book_size= self.tokenizer_params['code_book_size'],
                            beta = self.tokenizer_params['beta'])
            print('Loading pretrained VQVAE model :',args.vqvae_pretrained_path)
            self.tokenizer.load_state_dict(torch.load(args.vqvae_pretrained_path))
            self.tokenizer.to(self.device).eval()
        else:
            print('Using Temporal VQVAE')
            self.tokenizer = Temporal_VQVAE(in_channels=self.tokenizer_params['in_channels'],
                                        kernel_size = tokenizer_params['kernel_size'],
                                        stride = tokenizer_params['stride'],
                                        smallest_kernel_divider = tokenizer_params['smallest_kernel_divider'],
                                        emb_size = self.tokenizer_params['emb_size'],
                                        code_book_size = self.tokenizer_params['code_book_size'],
                                        beta = self.tokenizer_params['beta'])
            print('Loading pretrained VQVAE model :',args.vqvae_pretrained_path)
            self.tokenizer.load_state_dict(torch.load(args.vqvae_pretrained_path))
            self.tokenizer.to(self.device).eval()
            
            
        
            
        
    
        self.channel_tokens = nn.Embedding(self.num_channels, self.tokenizer_params['emb_size'])
        self.positional_encoding = PositionalEncoding(self.tokenizer_params['emb_size'], max_len=4096)#self.tokenizer_params['max_seq_len'])
        
        self.lat_classifier = LAT_Classifier(
                            emb_size = self.tokenizer_params['emb_size'],
                            num_heads =  8,
                            depth = 12,
                            max_seq_len = 4096,
                            n_classes = self.tokenizer_params['code_book_size'])
        
        
        with open(os.path.join(self.save_path,'model_summary.txt'),'w') as f:
            f.write('VQVAE model summary\n')
            f.write(str(self.tokenizer))
            f.write('\n\n')
            f.write('Classifier model summary\n')
            f.write(str(self.lat_classifier))
            f.close()
        
        self.val_step_outputs = []
        self.test_step_outputs = []
        
        
    def forward(self, X):

        
        self.tokenizer.eval()
        
        if self.args.signal_transform:
            B,C,F,T = X.shape
            X = X.view(-1,F,T)
            x_embed, x_tokens = self.tokenizer.tokenize(X)
            
            
            ##### Masked Modeling ##########
            #create a random mask of 0s and 1s for x_tokens
            mask = torch.randint(0,2,x_tokens.shape,device=x_tokens.device)
            # print(f'Mask shape: {mask.shape}')
            #place mask token where the mask is 0
            x_tokens_masked = torch.where(mask==0,torch.tensor(self.tokenizer_params['code_book_size'],device=x_tokens.device),x_tokens)
            # print(f'X tokens masked shape: {x_tokens_masked.shape}')
            
            # ###### Replaced Token Prediction ##########
            # mask = torch.randint(0,2,x_tokens.shape,device=x_tokens.device)
            # replace_token = torch.randint(0,self.tokenizer_params['code_book_size'],x_tokens.shape,device=x_tokens.device)
            # x_tokens_masked = torch.where(mask==0,replace_token,x_tokens)
            
            if self.args.restart_embedding:
                x_embed = self.eeg_token_embedding(x_tokens_masked)
            
            # add positional encoding
            x_embed = self.positional_encoding(x_embed)
            
            x_embed = x_embed.view(B,C,x_embed.shape[-2],x_embed.shape[-1])
            x_tokens = x_tokens.view(B,C,x_tokens.shape[-1])
            mask = mask.view(B,C,mask.shape[-1])
            # print(f'X tokens shape: {x_tokens.shape}, mask shape: {mask.shape}')
            
            # add channel embeddings
            channel_embeddings = self.channel_tokens(torch.arange(C, device=x_embed.device))
            x_embed = x_embed + channel_embeddings.unsqueeze(0).unsqueeze(2)
            
            x_embed = x_embed.view(B,C*x_embed.shape[-2],x_embed.shape[-1])
            x_tokens = x_tokens.view(B,C*x_tokens.shape[-1])
            mask = mask.view(B,C*mask.shape[-1])
            # print(f'X tokens shape: {x_tokens.shape}, mask shape: {mask.shape}')
        
        
        pred_x_token = self.lat_classifier.masked_prediction(x_embed)
        
        pred_x_token = pred_x_token.view(-1,self.tokenizer_params['code_book_size'])
        x_tokens = x_tokens.view(-1,)
        mask = mask.view(-1,)
        # print(f'Pred x token shape: {pred_x_token.shape}, x tokens shape: {x_tokens.shape}')
        # print(f'Mask shape: {mask.shape}')
        
        return pred_x_token, x_tokens, mask
        
    def configure_optimizers(self):
        if self.training_params['optimizer'] == 'AdamW':
            if self.args.restart_embedding:
                optimizer = torch.optim.AdamW(
                    list(self.eeg_token_embedding.parameters()) + list(self.lat_classifier.parameters()),
                    lr=self.training_params['lr'], 
                    weight_decay=self.training_params['weight_decay'],
                    betas = (self.training_params['beta1'],self.training_params['beta2'])
                )
            else:
                optimizer = torch.optim.AdamW(
                    self.lat_classifier.parameters(),
                    lr=self.training_params['lr'], 
                    weight_decay=self.training_params['weight_decay'],
                    betas = (self.training_params['beta1'],self.training_params['beta2'])
                )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )
        
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        
        pred_x_token, x_tokens, mask = self(X)
        # only calculate the loss for the masked tokens
        pred_x_token = pred_x_token[mask==0]
        x_tokens = x_tokens[mask==0]
        
        loss = nn.CrossEntropyLoss()(pred_x_token, x_tokens)
        
        self.log('train_step_loss',loss,prog_bar=True,sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        
        with torch.no_grad():
            pred_x_token, x_tokens, mask = self(X)
            # only calculate the loss for the masked tokens
            pred_x_token = pred_x_token[mask==0]
            x_tokens = x_tokens[mask==0]
            
            loss = nn.CrossEntropyLoss()(pred_x_token, x_tokens)
            
            
        
        self.log('val_step_loss', loss,prog_bar=True,sync_dist=True)
        self.val_step_outputs.append(loss)
        return  loss
    
    
    def on_validation_epoch_end(self):
        val_result = torch.stack(self.val_step_outputs).mean()
        self.log('val_epoch_loss', val_result,prog_bar=False,sync_dist=True)
        ## Add other metrics here
        self.val_step_outputs = []
        return val_result
    
    
        
            
            
         

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='SHHS', help='Dataset name')
    parser.add_argument('--signal_transform', type=str, default=None, help='Signal transformation')
    parser.add_argument('--resampling_rate', type=int, default=256, help='Resampling rate')
    
    parser.add_argument('--gpu',type=str, default=None, help='GPU to use')
    parser.add_argument('--vqvae_pretrained_path',type=str, default='vqvae_pretrained.pth', help='Path to the pretrained vqvae model')
    parser.add_argument('--restart_embedding',action='store_true', help='Restart the embedding layer')
    
    # additional arguments
    parser.add_argument('--test_only', action='store_true', help='Test only mode')
    # parser.add_argument('--saved_checkpoint', type=str, default=None, help='Path to the saved checkpoint')
    
    args = parser.parse_args()
    
    # read config file
    with open("./configs/vqvae_spec_pretraining.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
        
    # Parameters
    data_set_params = config['Dataset'][args.dataset_name]
    training_params = config['masked_modeling_training']
    if args.signal_transform is None:
        tokenizer_params = config['temporal']['vqvae']
    else:
        tokenizer_params = config[args.signal_transform]['vqvae']
        
    pretrain_experiment_name = args.vqvae_pretrained_path.split('/')[-2]
    if args.restart_embedding:
        experiment_name = f"masked_token_pred_masked_loss_freq_bin_masking_temporal_separate_full_freq_window_len_200_{args.dataset_name}_restart_embedding_{args.signal_transform}_beta{tokenizer_params['beta']}_LAT_Classification_VQVAE_num_epochs_{training_params['num_epochs']}"
        # experiment_name = f"RTP_temporal_separate_full_freq_window_len_200_{args.dataset_name}_restart_embedding_{args.signal_transform}_beta{tokenizer_params['beta']}_LAT_Classification_VQVAE_num_epochs_{training_params['num_epochs']}"
        
        # experiment_name = f"{args.dataset_name}_restart_embedding_{args.signal_transform}_beta{tokenizer_params['beta']}_LAT_Classification_VQVAE_num_epochs_{training_params['num_epochs']}_pretrained_on_{pretrain_experiment_name}"

    else:
        experiment_name = f"{args.dataset_name}_{args.signal_transform}_beta{tokenizer_params['beta']}_LAT_Classification_VQVAE_num_epochs_{training_params['num_epochs']}_pretrained_on_{pretrain_experiment_name}"
    save_path = os.path.join(training_params['experiment_path'],experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #save the arguments and configuration in a text file
    with open(os.path.join(save_path,'args_config.txt'),'w') as f:
        f.write('Experiment name: ' + experiment_name + '\n')
        f.write('Arguments:\n')
        f.write(str(args))
        f.write('\n----------------------\n')
        f.write('Data set parameters:\n')
        f.write(str(data_set_params))
        f.write('\n----------------------\n')
        f.write('Training parameters:\n')
        f.write(str(training_params))
        f.write('\n----------------------\n')
        f.write('Tokenizer parameters:\n')
        f.write(str(tokenizer_params))
        f.close()
    
    
    
    # Create the dataloaders
    seed_everything(5)
    train_loader = get_dataloaders(data_dir=data_set_params['data_dir'],
                                   dataset_name=args.dataset_name,
                                   train_val_test='train',
                                   batch_size=training_params['batch_size'], 
                                   channels=data_set_params['channels'],
                                   num_workers=training_params['num_workers'],
                                   signal_transform = args.signal_transform,
                                   resampling_rate = args.resampling_rate)
    
    val_loader = get_dataloaders(data_dir=data_set_params['data_dir'],
                                dataset_name=args.dataset_name,
                                train_val_test='val',
                                batch_size=training_params['batch_size'], 
                                channels=data_set_params['channels'],
                                num_workers=training_params['num_workers'],
                                signal_transform = args.signal_transform,
                                resampling_rate = args.resampling_rate)
    
    
    
    
    
    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",  # replace with your own
        project="jathurshan/BPE-EEG-VQVAE",  # format "workspace-name/project-name"
        tags=[experiment_name, args.dataset_name,str(args.signal_transform)],
        log_model_checkpoints=True,  # save checkpoints
    )
    
    print('Testing the dataloaders')
    for i, (X, _) in enumerate(train_loader):
        print(f'Batch {i}: {X.shape}')
        break
    
    # training the classifier
    pl_classifier = Pl_LAT_Masked_Modelig_Freq_VQVAE(args = args,
                                                    tokenizer_params = tokenizer_params,
                                                    data_set_params = data_set_params,
                                                    training_params = training_params,
                                                    save_path = save_path)
    
    logger.log_model_summary(pl_classifier)
    

    checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                        save_top_k=1, 
                                        monitor="val_epoch_loss",
                                        mode="min", 
                                        filename="best_model",
                                        save_weights_only=True,
                                        save_last = True,)
    
    # Train the model
    if args.gpu:
        trainer = pl.Trainer(
            default_root_dir = save_path,
            devices = [int(gpu) for gpu in args.gpu.split(',')],
            accelerator = 'gpu',
            strategy = DDPStrategy(find_unused_parameters=True),#'ddp',
            enable_checkpointing = True,
            callbacks = [checkpoint_callback],
            max_epochs = training_params['num_epochs'],
            logger = logger,
            deterministic = True,
            fast_dev_run = False # for development purpose only
        )
    else:
        trainer = pl.Trainer(
            default_root_dir = save_path,
            accelerator = 'gpu',
            strategy = DDPStrategy(find_unused_parameters=True),#'ddp',
            enable_checkpointing = True,
            callbacks = [checkpoint_callback],
            max_epochs = training_params['num_epochs'],
            logger = logger,
            deterministic = True,
            fast_dev_run = False) # for development purpose only)
    
    
    trainer.fit(pl_classifier, 
                train_dataloaders=train_loader, 
                val_dataloaders = val_loader)
    
    print('Training completed')
        
        
        # print(test_results)
        
        
    best_model_path = os.path.join(save_path,'best_model.ckpt')
            
    pl_classifier = Pl_LAT_Masked_Modelig_Freq_VQVAE.load_from_checkpoint(best_model_path, 
                                                    args = args, 
                                                    tokenizer_params = tokenizer_params, 
                                                    data_set_params = data_set_params, 
                                                    training_params = training_params, 
                                                    save_path = save_path)
    
    torch.save(pl_classifier.lat_classifier.LAT.state_dict(),os.path.join(save_path,'lat_classifier_best_model.pth'))
    torch.save(pl_classifier.eeg_token_embedding.state_dict(),os.path.join(save_path,'eeg_token_embedding_best_model.pth'))
    torch.save(pl_classifier.channel_tokens.state_dict(),os.path.join(save_path,'channel_tokens_best_model.pth'))
    torch.save(pl_classifier.tokenizer.state_dict(),os.path.join(save_path,'tokenizer_best_model.pth'))
    
    # save the last model as pth
    last_model_path = os.path.join(save_path,'last.ckpt')
    pl_classifier = Pl_LAT_Masked_Modelig_Freq_VQVAE.load_from_checkpoint(last_model_path, 
                                                    args = args, 
                                                    tokenizer_params = tokenizer_params, 
                                                    data_set_params = data_set_params, 
                                                    training_params = training_params, 
                                                    save_path = save_path)
        
    torch.save(pl_classifier.lat_classifier.LAT.state_dict(),os.path.join(save_path,'lat_classifier_last_model.pth'))
    torch.save(pl_classifier.eeg_token_embedding.state_dict(),os.path.join(save_path,'eeg_token_embedding_last_model.pth'))
    torch.save(pl_classifier.channel_tokens.state_dict(),os.path.join(save_path,'channel_tokens_last_model.pth'))
    torch.save(pl_classifier.tokenizer.state_dict(),os.path.join(save_path,'tokenizer_last_model.pth'))
    print('Models saved')
    


