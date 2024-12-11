import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import torch
import yaml


import lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy


from utils.utils import seed_everything
from datasets.data_loaders import get_dataloaders
from models.vqvae_2 import LaBRAM_VQVAE
from models.model_utils import compute_fft



class Pl_VQVAE_tokenizer(pl.LightningModule):
    def __init__(self,args,vqvae_params,data_set_params,training_params,save_path):
        super().__init__()
        self.args = args
        self.vqvae_params = vqvae_params
        self.data_set_params = data_set_params
        self.training_params = training_params
        
        self.save_path = save_path
        
       
        print('Using LaBRAM VQVAE')
        self.vqvae = LaBRAM_VQVAE(in_channels=1,
                        kernel_size =200,
                        stride=200,
                         smallest_kernel_divider=5,
                         emb_size=64,
                         code_book_size= self.vqvae_params['code_book_size'],
                         beta=0.2)
       
        with open(os.path.join(self.save_path,'vqvae_model.txt'),'w') as f:
            f.write(str(self.vqvae))
            f.close()
            
        self.val_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        return self.vqvae(x)
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.AEmodel.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        # )
        # ADAMW optimizer
        if self.training_params['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.vqvae.parameters(), 
                lr=self.training_params['lr'], 
                weight_decay=self.training_params['weight_decay'],
                betas = (self.training_params['beta1'],self.training_params['beta2'])
            )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )
        
        return [optimizer], [scheduler]
    

    def training_step(self, train_batch, batch_idx):
        X, _ = train_batch # X: (batch, channels, freq, time)
        
        

        X = X.view(-1,X.shape[-1]).unsqueeze(1)
        fft_mag,fft_angle = compute_fft(X.view(-1,X.shape[-1]//200,200))
        
        
            
            
        fft_mag_pred,fft_angle_pred,x_tokens,quant_out,quant_in = self.vqvae(X)
            
        recon_loss = torch.nn.MSELoss()(fft_mag,fft_mag_pred) + torch.nn.MSELoss()(fft_angle,fft_angle_pred)
        quant_loss, code_book_loss, commitment_loss = self.vqvae.vec_quantizer_loss(quant_in,quant_out)
            
        loss = recon_loss + quant_loss
            
            
            
                
        self.log('train_step_loss', loss,prog_bar=True,sync_dist=True)
        self.log('train_step_code_book_loss', code_book_loss,sync_dist=True)
        self.log('train_step_commitment_loss', commitment_loss,sync_dist=True)
        self.log('train_step_recon_loss', recon_loss,sync_dist=True)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        X, _ = val_batch# X: (batch, channels, freq, time)
        
        
        with torch.no_grad():
            
            X = X.view(-1,X.shape[-1]).unsqueeze(1)
            fft_mag,fft_angle = compute_fft(X.view(-1,X.shape[-1]//200,200))
                
            fft_mag_pred,fft_angle_pred,x_tokens,quant_out,quant_in = self.vqvae(X)
            
            recon_loss = torch.nn.MSELoss()(fft_mag,fft_mag_pred) + torch.nn.MSELoss()(fft_angle,fft_angle_pred)
            quant_loss, code_book_loss, commitment_loss = self.vqvae.vec_quantizer_loss(quant_in,quant_out)
        
            loss = recon_loss.cpu().numpy() + quant_loss.cpu().numpy()
                    
                
            self.val_step_outputs.append([loss, code_book_loss.cpu().numpy(), commitment_loss.cpu().numpy(), recon_loss.cpu().numpy()])
            
            return loss
    
    def on_validation_epoch_end(self):
        val_loss = np.mean([i[0] for i in self.val_step_outputs])
        code_book_loss = np.mean([i[1] for i in self.val_step_outputs])
        commitment_loss = np.mean([i[2] for i in self.val_step_outputs])
        recon_loss = np.mean([i[3] for i in self.val_step_outputs])
        
        self.log('val_epoch_loss', val_loss,sync_dist=True)
        self.log('val_epoch_code_book_loss', code_book_loss,sync_dist=True)
        self.log('val_epoch_commitment_loss', commitment_loss,sync_dist=True) 
        self.log('val_epoch_recon_loss', recon_loss,sync_dist=True)
        
        self.val_step_outputs = []
        
        return val_loss
        

                
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='SHHS', help='Dataset name')
    parser.add_argument('--signal_transform', type=str, default=None, help='Signal transformation')
    parser.add_argument('--resampling_rate', type=int, default=256, help='Resampling rate')
    parser.add_argument('--gpu',type=str, default=None, help='GPU to use')
    
    args = parser.parse_args()
    
    # read the configuration file
    with open("./configs/vqvae_spec_pretraining.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    # Parameters
    data_set_params = config['Dataset'][args.dataset_name]
    training_params = config['vqvae_training']
    if args.signal_transform is None:
        vqvae_params = config['temporal']['vqvae']
    else:
        vqvae_params = config[args.signal_transform]['vqvae']
        
    # 'window_len_256'+
    # experiment_name = 'freq_bin_masking_temporal_separate_MSE_full_freq_window_len_200'+ args.dataset_name + '_' + str(args.signal_transform) + '_FREQ_VQVAE_pretraining_codebook_size_' + str(vqvae_params['code_book_size']) + '_emb_size_' + str(vqvae_params['emb_size'])+ 'beta' + str(vqvae_params['beta'])
    experiment_name = 'LaBRAM_decoding_task_'+ args.dataset_name + '_' + str(args.signal_transform) + '_FREQ_VQVAE_pretraining_codebook_size_' + str(vqvae_params['code_book_size']) + '_emb_size_' + str(vqvae_params['emb_size'])+ 'beta' + str(vqvae_params['beta'])
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
        f.write('VQVAE parameters:\n')
        f.write(str(vqvae_params))
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
    
    print('Testing the dataloaders')
    for i, (X, _) in enumerate(train_loader):
        print(f'Batch {i}: {X.shape}')
        break
    
    # initializing logger
    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",  # replace with your own
        project="jathurshan/BPE-EEG-VQVAE",  # format "workspace-name/project-name"
        tags=[experiment_name, args.dataset_name,'LaBRAM'],
        log_model_checkpoints=True,  # save checkpoints
    )
    
    ## Pretraining the VQVAE
    
    pl_vqvae = Pl_VQVAE_tokenizer(
        args = args,
        vqvae_params = vqvae_params,
        data_set_params = data_set_params,
        training_params = training_params,
        save_path = save_path
    )
    logger.log_model_summary(pl_vqvae)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                          save_top_k=1, 
                                          monitor="val_epoch_recon_loss",
                                          mode="min", 
                                          filename="best_model",
                                          save_weights_only=True,
                                          save_last = True)
    
    
    
    
    # Train the model
    if args.gpu:
        trainer = pl.Trainer(
            default_root_dir = save_path,
            devices = [int(gpu) for gpu in args.gpu.split(',')],
            accelerator = 'gpu',
            strategy = DDPStrategy(find_unused_parameters=True),#'ddp',
            enable_checkpointing = True,
            callbacks = [checkpoint_callback],
            max_epochs = training_params['num_pretrain_epochs'],
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
            max_epochs = training_params['num_pretrain_epochs'],
            logger = logger,
            deterministic = True,
            fast_dev_run = False # for development purpose only
        )
    
    trainer.fit(pl_vqvae, 
                train_dataloaders=train_loader, 
                val_dataloaders = val_loader)
    
    print('Pretraining completed')
    
    # save vqvae model
    # load the best model
    best_model_path = os.path.join(save_path,'best_model.ckpt')
    pl_vqvae = Pl_VQVAE_tokenizer.load_from_checkpoint(best_model_path,args = args,vqvae_params = vqvae_params,
        data_set_params = data_set_params,
        training_params = training_params,
        save_path = save_path)
    torch.save(pl_vqvae.vqvae.state_dict(),os.path.join(save_path,'vqvae_best_model.pth'))
    
    # load and save last model
    last_model_path = os.path.join(save_path,'last.ckpt')
    pl_vqvae = Pl_VQVAE_tokenizer.load_from_checkpoint(last_model_path,args = args,vqvae_params = vqvae_params,
        data_set_params = data_set_params,
        training_params = training_params,
        save_path = save_path)
    torch.save(pl_vqvae.vqvae.state_dict(),os.path.join(save_path,'vqvae_last_model.pth'))
    
    print('Model saved')
    
    

# temporal
# python LaBRAM_vqvae_tokenizer_pretraining.py --dataset_name 'TUEV'  --resampling_rate 200 --gpu '6'