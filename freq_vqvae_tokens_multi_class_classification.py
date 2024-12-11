from cgi import test
import os
import argparse
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





class Pl_LAT_Classification_Freq_VQVAE(pl.LightningModule):
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
                self.eeg_token_embedding = nn.Embedding(self.tokenizer_params['code_book_size'], self.tokenizer_params['emb_size'])
                
        
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
                            n_classes = self.data_set_params['num_classes'])
        
        
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
            
            if self.args.restart_embedding:
                x_embed = self.eeg_token_embedding(x_tokens)
            
            # add positional encoding
            x_embed = self.positional_encoding(x_embed)
            
            x_embed = x_embed.view(B,C,x_embed.shape[-2],x_embed.shape[-1])
            x_tokens = x_tokens.view(B,C,x_tokens.shape[-1])
            
            # add channel embeddings
            channel_embeddings = self.channel_tokens(torch.arange(C, device=x_embed.device))
            x_embed = x_embed + channel_embeddings.unsqueeze(0).unsqueeze(2)
            # x_embed = x_embed + self.channel_tokens(torch.arange(C).to(X.device)).unsqueeze(0).unsqueeze(0)
            # print(x_embed.shape,x_tokens.shape,channel_embeddings.shape)
            # print(self.channel_tokens(torch.arange(C).to(X.device)).unsqueeze(0).unsqueeze(0))
            
            x_embed = x_embed.view(B,C*x_embed.shape[-2],x_embed.shape[-1])
            # print(x_embed.shape)
        
        else:
            B,C,T = X.shape
            X = X.view(-1,T).unsqueeze(1)
            x_embed, x_tokens = self.tokenizer.tokenize(X)
            
            if self.args.restart_embedding:
                x_embed = self.eeg_token_embedding(x_tokens)
            
            # add positional encoding
            x_embed = self.positional_encoding(x_embed)
            
            x_embed = x_embed.view(B,C,x_embed.shape[-2],x_embed.shape[-1])
            x_tokens = x_tokens.view(B,C,x_tokens.shape[-1])
            
            # add channel embeddings
            channel_embeddings = self.channel_tokens(torch.arange(C, device=x_embed.device))
            x_embed = x_embed + channel_embeddings.unsqueeze(0).unsqueeze(2)
            # x_embed = x_embed + self.channel_tokens(torch.arange(C).to(X.device)).unsqueeze(0).unsqueeze(0)
            # print(x_embed.shape,x_tokens.shape,channel_embeddings.shape)
            # print(self.channel_tokens(torch.arange(C).to(X.device)).unsqueeze(0).unsqueeze(0))

            x_embed = x_embed.view(B,C*x_embed.shape[-2],x_embed.shape[-1])
            # print(x_embed.shape)
            
        pred = self.lat_classifier(x_embed)
        return pred, x_tokens, x_embed
        
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
        
        pred, _, _ = self(X)
        
        if self.data_set_params['classification_task']=='binary':
            if self.args.dataset_name == 'CHB-MIT':
                loss = focal_loss(pred, y)
            else:
                loss = BCE(pred, y)
        elif self.data_set_params['classification_task']=='multi_class':
            loss = nn.CrossEntropyLoss()(pred, y)
        
        self.log('train_step_loss',loss,prog_bar=True,sync_dist=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        
        with torch.no_grad():
            pred, _, _ = self(X)
            
            if self.data_set_params['classification_task']=='binary':
                if self.args.dataset_name == 'CHB-MIT':
                    loss = focal_loss(pred, y)
                else:
                    loss = BCE(pred, y)
            elif self.data_set_params['classification_task']=='multi_class':
                loss = nn.CrossEntropyLoss()(pred, y)
                
            step_pred = pred.cpu().numpy()
            step_y = y.cpu().numpy()
        
        self.log('val_step_loss', loss,prog_bar=True,sync_dist=True)
        self.val_step_outputs.append([step_pred, step_y])
        return  loss
    
    
    def on_validation_epoch_end(self):
        val_pred = np.concatenate([i[0] for i in self.val_step_outputs])
        val_y = np.concatenate([i[1] for i in self.val_step_outputs])
        
        
        if self.data_set_params['classification_task']=='binary':
            if (sum(val_y) * (len(val_y) - sum(val_y)) != 0):  # to prevent all 0 or all 1 and raise the AUROC error
                self.threshold = np.sort(val_pred)[-int(np.sum(val_y))]
                val_result = binary_metrics_fn(
                    val_y,
                    val_pred,
                    metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                    threshold=self.threshold,
                    )
            else:
                val_result = {
                    "accuracy": 0.0,
                    "balanced_accuracy": 0.0,
                    "pr_auc": 0.0,
                    "roc_auc": 0.0,
                }
            self.log('val_acc', val_result['accuracy'],prog_bar=False,sync_dist=True)
            self.log('val_pr_auc', val_result['pr_auc'],prog_bar=False,sync_dist=True)
            self.log('val_roc_auc', val_result['roc_auc'],prog_bar=False,sync_dist=True)
            self.log('val_balanced_acc', val_result['balanced_accuracy'],prog_bar=False,sync_dist=True)
            
            
        elif self.data_set_params['classification_task']=='multi_class':
            val_result = multiclass_metrics_fn(
                val_y, val_pred, metrics=["accuracy", "cohen_kappa", "f1_weighted",'balanced_accuracy']
            )
        
            self.log('val_acc', val_result['accuracy'],prog_bar=False,sync_dist=True)
            self.log('val_f1', val_result['f1_weighted'],prog_bar=False,sync_dist=True)
            self.log('val_kappa', val_result['cohen_kappa'],prog_bar=True,sync_dist=True)
            self.log('val_balanced_acc', val_result['balanced_accuracy'],prog_bar=False,sync_dist=True)
        ## Add other metrics here
        self.val_step_outputs = []
        return val_result
    
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        
        with torch.no_grad():
            pred, _, _ = self(X)
            
            if self.data_set_params['classification_task']=='binary':
                loss = BCE(pred, y)
                step_pred = torch.sigmoid(pred).cpu().numpy()
            elif self.data_set_params['classification_task']=='multi_class':
                loss = nn.CrossEntropyLoss()(pred, y)
                step_pred = torch.nn.functional.softmax(pred,dim=-1).cpu().numpy()
                
            step_y = y.cpu().numpy()
        

        self.test_step_outputs.append([step_pred, step_y])
        return  loss
    
    def on_test_epoch_end(self):
        test_pred = np.concatenate([i[0] for i in self.test_step_outputs])
        test_y = np.concatenate([i[1] for i in self.test_step_outputs])
        
        
        
        if self.data_set_params['classification_task']=='binary':
            if (sum(test_y) * (len(test_y) - sum(test_y)) != 0):  # to prevent all 0 or all 1 and raise the AUROC error
                self.threshold = np.sort(test_pred)[-int(np.sum(test_y))]
                test_result = binary_metrics_fn(
                    test_y,
                    test_pred,
                    metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                    threshold=self.threshold,
                    )
            else:
                test_result = {
                    "accuracy": 0.0,
                    "balanced_accuracy": 0.0,
                    "pr_auc": 0.0,
                    "roc_auc": 0.0,
                }
            self.log('test_acc', test_result['accuracy'],prog_bar=False,sync_dist=True)
            self.log('test_pr_auc', test_result['pr_auc'],prog_bar=False,sync_dist=True)
            self.log('test_roc_auc', test_result['roc_auc'],prog_bar=False,sync_dist=True)
            self.log('test_balanced_acc', test_result['balanced_accuracy'],prog_bar=False,sync_dist=True)
            
            
        elif self.data_set_params['classification_task']=='multi_class':
            test_result = multiclass_metrics_fn(
                test_y, test_pred, metrics=["accuracy", "cohen_kappa", "f1_weighted",'balanced_accuracy']
            )
        
            self.log('test_acc', test_result['accuracy'],prog_bar=False,sync_dist=True)
            self.log('test_f1', test_result['f1_weighted'],prog_bar=False,sync_dist=True)
            self.log('test_kappa', test_result['cohen_kappa'],prog_bar=True,sync_dist=True)
            self.log('test_balanced_acc', test_result['balanced_accuracy'],prog_bar=False,sync_dist=True)
            
        print(test_result)
        # convert to pd dataframe and save
        # remove f1_classwise from the dictionary
        test_result_df = pd.DataFrame(test_result,index=[0])
        test_result_df.to_csv(os.path.join(self.save_path,'test_results.csv'))
        
        print('Bootstrapping the test results')
        # do bootstrap for all the metrics (except f1_classwise)
        num_of_resamples = 10000
        number_of_samples = len(test_y)//1000
        ci = 0.95
        
        if self.data_set_params['classification_task']=='binary':
            bootstrap_dict = {metric: [] for metric in ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]}
        elif self.data_set_params['classification_task']=='multi_class':
            bootstrap_dict = {metric: [] for metric in ["accuracy", "cohen_kappa", "f1_weighted",'balanced_accuracy']}
        
        
        for i in range(num_of_resamples):
            bootstrap_samples_idx = np.random.choice(len(test_y), number_of_samples, replace=False)
            boot_pred = test_pred[bootstrap_samples_idx]
            boot_y = test_y[bootstrap_samples_idx]
            if self.data_set_params['classification_task']=='binary':
                if (sum(boot_y) * (len(boot_y) - sum(boot_y)) != 0):
                    self.threshold = np.sort(boot_pred)[-int(np.sum(boot_y))]
                    boot_result = binary_metrics_fn(
                        boot_y,
                        boot_pred,
                        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                        threshold=self.threshold,
                    )
                else:
                    boot_result = {
                        "accuracy": 0.0,
                        "balanced_accuracy": 0.0,
                        "pr_auc": 0.0,
                        "roc_auc": 0.0,
                    }
            
            elif self.data_set_params['classification_task']=='multi_class':
                boot_result = multiclass_metrics_fn(
                    boot_y, boot_pred, metrics=["accuracy", "cohen_kappa", "f1_weighted",'balanced_accuracy']  
                )
                
            for metric in bootstrap_dict.keys():
                bootstrap_dict[metric].append(boot_result[metric])
        
        if self.data_set_params['classification_task']=='binary':
            boostrap_results = {metric: {} for metric in ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"]}
        elif self.data_set_params['classification_task']=='multi_class':
            boostrap_results = {metric: {} for metric in ["accuracy", "cohen_kappa", "f1_weighted", 'balanced_accuracy']}
        
        
        for metric, values in bootstrap_dict.items():
            boostrap_results[metric]['boostrap mean'] = np.mean(values)
            boostrap_results[metric]['ci'] = 1.96 * np.std(values) / np.sqrt(num_of_resamples)
        
        # convert to pd dataframe and save
        # bootstrap_df = pd.DataFrame(boostrap_results,index=[0])
        # bootstrap_df.to_csv(os.path.join(self.save_path,'test_bootstrap_results.csv'))
        # save as JSON
        with open(os.path.join(self.save_path,'test_bootstrap_results.json'),'w') as f:
            json.dump(boostrap_results,f)
        
        print('Bootstrapping done and saved')
        self.test_step_outputs = []
        return test_result
        
            
            
         

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
    training_params = config['classifier_training']
    if args.signal_transform is None:
        tokenizer_params = config['temporal']['vqvae']
    else:
        tokenizer_params = config[args.signal_transform]['vqvae']
        
    pretrain_experiment_name = args.vqvae_pretrained_path.split('/')[-2]
    if args.restart_embedding:
        experiment_name = f"temporal_separate_full_freq_window_len_200_{args.dataset_name}_restart_embedding_{args.signal_transform}_beta{tokenizer_params['beta']}_LAT_Classification_VQVAE_num_epochs_{training_params['num_epochs']}"
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
    
    test_loader = get_dataloaders(data_dir=data_set_params['data_dir'],
                                dataset_name=args.dataset_name,
                                train_val_test='test',
                                batch_size=training_params['batch_size'], 
                                channels=data_set_params['channels'],
                                num_workers=training_params['num_workers'],
                                signal_transform = args.signal_transform,
                                resampling_rate = args.resampling_rate)
    
    
    # initializing logger
    if args.test_only:
        logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",  # replace with your own
        project="jathurshan/EEG-BPE-Classifier-Experiments",  # format "workspace-name/project-name"
        tags=[experiment_name, args.dataset_name,str(args.signal_transform),'test_only'],
        log_model_checkpoints=True,  # save checkpoints
    )
    else:
        logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODE4MzUxMC02NWQ0LTQzNjYtYTkwMy1kZDU5YTU0ZjMzZmYifQ==",  # replace with your own
            project="jathurshan/EEG-BPE-Classifier-Experiments",  # format "workspace-name/project-name"
            tags=[experiment_name, args.dataset_name,str(args.signal_transform)],
            log_model_checkpoints=True,  # save checkpoints
        )
    
    print('Testing the dataloaders')
    for i, (X, _) in enumerate(train_loader):
        print(f'Batch {i}: {X.shape}')
        break
    
    # training the classifier
    pl_classifier = Pl_LAT_Classification_Freq_VQVAE(args = args,
                                                    tokenizer_params = tokenizer_params,
                                                    data_set_params = data_set_params,
                                                    training_params = training_params,
                                                    save_path = save_path)
    
    logger.log_model_summary(pl_classifier)
    
    # Callbacks
    if data_set_params['classification_task']=='binary':
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                          save_top_k=1, 
                                          monitor="val_roc_auc", 
                                          mode="max", 
                                          filename="best_model_roc_auc",
                                          save_weights_only=True,
                                          save_last = True,)
        
    elif data_set_params['classification_task']=='multi_class':
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, 
                                            save_top_k=1, 
                                            monitor="val_kappa", 
                                            mode="max", 
                                            filename="best_model_kappa",
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
    
    if args.test_only:
        if data_set_params['classification_task']=='binary':
            # saved_checkpoint_path = os.path.join(save_path,'best_model_roc_auc.ckpt')
            saved_checkpoint_path = os.path.join(save_path,'last.ckpt')
        elif data_set_params['classification_task']=='multi_class':
            # saved_checkpoint_path = os.path.join(save_path,'best_model_kappa.ckpt')
            saved_checkpoint_path = os.path.join(save_path,'last.ckpt')
        print(f'Testing the model using the saved checkpoint: {saved_checkpoint_path}')
        test_results = trainer.test(model = pl_classifier,
                     ckpt_path = saved_checkpoint_path,
                     dataloaders=test_loader)   
        print(test_results)
        print('Testing completed') 
    
    else:
        trainer.fit(pl_classifier, 
                    train_dataloaders=train_loader, 
                    val_dataloaders = val_loader)
        
        print('Training completed')
        
        # Testing the classifier
        test_results = trainer.test(model = pl_classifier,
                        ckpt_path = 'last', #'best',
                        dataloaders=test_loader)
        
        # print(test_results)
        
        # save LAT_Classifier and tokenizer as pth
        if data_set_params['classification_task']=='binary':
            best_model_path = os.path.join(save_path,'best_model_roc_auc.ckpt')
        elif data_set_params['classification_task']=='multi_class':
            best_model_path = os.path.join(save_path,'best_model_kappa.ckpt')
            
        pl_classifier = Pl_LAT_Classification_Freq_VQVAE.load_from_checkpoint(best_model_path, 
                                                        args = args, 
                                                        tokenizer_params = tokenizer_params, 
                                                        data_set_params = data_set_params, 
                                                        training_params = training_params, 
                                                        save_path = save_path)
        
        torch.save(pl_classifier.lat_classifier.state_dict(),os.path.join(save_path,'lat_classifier_best_model.pth'))
        torch.save(pl_classifier.tokenizer.state_dict(),os.path.join(save_path,'tokenizer_best_model.pth'))
        
        # save the last model as pth
        last_model_path = os.path.join(save_path,'last.ckpt')
        pl_classifier = Pl_LAT_Classification_Freq_VQVAE.load_from_checkpoint(last_model_path, 
                                                        args = args, 
                                                        tokenizer_params = tokenizer_params, 
                                                        data_set_params = data_set_params, 
                                                        training_params = training_params, 
                                                        save_path = save_path)
        
        torch.save(pl_classifier.lat_classifier.state_dict(),os.path.join(save_path,'lat_classifier_last_model.pth'))
        torch.save(pl_classifier.tokenizer.state_dict(),os.path.join(save_path,'tokenizer_last_model.pth'))
        
        print('Models saved')
    

# TUAB
    # stft window 4s hoplen 0.5s #last model
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '0'
    # stft window 4s hoplen 0.5s - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '3' --restart_embedding
    
    # multitaper window 4s hoplen 0.5s #last model
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'multitaper' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_multitaper_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1'
    # multitaper window 4s hoplen 0.5s - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'multitaper' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_multitaper_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '4' --restart_embedding
    
    
    # temporal window 4s hoplen 0.5s #last model
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_None_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '2'
    # temporal window 4s hoplen 0.5s - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUAB_None_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '5' --restart_embedding

    # stft window 2s hoplen 0.5s #last model - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/window_len 512TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1' --restart_embedding
    # stft window 1s hoplen 0.5s #last model - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/window_len 256TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1' --restart_embedding
    
    # stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/window_len 256TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1' --restart_embedding
    
# SHHS
    # stft window 4s hoplen 0.5s - restart embedding  # best model (because recon loss diverged at the end)
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'SHHS' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/SHHS_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_best_model.pth'  --restart_embedding --gpu '3'
    
    

# TUEV
    # stft window 4s hoplen 0.5s  # last model
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --restart_embedding --gpu '0'
    # stft window 2s hoplen 0.5s #last model - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/window_len_256TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --restart_embedding --gpu '1'
    
    # stft window 1s hoplen 0.5s  # last model # Full Frequency
    
# Trained on SHHS fine-tuned on TUAB # SHHS best model (because recon loss diverged at the end)
    # stft window 4s hoplen 0.5s - restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/SHHS_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_best_model.pth' --gpu '1' --restart_embedding
    
    
    
###### Full Freq experiments window 1s hoplen 0.5s
# TUAB  stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_256TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1' --restart_embedding
# TUEV stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_256TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '1' --restart_embedding
# TUEV stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '4' --restart_embedding
# TUEV stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200  filtered 75
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/filtered_75_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '6' --restart_embedding
# TUAB stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_200TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '7' --restart_embedding
# TUAB stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200  filtered 75
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/filtered_75_full_freq_window_len_200TUAB_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '3' --restart_embedding




# TUAB  multitaper window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUAB' --signal_transform 'multitaper' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_256TUAB_multitaper_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '4' --restart_embedding
# TUEV multitaper window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'multitaper' --resampling_rate 256 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/full_freq_window_len_256TUEV_multitaper_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '5' --restart_embedding


#### Frequecy Importance
# TUEV powerweighting stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/power_weighted_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '6' --restart_embedding

# TUEV temporal_separate stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/temporal_separate_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '4' --restart_embedding

# TUEV without positional encoding + temporal_separate stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/without_pe_temporal_separate_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '3' --restart_embedding

# TUEV freq embedding + temporal_separate stft window 1s hoplen 0.5s  # last model # Full Frequency # restart embedding -resampling rate 200
    # python freq_vqvae_tokens_multi_class_classification.py --dataset_name 'TUEV' --signal_transform 'stft' --resampling_rate 200 --vqvae_pretrained_path '/home/jp65/Biosignals_Research/EEG_BPE_Experiments/spectrum_vqvae_experiments/freq_embed_temporal_separate_MSE_full_freq_window_len_200TUEV_stft_FREQ_VQVAE_pretraining_codebook_size_4096_emb_size_64beta0.2/vqvae_last_model.pth' --gpu '3' --restart_embedding
   