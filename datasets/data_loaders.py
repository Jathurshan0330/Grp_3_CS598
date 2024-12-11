import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


from utils.utils import seed_everything

from scipy.signal import resample
from scipy import signal

from .multitaper_spectrogram import multitaper_spectrogram_multiple,multitaper_spectrogram_torch


def get_dataloaders(data_dir,dataset_name, train_val_test, batch_size, channels = [0],num_workers=8,
                    resampling_rate = 256, signal_transform=None,is_bpe=False):
    seed_everything(5)
    
    if train_val_test == 'train':
        shuffle = True
    else:
        shuffle = False
    
    if is_bpe:
        shuffle = False
    
    if dataset_name == 'SHHS':
        
        t_dataset = SHHS_Dataset(data_dir=data_dir, 
                                    train_val_test=train_val_test, 
                                    channels=channels,
                                    resampling_rate = resampling_rate,
                                    signal_transform=signal_transform
                                    )
        # X: (batch, channels, timesamples)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle, 
                                               num_workers=num_workers)
    if dataset_name == 'TUAB':
        
        t_dataset = TUAB_Dataset(data_dir=data_dir, 
                                    train_val_test=train_val_test, 
                                    resampling_rate = resampling_rate,
                                    signal_transform=signal_transform
                                    )
        # X: (batch, channels, timesamples)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle, 
                                               num_workers=num_workers)
    if dataset_name == 'TUEV':
            
        t_dataset = TUEV_Dataset(data_dir=data_dir, 
                                    train_val_test=train_val_test, 
                                    resampling_rate = resampling_rate,
                                    signal_transform=signal_transform
                                    )
        # X: (batch, channels, timesamples)
        t_loader = torch.utils.data.DataLoader(t_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers)
    
    return t_loader


# def get_dataloaders_bpe(data_dir,dataset_name, train_val_test, batch_size, channels = [0],num_workers=8,
#                         resampling_rate = 256, signal_transform=None):
#     seed_everything(5)
    
   
#     shuffle = False
    
    
#     if dataset_name == 'SHHS':
        
#         t_dataset = SHHS_Dataset(data_dir=data_dir, 
#                                     train_val_test=train_val_test, 
#                                     channels=channels,
#                                     resampling_rate=resampling_rate,
#                                     signal_transform=signal_transform
#                                     )
#         # X: (batch, channels, timesamples)
#         t_loader = torch.utils.data.DataLoader(t_dataset, 
#                                                batch_size=batch_size, 
#                                                shuffle=shuffle, 
#                                                num_workers=num_workers)
    
#     return t_loader

class SHHS_Dataset(Dataset):
    def __init__(self, data_dir, train_val_test, channels=[0], resampling_rate = 256, signal_transform=None):
        '''
        Args:
            data_dir: str, path to the directory containing the data files
            train_val_test: str, 'train', 'val', or 'test'
            channels: list of int, list of channel indices to use
            resampling_rate: int, resampling rate, default is 256
            signal_transform: str, 'stft', 'multitaper', default is None
        '''
        
        # order of the channels: ['EEG', 'ECG', 'EMG', 'EOG-L', 'EOG-R', 'EEG]
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.channels = channels
        
        self.signal_len = 30
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings: ", len(self.data_files))

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # read pickle
        signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = []#signal_data['X'][self.channels]
        for i in self.channels:
            if i ==3:
                X.append(signal_data['X'][i]-signal_data['X'][i+1])
            elif i == 4:
                continue
            else:
                X.append(signal_data['X'][i])
        X = np.array(X)
        
        # Mapping N3 and N4 to N3
        if signal_data['y'] == 4:
            labels = 3
        #changing label of REM to 4
        elif signal_data['y'] == 5:
            labels = 4
        else:
            labels = signal_data['y']
        
        # resample
        X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-10)
        
        
    
        if self.signal_transform == 'stft':
            window_len = self.resampling_rate#*4 # 4s
            hop_len = self.resampling_rate//2 #0.5 #self.resampling_rate//4 #0.25s
            X_fft = []
            for i in range(X.shape[0]):
                f, _, Zxx = signal.stft(X[i], 
                            fs=self.resampling_rate, 
                            nperseg=window_len, 
                            noverlap=window_len-hop_len,
                            return_onesided=True,boundary=None,
                            padded = False, scaling='spectrum')
                X_fft.append(np.abs(Zxx))
            X = np.stack(X_fft, axis=0)
            # f_ind = np.where(f == 80)[0][0]
            # X = X[ :, :f_ind, :]
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)

        elif self.signal_transform == 'multitaper':
            window_duration = 1 # 4s
            hop_duration = 0.5 #0.25
            
            # X, _,f = multitaper_spectrogram_multiple(
            #     X,
            #     fs=self.resampling_rate,
            #     time_bandwidth=2,
            #     window_params = [window_duration, hop_duration],
            #     multiprocess = True,
            #     verbose = False,plot_on = False)
            X, _, f = multitaper_spectrogram_torch(X, 
                                                    fs = self.resampling_rate,
                                                    time_bandwidth=2,
                                                    window_params= [window_duration, hop_duration],
                                                    device='cpu')
            X = X.permute(0,2,1)
            # f = f.cpu().numpy()
            # f_ind = np.where(f == 80)[0][0]
            # # f_ind = torch.where(f == 80)[0].item()
            # X = X[ :, :f_ind, :] # for easier processing
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)
            
            
            
        
        X = torch.FloatTensor(X)
        
        
        
        # X: (batch, channels, timesamples)
        return X, labels
    
    
    
class TUAB_Dataset(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        '''
        Args:
            data_dir: str, path to the directory containing the data files
            train_val_test: str, 'train', 'val', or 'test'
            channels: list of int, list of channel indices to use
            resampling_rate: int, resampling rate, default is 256
            signal_transform: str, 'stft', 'multitaper', default is None
        '''
        
        # order of the channels: ['EEG', 'ECG', 'EMG', 'EOG-L', 'EOG-R', 'EEG]
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        
        self.signal_len = 10
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings: ", len(self.data_files))

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # read pickle
        signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = signal_data["X"]
        
        X = np.array(X)
        labels = signal_data["y"]
        
         # resample
        X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-10)
        
        
    
        if self.signal_transform == 'stft':
            window_len = self.resampling_rate # 4s
            hop_len = self.resampling_rate//2 #0.5 #self.resampling_rate//4 #0.25s
            X_fft = []
            for i in range(X.shape[0]):
                f, _, Zxx = signal.stft(X[i], 
                            fs=self.resampling_rate, 
                            nperseg=window_len, 
                            noverlap=window_len-hop_len,
                            return_onesided=True,boundary=None,
                            padded = False, scaling='spectrum')
                X_fft.append(np.abs(Zxx))
            X = np.stack(X_fft, axis=0)
            # f_ind = np.where(f == 80)[0][0]
            # X = X[ :, :f_ind, :]
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)

        elif self.signal_transform == 'multitaper':
            window_duration = 1 # 4s
            hop_duration = 0.5 #0.25
            
            # X, _,f = multitaper_spectrogram_multiple(
            #     X,
            #     fs=self.resampling_rate,
            #     time_bandwidth=2,
            #     window_params = [window_duration, hop_duration],
            #     multiprocess = True,
            #     verbose = False,plot_on = False)
            X, _, f = multitaper_spectrogram_torch(X, 
                                                    fs = self.resampling_rate,
                                                    time_bandwidth=2,
                                                    window_params= [window_duration, hop_duration],
                                                    device='cpu')
            X = X.permute(0,2,1)
            # f = f.cpu().numpy()
            # f_ind = np.where(f == 80)[0][0]
            # # f_ind = torch.where(f == 80)[0].item()
            # X = X[ :, :f_ind, :] # for easier processing
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)
            
            
            
        
        X = torch.FloatTensor(X)
        
        
        
        # X: (batch, channels, timesamples)
        return X, labels
    


class TUEV_Dataset(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        '''
        Args:
            data_dir: str, path to the directory containing the data files
            train_val_test: str, 'train', 'val', or 'test'
            channels: list of int, list of channel indices to use
            resampling_rate: int, resampling rate, default is 256
            signal_transform: str, 'stft', 'multitaper', default is None
        '''
        
        # order of the channels: ['EEG', 'ECG', 'EMG', 'EOG-L', 'EOG-R', 'EEG]
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        
        self.signal_len = 5
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        if train_val_test == 'train':
            train_files = os.listdir(os.path.join(data_dir, "processed_train"))
            train_sub = list(set([f.split("_")[0] for f in train_files]))
            val_sub = np.random.choice(train_sub, size=int(len(train_sub)*0.1), replace=False)
            train_sub = list(set(train_sub) - set(val_sub))
            self.data_files = [f for f in train_files if f.split("_")[0] in train_sub]
            
        elif train_val_test == 'val':
            train_files = os.listdir(os.path.join(data_dir, "processed_train"))
            train_sub = list(set([f.split("_")[0] for f in train_files]))
            val_sub = np.random.choice(train_sub, size=int(len(train_sub)*0.1), replace=False)
            self.data_files = [f for f in train_files if f.split("_")[0] in val_sub]
        elif train_val_test == 'test':
            self.data_files = os.listdir(os.path.join(data_dir, "processed_eval"))
        # self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings: ", len(self.data_files))

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # read pickle
        if self.train_val_test == 'test':
            signal_data = pickle.load(open(os.path.join(self.data_dir,"processed_eval",self.data_files[idx]), 'rb'))
        else:
            signal_data = pickle.load(open(os.path.join(self.data_dir,"processed_train",self.data_files[idx]), 'rb'))
        # signal_data = pickle.load(open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb'))
        X = signal_data["signal"]
        
        X = np.array(X)
        labels = int(signal_data["label"][0] - 1)
        
        # resample
        X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-10)
        
        
    
        if self.signal_transform == 'stft':
            window_len = self.resampling_rate # 4s
            hop_len = self.resampling_rate//2 #0.5 #self.resampling_rate//4 #0.25s
            X_fft = []
            for i in range(X.shape[0]):
                f, _, Zxx = signal.stft(X[i], 
                            fs=self.resampling_rate, 
                            nperseg=window_len, 
                            noverlap=window_len-hop_len,
                            return_onesided=True,boundary=None,
                            padded = False, scaling='spectrum')
                X_fft.append(np.abs(Zxx))
            X = np.stack(X_fft, axis=0)
            # f_ind = np.where(f == 80)[0][0]
            # X = X[ :, :f_ind, :]
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)

        elif self.signal_transform == 'multitaper':
            window_duration = 1 # 4s
            hop_duration = 0.5 #0.25
            
            # X, _,f = multitaper_spectrogram_multiple(
            #     X,
            #     fs=self.resampling_rate,
            #     time_bandwidth=2,
            #     window_params = [window_duration, hop_duration],
            #     multiprocess = True,
            #     verbose = False,plot_on = False)
            X, _, f = multitaper_spectrogram_torch(X, 
                                                    fs = self.resampling_rate,
                                                    time_bandwidth=2,
                                                    window_params= [window_duration, hop_duration],
                                                    device='cpu')
            X = X.permute(0,2,1)
            # f = f.cpu().numpy()
            # f_ind = np.where(f == 80)[0][0]
            # # f_ind = torch.where(f == 80)[0].item()
            # X = X[ :, :f_ind, :] # for easier processing
            # X: (batch, channels, freqs, timesamples) --> for 256Hz -> (batch, channels, 513, 104)
            
            
            
        
        X = torch.FloatTensor(X)
        
        
        
        # X: (batch, channels, timesamples)
        return X, labels



