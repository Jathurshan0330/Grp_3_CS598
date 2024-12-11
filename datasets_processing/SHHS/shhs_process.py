import os
import mne
import numpy as np
from multiprocessing import Process
import xml.etree.ElementTree as ET
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

np.random.seed(5)
def pretext_train_test(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec):
    np.random.seed(5)
    # get all data indices
    all_index = sorted([int(path[6:12]) - 200000 for path in os.listdir(root_folder + 'shhs1')])
    
    # split into 
    train_index = np.random.choice(list(set(all_index)), int(len(all_index) * 0.7), replace=False)
    # split last 100 into validation
    val_index = np.random.choice(list(set(train_index)),100, replace=False)
    train_index = list(set(train_index) - set(val_index))

    # rest 30% is test
    test_index = list(set(all_index) - set(train_index)- set(val_index))
    
    ## assert no overlap
    assert len(set(train_index).intersection(set(val_index))) == 0
    assert len(set(train_index).intersection(set(test_index))) == 0
    assert len(set(val_index).intersection(set(test_index))) == 0

    
    print ('start train process!')
    sample_process(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec, 'train', train_index)
    print ()
    
    print ('start val process!')
    sample_process(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec, 'val', val_index)
    print ()
    
    print ('start test process!')
    sample_process(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec, 'test', test_index)
    print ()


def sample_process(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec, train_test_val, index):
    # process each EEG sample: further split the samples into window sizes and using multiprocess
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            # load the signal "X" part
            data = mne.io.read_raw_edf(root_folder + 'shhs1/' + 'shhs1-' + str(200000 + j) + '.edf',verbose=False)
            X = data.get_data()
            
            # some EEG signals have missing channels, we treat them separately
            if X.shape[0] == 16:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
            elif X.shape[0] == 15:
                X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
            X = X[[2,3,4,5,6,7], :] ## Extract [2,3,4,5,6,7] --> ['EEG', 'ECG', 'EMG', 'EOG-L', 'EOG-R', 'EEG]


            # load the label "Y" part
            with open(root_label_folder + 'shhs1-' + str(200000 + j) + '-profusion.xml', 'r') as infile:
                text = infile.read()
                root = ET.fromstring(text)
                y = [i.text for i in root.find('SleepStages').findall('SleepStage')]

            # slice the EEG signals into non-overlapping windows, window size = sampling rate per second * second time = 125 * windowsize
            for slice_index in range(X.shape[1] // (125 * epoch_sec)):
                path = save_processed_folder + 'processed_all/{}/'.format(train_test_val) + 'shhs1-' + str(200000 + j) + '-' + str(slice_index) + '.pkl'
                if os.path.exists(path):
                    continue
                if slice_index >= len(y):
                    break
                
                y_test = int(y[slice_index])
                if y_test > 5:
                    continue
                pickle.dump({'X': X[:, slice_index * 125 * epoch_sec: (slice_index+1) * 125 * epoch_sec], \
                    'y': int(y[slice_index])}, open(path, 'wb'))


"""
SHHS dataset is downloaded from https://sleepdata.org/datasets/shhs
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=30, help="How many processes to use")
    parser.add_argument('--signals_path', type=str, default="", help='Path to the data main directory')
    args = parser.parse_args()

    # print(args.signals_path)
    root_folder = os.path.join(args.signals_path,"polysomnography/edfs/")#"/srv/local/data/SHHS/polysomnography/edfs/" #for serv01 serv04 and serv05 -> "/srv/local/data/SHHS/" || for serv03 -> "/srv/local/data/SHHS/polysomnography/edfs/"
    # print(root_folder)
    root_label_folder = os.path.join(args.signals_path,"label/")#'/srv/local/data/SHHS/label/'#"/srv/local/data/SHHS_processed/label/"
    save_processed_folder = args.signals_path#'/srv/local/data/SHHS/'#"/srv/local/data/SHHS_processed/"
    if not os.path.exists(f'{save_processed_folder}/processed_all/'):
        os.makedirs(f'{save_processed_folder}/processed_all/train')
        os.makedirs(f'{save_processed_folder}/processed_all/val')
        os.makedirs(f'{save_processed_folder}/processed_all/test')

    

    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(root_folder,root_label_folder,save_processed_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
        
        
# python datasets/shhs_process.py --windowsize 30 --multiprocess 30

    print("All Done!")
    #compare the number of files in the original folder and the processed folder based on the first part of the name
    print("Original Folder: ", len(os.listdir(root_folder + 'shhs1')))
    print("Processed Folder: ", len(os.listdir(save_processed_folder + 'processed_all/train')))
    print("Processed Folder: ", len(os.listdir(save_processed_folder + 'processed_all/val')))
    print("Processed Folder: ", len(os.listdir(save_processed_folder + 'processed_all/test')))
    
    
    