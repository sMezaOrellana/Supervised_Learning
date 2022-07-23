import torch
import numpy as np
import os
import librosa
import random
from torch.utils.data import DataLoader, Dataset

def load_files(dirname):
    files_paths = np.empty(0, dtype=np.str_)
    for filename in os.listdir(dirname):
        f = os.path.join(dirname, filename)
        # checking if it is a file
        if os.path.isfile(f):
            files_paths = np.append(files_paths,f)
    return files_paths


class Dataset_M(Dataset):
    def __init__(self, XY):
        self.X = XY[0]
        self.y = XY[1]
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


class Dataset_Load:
    def __init__(self):
        self.maj_files = os.path.join(os.getcwd(),'Data', 'allmaj')
        self.min_files = os.path.join(os.getcwd(),'Data', 'allmin')
        self.maj_files_paths = load_files(self.maj_files)
        self.min_files_paths = load_files(self.min_files)
        self.sr_n = 2000
        self.X, self.Y = self.load_files_to_tensor()
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(self.X,self.Y)
    
    def complement_indices(self,A, size):
        B = []
        for i in range(0, size):
            if i not in A:
                B.append(i)
        print(len(B))
        return B    
    
    def train_test_split(self,X, Y, test_size=0.33):
        num_test_set = int( 3012 * test_size) #165
        random_test_indices = torch.tensor(random.sample(range(0, 3012), num_test_set))
        random_train_indices = torch.tensor(self.complement_indices(random_test_indices, 3012))
        
        X_train = X[random_train_indices]
        X_test = X[random_test_indices]
        y_train = Y[random_train_indices]
        y_test = Y[random_test_indices]
        
        return X_train, X_test, y_train, y_test

    def load_wav(self, filename):
        sig, sr = librosa.load(filename, mono=True)
        sig = librosa.resample(sig, orig_sr=sr, target_sr = self.sr_n)
        sig = torch.from_numpy(sig)
        return sig
        
    def preprocess(self, file):
        #print(file)
        time_domain = self.load_wav(file)
        time_domain = time_domain[:4000]
        padding = torch.zeros(4000 - time_domain.shape[0])
        time_domain = torch.cat([padding, time_domain])
        return time_domain
    
    def spectogram(self, signal):
        return torch.stft(signal, n_fft=2048, hop_length = 120,return_complex=True)
    
    def load_files_to_tensor(self):
        X = torch.zeros([3012, 1 , 1025 , 34], dtype=torch.float32)
        Y = torch.cat([torch.zeros(1500), torch.ones(1512)]).type(torch.LongTensor)
        
        i = 0
        for file in self.maj_files_paths:
            signal = self.preprocess(file)
            specgram = self.spectogram(signal)
            specgram = torch.abs(specgram)
            specgram = torch.reshape(specgram, [1, specgram.shape[0], specgram.shape[1]])
            
            X[i] = specgram
            i+=1

        for file in self.min_files_paths:
            signal = self.preprocess(file)
            specgram = self.spectogram(signal)
            specgram = torch.abs(specgram)
            specgram = torch.reshape(specgram, [1, specgram.shape[0], specgram.shape[1]])
            
            X[i] = specgram
            i+=1
        
        i -= 1
        
        return (X, Y)

        
    def get_test(self):
        return self.X_test, self.y_test
    
    def get_training(self):
        return self.X_train, self.y_train
    
if __name__ == '__main__':
    dataset = Dataset_Load()