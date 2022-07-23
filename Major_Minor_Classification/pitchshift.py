#!/usr/bin/python
import fileinput
import sys
import librosa
import warnings
import soundfile as sf
warnings.filterwarnings('ignore')

def load_wav(filename):
    sig, sr = librosa.load(filename, mono=True)
    return sig, sr

path = '/home/makhmalbaf/Desktop/ML/Supervised-Learning/Major_Minor_Classification/Data/minor/'

for line in fileinput.input():
    if len(line) == 0:
        continue
        
    fpath = path + line
    sys.stdout.write(fpath)
    
    fpath = fpath.strip('\n')
    

    sig, sr = load_wav(fpath)
    steps = [-3,-4,4,5,7]
    for i in steps:
        sig_hat = librosa.effects.pitch_shift(sig, sr=sr, n_steps=i)
        lstrip = line.strip('\n').replace(".wav","")
        outfpath = '/home/makhmalbaf/Desktop/ML/Supervised-Learning/Major_Minor_Classification/Data/augmin/' + lstrip + 'shift_' + str(i) + '.wav'
 
        sf.write(outfpath, sig_hat, sr)
            