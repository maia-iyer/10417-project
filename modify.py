import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import torch

def read_audio_from_filename(filename):
  audio, sr = librosa.load(filename)
  ftd = librosa.stft(audio)
  D = librosa.power_to_db(np.abs(ftd)**2)
  newAudio = librosa.feature.melspectrogram(y=audio, sr=sr, S=D)
  return ftd, sr, newAudio

def convert_data(filename):
  spect, sr, mel = read_audio_from_filename(filename)
  return spect, sr, mel

filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav.wav'
#filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0aed2359-7d66-5da2-f041-8fb5d78b61c1.wav.wav'
#filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-5_0a8d62bd-4d37-522d-fc71-75fab6e20a7b.wav.wav'
target = '/mnt/c/Users/Maia/Downloads/here.wav'

audio, sr = librosa.load(filename)
print(np.shape(audio), np.shape(sr))
print(np.min(audio), np.max(audio))
ftd = librosa.stft(audio)
noise = np.random.uniform(-0.001, 0.001, 65536)
new = audio + noise
wavfile.write(target, sr, np.array(new*32767, dtype=np.int16))
#conv, sr, spect = convert_data(filename)
#write_spect(conv, "testconv.png")
#write_spect(np.sqrt(np.real(conv)**2 + np.imag(conv)**2), "testconv-norm.png")

"""wav_dir = ['va-data/', 'te-data/']
target = ['va-data.pt', 'te-data.pt']
ints = [27, 21]
for i in range(2):
  numFiles = 0
  files = os.listdir(wav_dir[i])
  x = torch.empty(size=(len(files), 1025, 129))
  y = torch.empty(size=(len(files), 1))
  for filename in files:
    if numFiles % 200 == 0: print(numFiles, " files processed out of ", len(files), "! ")
    conv, sr, spect = convert_data(wav_dir[i] + filename)
    x[numFiles] = torch.from_numpy(conv)
    y[numFiles] = int(filename[ints[i]])      
    numFiles += 1

  m = {'x': x, 'y': y}
  torch.save(m, target[i])"""
