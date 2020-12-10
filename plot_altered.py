import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
from pathlib import Path

def read_audio_from_filename(filename):
  audio, sr = librosa.load(filename)
  ftd = librosa.stft(audio)
  D = librosa.power_to_db(np.abs(ftd)**2)
  newAudio = librosa.feature.melspectrogram(y=audio, sr=sr, S=D)
  return ftd, sr, newAudio

def convert_data(filename):
  spect, sr, mel = read_audio_from_filename(filename)
  return spect, sr, mel

def write_spect(spect, target, sr):
  plt.figure()
  S_DB = librosa.power_to_db(np.abs(spect)**2, ref=np.max)
  librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+2.0f dB')
  plt.savefig(target)
  plt.close()

#filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav.wav'
#filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0aed2359-7d66-5da2-f041-8fb5d78b61c1.wav.wav'
#filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-5_0a8d62bd-4d37-522d-fc71-75fab6e20a7b.wav.wav'
#target = '/mnt/c/Users/Maia/Downloads/here.png'

#conv, sr, spect = convert_data(filename)
#write_spect(conv, "testconv.png")
#write_spect(np.sqrt(np.real(conv)**2 + np.imag(conv)**2), "testconv-norm.png")

wav_dir = '/mnt/c/Users/Maia/Downloads/spects/modelB/dlresults/'
target_dir =  '/mnt/c/Users/Maia/Downloads/spects/modelB/'
num_classes = 8

files = os.listdir(wav_dir)
for i in range(num_classes):
  for j in range(num_classes):
    target = str(i) + 'to' + str(j) + '.png'
    inst = files[i*(num_classes + 1) + j + 1]
    ftd, sr, _ = convert_data(wav_dir + inst)
    conv = np.sqrt(np.real(ftd)**2 + np.imag(ftd)**2)
    write_spect(conv, target_dir+target, sr)

