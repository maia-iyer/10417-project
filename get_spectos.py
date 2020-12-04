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

def write_spect(spect, target):
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

wav_dir = ['tr-data/']
target_dir =  ['va-spectrograms/','te-spectrograms/']

for i in range(2):
  numFiles = 0
  for filename in os.listdir(wav_dir[i]):
    target_filename = target_dir[i] + filename[16:-8] + '.png'
    if not Path(target_filename).is_file():
      if numFiles % 200 == 0: print(numFiles, " files processed! ")
      conv, sr, spect = convert_data(wav_dir[i] + filename)
      
      #write_spect(conv, "test1.png")
      #write_spect(sqrt(Re(conv)**2 + Im(conv)**2), "test1norm.png") 
      numFiles += 1

