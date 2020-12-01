import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display

filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav.wav'
filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0aed2359-7d66-5da2-f041-8fb5d78b61c1.wav.wav'
filename = '/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-5_0a8d62bd-4d37-522d-fc71-75fab6e20a7b.wav.wav'
y, sr = librosa.load(filename)
song, _ = librosa.effects.trim(y)

n_fft = 2048
hop_length = 512
S = librosa.feature.melspectrogram(song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=512)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB')

plt.savefig('/mnt/c/Users/Maia/Downloads/here.png')
