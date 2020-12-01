import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.io import wavfile
import numpy as np

sample_rate, samples = wavfile.read('/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0a282672-c22c-59ff-faaa-ff9eb73fc8e6.wav.wav')
#sample_rate, samples = wavfile.read('/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-0_0aed2359-7d66-5da2-f041-8fb5d78b61c1.wav.wav')
sample_rate, samples = wavfile.read('/mnt/c/Users/Maia/Downloads/a/Medley-solos-DB_test-5_0a8d62bd-4d37-522d-fc71-75fab6e20a7b.wav.wav')


frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
# spectrogram is row-wise frequencies and column wise time

print(sample_rate)
print(np.shape(spectrogram), np.max(spectrogram), np.min(spectrogram))
#plt.axes([0,0,3,0.5])
#plt.figure(figsize=(7,3))
times = times * 100

plt.pcolormesh(times, frequencies, np.log(spectrogram), shading='auto')
#plt.xlim(0,4)
#plt.ylim(0,22050)
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.savefig('/mnt/c/Users/Maia/Downloads/here.png')
