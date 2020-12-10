import os
import numpy as np
import imageio
from pathlib import Path
import pickle
import torch


spec_dir =  'tr-spectrograms/'
processed = 0
files = os.listdir(spec_dir)
numFiles = len(files)
x = []
y = []
for filename in files:
  label = int(filename[9])
  if processed % 1000 == 0: print(processed, " files processed! ")
  im = imageio.imread(spec_dir + filename)[:,:,0:3]
  x.append(np.array(im/255))
  y.append(label)
  processed += 1

print("done...", len(x))
with open('tr-data.pkl', 'wb') as f:
  pickle.dump([x,y], f)

"""
with open('tr-data.pkl', 'wb') as f: 
  print("dumping y")
  pickle.dump(y, f)
  print("dumping x")
  pickle.dump(x, f)
  print("dump done")"""

