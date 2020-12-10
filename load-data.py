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
x = torch.empty(size=(numFiles, 480, 640, 3))
y = torch.empty(size=(numFiles, 1))
for filename in files:
  label = int(filename[9])
  if processed % 1000 == 0: print(processed, " files processed! ")
  im = imageio.imread(spec_dir + filename)[:,:,0:3]
  x[processed] = torch.from_numpy(im/255)
  y[processed] = label
  processed += 1

print("done...")
m = {'x': x, 'y': y}
torch.save(m, 'tr-data.pt')

"""
with open('tr-data.pkl', 'wb') as f: 
  print("dumping y")
  pickle.dump(y, f)
  print("dumping x")
  pickle.dump(x, f)
  print("dump done")"""

