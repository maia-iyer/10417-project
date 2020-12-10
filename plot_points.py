import matplotlib.pyplot as plt
import numpy as np

x = [0, 10, 20, 30, 40, 50, 60, 70]
lBase = [1.707, 1.462, 2.022, 1.821, 1.874, 1.950, 2.098, 2.855]
aBase = [0.434, 0.695, 0.701, 0.754, 0.772, 0.788, 0.773, 0.758]
lVert = [1.704, 1.374, 1.347, 1.457, 1.534, 1.510, 1.787, 1.841]
aVert = [0.382, 0.524, 0.663, 0.695, 0.810, 0.730, 0.727, 0.741]
lCMod = []
aCMod = []

x = [0, 10, 20, 30, 40, 50]
lBase = [1.707, 1.462, 2.022, 1.821, 1.874, 1.950]
aBase = [0.434, 0.695, 0.701, 0.754, 0.772, 0.788]
lVert = [1.704, 1.374, 1.347, 1.457, 1.534, 1.510]
aVert = [0.382, 0.524, 0.663, 0.695, 0.810, 0.730]
lCMod = [1.978, 1.465, 1.242, 1.290, 1.312, 1.464]
aCMod = [0.215, 0.525, 0.690, 0.701, 0.744, 0.725]



plt.plot(x, lBase, label = "Baseline Model (Model A) Loss over Time")
plt.plot(x, lVert, label = "Vertical Model (Model B) Loss over Time")
plt.plot(x, lCMod, label = "Middle Model (Model C) Loss over Time")
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')

plt.title('Test Loss Curve')
plt.legend()
plt.savefig('/mnt/c/Users/Maia/Downloads/spects/lossCurve.png')

plt.figure()

plt.plot(x, aBase, label = "Baseline Model (Model A) Accuracy over Time")
plt.plot(x, aVert, label = "Vertical Model (Model B) Accuracy over Time")
plt.plot(x, aCMod, label = "Middle Model (Model C) Accuracy over Time")
plt.xlabel('Number of Epochs')
plt.ylabel('Test Accuracy')

plt.title('Test Accuracy Curve')
plt.legend()
plt.savefig('/mnt/c/Users/Maia/Downloads/spects/accCurve.png')


