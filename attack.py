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
  return audio, sr


model = torch.load('models/test1/model50.pt')
validation_dir = 'va-data/'
target_dir = 'attacked-audio/'
validation_files = os.listdir(validation_dir)
logfile = 'logger.txt'
validation_indices = [0, 329, 824, 1108, 1341, 2363, 2392, 2464, 3494]
numClasses = 8

def model_output_from_audio(audio):
  audio = audio.numpy()
  spect = librosa.stft(audio)
  spect = torch.from_numpy(spect)
  inputs = np.sqrt(np.real(spect)**2 + np.imag(spect)**2)
  inputs = torch.reshape(inputs, (1, 1, 1025, 129))
  return model(inputs.float())

def ea_attack(audio, target_class, epsilon):
  audio = torch.from_numpy(audio)
  # attack hyperparameters
  N = 10
  num_samples = 65536
  rho_min = 0.1
  beta_min = 0.15
  num_generations = 50

  rho = 0.5
  beta = 0.4
  num_plateaus = 0

  last_fitness = -10000
  adversarial_audio = audio
  
  # create population
  population = [torch.zeros(num_samples)]
  for n in range(N-1):
    rand_audio = torch.empty(num_samples).uniform_(-epsilon, epsilon)
    rand_audio = torch.clamp(rand_audio + audio, 0, 1) - audio
    #rand_audio = np.random.uniform(-epsilon, epsilon, (num_samples,))
    #rand_audio = np.clip(rand_audio + audio, -1, 1) - audio
    population.append(rand_audio)

  # iterate num generations
  for i in range(num_generations):
    if i%50 == 0 and i != 0: print("Generation ", i, " out of 100") 
    # compute fitness scores
    fitness_scores = []
    logs = [model_output_from_audio(noise + audio) for noise in population]
    summedLogs = [torch.sum(logs[j]) for j in range(len(population))]
    fitness_scores = [torch.clamp(2 * logs[j][0][target_class] - summedLogs[j], -100, 1000) for j in range(len(logs))]

    answers = [model_output_from_audio(audio + member) for member in population]
    #print(answers)

    # find elite member
    eliteIndex = np.argmax(fitness_scores)
    #print(eliteIndex)
    eliteMember = population[eliteIndex]
    eliteScore = fitness_scores[eliteIndex]
    new_population = [eliteMember]
    if torch.argmax(model_output_from_audio(eliteMember + audio)) == target_class:
      print("finished at generation ", i)
      return (eliteMember + audio).numpy()
    else: adversarial_audio = eliteMember + audio
    # if plateau, increment plateau
    if eliteScore <= last_fitness + 1e-5:
      num_plateaus += 1
    last_fitness = eliteScore

    # get selection probabilities
    exps = np.exp(fitness_scores)
    #print(np.max(exps))
    exps = [exps[j].detach().item() for j in range(len(exps))]
    softmaxes = exps / np.linalg.norm(exps, ord = 1)

    # create new population
    for j in range(N-1):
      # sample parent1 and parent2
      drawInds = np.random.choice([j for j in range(len(population))], size=2, p=softmaxes, replace=True)
      parent_1 = population[drawInds[0]]
      parent_2 = population[drawInds[1]]
      # crossover and mutate
      p = fitness_scores[drawInds[0]] / (fitness_scores[drawInds[0]] + fitness_scores[drawInds[1]])
      sel = torch.randn(num_samples)
      child = torch.where(sel<p, parent_1, parent_2)

      if (np.random.uniform(0.0, 1.0, 1) < rho):
        noise = torch.empty(num_samples).uniform_(-epsilon*beta, epsilon*beta)
        child = child + noise
      
      # apply clipping
      child = torch.clamp(child + audio, -1, 1) - audio

      # add child
      new_population.append(child)
    
    # update params
    rho = max(rho_min, 0.5 * (0.9**num_plateaus))
    beta = max(beta_min, 0.4 * (0.9**num_plateaus))
    population = new_population
  return adversarial_audio.numpy()

#wavfile.write(target, sr, np.array(new*32767, dtype=np.int16))
#conv, sr, spect = convert_data(filename)
#write_spect(conv, "testconv.png")
#write_spect(np.sqrt(np.real(conv)**2 + np.imag(conv)**2), "testconv-norm.png")

for c in range(numClasses):
  numTestsPerClass = 1
  for t in range(numTestsPerClass):
    # get original audio
    index = torch.randint(validation_indices[c], validation_indices[c+1], (1,))
    fullname = validation_dir + validation_files[index]
    original_audio, sr = read_audio_from_filename(fullname)
    save_original_filename = target_dir + validation_files[index]
    newFilenames = [target_dir + validation_files[index][:-8] + 'to' + str(c1) + '.wav' for c1 in range(numClasses)]
    # save original audio
    wavfile.write(save_original_filename, sr, np.array(original_audio*32767, dtype = np.int16))
    # get attacked audio
    all_attacked_audio = []
    # print original and new confidence
    print("Testing for ", validation_files[index])
    print("Original logits: ", model_output_from_audio(torch.from_numpy(original_audio)))
    for c1 in range(numClasses):
      print("Attacking ", c1)
      new_audio = ea_attack(original_audio, c1, 0.001)
      all_attacked_audio.append(new_audio)
      # save attacked audio
      wavfile.write(newFilenames[c1], sr, np.array(new_audio * 32767, dtype = np.int16))
      print("Gets logits: ", model_output_from_audio(torch.from_numpy(all_attacked_audio[c1])))
   

