import collections
from random import shuffle
import numpy as np
import scipy.io

class DataCWRU():
  def __init__(self, n_repeats=3, random_state=None):
    self.n_splits = 4
    self.max_sample_size = 4096
    self.n_repeats = n_repeats
    self.fails = collections.defaultdict(list)
    self.datadir = 'datasets/cwru/'
    failures = open(self.datadir+'/failures.txt')
    for line in failures:
      fail = line.split()
      f = fail[0].split('_')
      f = f[0].split('@')
      self.fails[f[0]].append(fail[1])
    self.files = []
    self.target = []
    for data,filemat in self.fails.items():
      shuffle(filemat)
      for f in filemat:
        self.files.append(f)
        self.target.append(data)
    idx = np.argsort(self.target)
    idx2 = np.argsort(np.mod(idx, self.n_splits))
    self.idxs = idx[idx2]

  def split(self):
    y = self.target
    end = 0
    n = len(y)//self.n_splits
    for i in range(self.n_splits):
      train = collections.defaultdict(list)
      test = collections.defaultdict(list)
      boolean_idx = [True]*len(y)
      start = end
      end = start + n
      if len(y[end:])%n > 0 or (end-start)*n<len(y[end:]):
        end = end + 1
      for j in range(start,end):
        boolean_idx[j] = False
      train_index = [i for i,j in zip(self.idxs, boolean_idx) if j]
      test_index = [i for i,j in zip(self.idxs, boolean_idx) if not j]
      for i in train_index:
        matfile = scipy.io.loadmat(self.datadir+self.files[i])
        for k in matfile:
          if k.endswith('DE_time'):
            key = k
            break
        begsig = 0
        while begsig+self.max_sample_size < len(matfile[key]) :
          flat_list = [item for sublist in matfile[key][begsig:begsig+self.max_sample_size] for item in sublist]
          train["data"].append(flat_list)
          begsig += self.max_sample_size  
          train["target"].append(self.target[i])
      for i in test_index:
        matfile = scipy.io.loadmat(self.datadir+self.files[i])
        for k in matfile:
          if k.endswith('DE_time'):
            key = k
            break
        begsig = 0
        while begsig+self.max_sample_size < len(matfile[key]) :
          flat_list = [item for sublist in matfile[key][begsig:begsig+self.max_sample_size] for item in sublist]
          test["data"].append(flat_list)
          begsig += self.max_sample_size  
          test["target"].append(self.target[i])
      yield train, test


x = DataCWRU()
x.split()
