import collections
from random import shuffle
import numpy as np
import scipy.io
#from FeatureExtraction import StatisticalTime

class DataCWRU():
  def __init__(self, feature_model=None, debug=False, n_repeat=1):
    self.n_splits = 4
    self.feature_model = feature_model
    self.n_repeat = n_repeat
    self.debug = debug
    self.max_sample_size = 4096# 2048# 1024# 8192# 
    self.fails = collections.defaultdict(list)
    self.datadir = 'datasets/cwru/'
    mode = 'failures.txt'
    if debug:
      mode = 'debug.txt'
    failures = open(self.datadir+mode)
    for line in failures:
      if line.startswith('#'):
        continue
      fail = line.split()
      if debug and not fail[0].startswith('normal'): #When it is True, it ignores the fault severity.
        f = fail[0].split('0')
      else:
        f = fail[0].split('_')
      #f = f[0].split('@')
      self.fails[f[0]].append(fail[1])
    failures.close()

  def split(self):
    for _ in range(self.n_repeat):
      self.files = []
      self.target = []
      for data,filemat in self.fails.items():
        shuffle(filemat)
        for f in filemat:
          self.files.append(f)
          self.target.append(data)
      idx = np.argsort(self.target)
      self.idxs = idx[np.argsort(np.mod(idx, self.n_splits))]
      end = 0
      n = len(self.target)//self.n_splits
      for i in range(self.n_splits):
        boolean_idx = [True]*len(self.target)
        start = end
        end = start + n
        if len(self.target[end:])%n > 0 or (end-start)*n<len(self.target[end:]):
          end = end + 1
        for j in range(start,end):
          boolean_idx[j] = False
        train = self.__readmatfiles([i for i,j in zip(self.idxs, boolean_idx) if j])
        test = self.__readmatfiles([i for i,j in zip(self.idxs, boolean_idx) if not j])
        yield train, test

  def __readmatfiles(self, indexes):
    fold = collections.defaultdict(list)
    for i in indexes:
      matfile = scipy.io.loadmat(self.datadir+self.files[i])
      for k in matfile:
        begsig = 0
        if self.feature_model == None:
          if k.endswith(self.target[i][0:2].upper()+'_time') or self.target[i]=='normal' and k.endswith('DE_time' if self.debug else '_time'):
            while k in matfile and begsig+self.max_sample_size < len(matfile[k]):
              flat_list = [item for sublist in matfile[k][begsig:begsig+self.max_sample_size] for item in sublist]
              fold["data"].append(flat_list)
              fold["target"].append(self.target[i])
              begsig += self.max_sample_size  
        else:
          if k.endswith('FE_time'):
            key = k.split('_')[0]
            while begsig+self.max_sample_size < len(matfile[key+'_DE_time']) and begsig+self.max_sample_size < len(matfile[key+'_FE_time']):
              desig = [item for sublist in matfile[key+'_DE_time'][begsig:begsig+self.max_sample_size] for item in sublist]
              #defeat = self.feature_model.transform(desig)
              fesig = [item for sublist in matfile[key+'_FE_time'][begsig:begsig+self.max_sample_size] for item in sublist]
              #fefeat = self.feature_model.transform(fesig)
              #fold["data"].append(np.append(defeat,fefeat))
              fold["data"].append(self.feature_model.transform([desig,fesig]))
              fold["target"].append(self.target[i])
              begsig += self.max_sample_size  
    fold["data"] = np.array(fold["data"])
    return fold

