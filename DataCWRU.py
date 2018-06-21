import collections
from random import shuffle
import numpy as np
import scipy.io
#from FeatureExtraction import StatisticalTime

class DataCWRU():
  def __init__(self, feature_model=None, debug=False, n_repeat=1):
    self.n_repeat = n_repeat
    self.n_splits = 4
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
        train = self.__readmatfile([i for i,j in zip(self.idxs, boolean_idx) if j])
        test = self.__readmatfile([i for i,j in zip(self.idxs, boolean_idx) if not j])
        yield train, test

  def __readmatfile(self, indexes):
    fold = collections.defaultdict(list)
    for i in indexes:
      matfile = scipy.io.loadmat(self.datadir+self.files[i])
      for k in matfile:
        if k.endswith(self.target[i][0:2].upper()+'_time') or self.target[i]=='normal' and k.endswith('DE_time' if self.debug else '_time'):
          key = k
          begsig = 0
          while key in matfile and begsig+self.max_sample_size < len(matfile[key]):
            flat_list = [item for sublist in matfile[key][begsig:begsig+self.max_sample_size] for item in sublist]
            fold["data"].append(flat_list)
            begsig += self.max_sample_size  
            fold["target"].append(self.target[i])
    fold["data"] = np.array(fold["data"])
    #print(set(fold["target"]), fold["data"].shape)
    return fold

