import collections
import string

class DataCWRU():
  def __init__(self, n_splits=5, n_repeats=3, random_state=None):
    self.n_splits = n_splits
    self.n_repeats = n_repeats
    self.fails = collections.defaultdict(list)
    failures = open("datasets/cwru/failures.txt")
    for line in failures:
      fail = string.split(line)
      f = string.split(fail[0],'_')
      self.fails[f[0]].append(fail[1])
    print(self.fails)
  def split(self):
    train = {}
    test = {}
    for dataset in [datasets.load_iris()]:#, datasets.load_wine(), datasets.load_breast_cancer(),datasets.load_digits()]:
      for itr, ite in self.rkf.split(dataset["data"],dataset["target"]):
        train["data"] = dataset["data"][itr]
        train["target"] = dataset["target"][itr]
        test["data"] = dataset["data"][ite]
        test["target"] = dataset["target"][ite]
        yield train,test

x = DataCWRU()
print(x)
