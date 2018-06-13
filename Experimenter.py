from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import datasets, svm
import numpy as np

class DataDebug():
  def __init__(self, n_splits=5, n_repeats=3, random_state=None):
    self.rkf = RepeatedStratifiedKFold(n_splits, n_repeats, random_state)
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

class Experimenter():
  def __init__(self, data=DataDebug(), methods={"SVM": svm.SVC()}):
    self.data=data
    self.methods=methods
  def perform(self):
    targets = {}
    for train, test in self.data.split():
      if "actual" not in targets:
        targets["actual"] = []
      targets["actual"].append(test["target"])
      for clfname, clf in self.methods.items():
        if clfname not in targets:
          targets[clfname] = []
        clf.fit(train["data"],train["target"])
        targets[clfname].append(clf.predict(test["data"]))
    return targets


exp = Experimenter()
targets = exp.perform()
#print(targets)
#for i in range(len(targets["actual"])):
#  res = np.array(targets["actual"][i])==np.array(targets["SVM"][i])
#  print(sum(res)/len(res))

