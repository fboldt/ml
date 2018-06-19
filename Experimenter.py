from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, metrics, svm
from sklearn.decomposition import PCA
import numpy as np
import DataCWRU as cwru
import ExtractionStatistical as stat
from sklearn.feature_selection import SelectKBest

class DataDivision():
  def __init__(self, n_splits=3, n_repeats=2, random_state=None, dataset=datasets.load_iris()):
    self.rkf = RepeatedStratifiedKFold(n_splits, n_repeats, random_state)
    self.dataset = dataset
  def split(self):
    train = {}
    test = {}
    for itr, ite in self.rkf.split(self.dataset["data"],self.dataset["target"]):
      train["data"] = self.dataset["data"][itr]
      train["target"] = self.dataset["target"][itr]
      test["data"] = self.dataset["data"][ite]
      test["target"] = self.dataset["target"][ite]
      yield train,test

class Experimenter():
  def __init__(self, data=DataDivision(), methods={"SVM": svm.SVC()}):
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

class Performance():
  def __init__(self, metric=metrics.accuracy_score):
    self.metric = metric
    pass
  def estimate(self, targets):
    perfs = {}
    for i in range(len(targets["actual"])):
      actual = np.array(targets['actual'][i])
      for clfname, predictions in targets.items():
        if clfname == 'actual':
          continue
        if clfname not in perfs:
          perfs[clfname] = []
        pred = np.array(targets[clfname][i])
        perfs[clfname].append(self.metric(actual,pred))
    return perfs

methods = {"standardSVM": Pipeline([('scaler',StandardScaler()),
                                 ('SVM',svm.SVC())]),
           "RandomForest": RandomForestClassifier()}
methods = {'StatSVM': Pipeline([('Stat', stat.ExtractionStatistical()),
                               ('scaler', StandardScaler()),
                               ('SVM', svm.SVC())]),
           'RandFor': Pipeline([('Stat', stat.ExtractionStatistical()),
                               ('scaler',StandardScaler()),
                               ('RandomForest', RandomForestClassifier())])}
data=cwru.DataCWRU(True)
targets = Experimenter(data,methods).perform()
results = Performance(lambda a,p: metrics.f1_score(a,p,average='macro')).estimate(targets)
#results = Performance(metrics.accuracy_score).estimate(targets)
for method, performance in results.items():
  print(method, performance, np.mean(performance))

