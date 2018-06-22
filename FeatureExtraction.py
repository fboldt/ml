from sklearn.base import TransformerMixin
import numpy as np
import scipy.stats as stats

# roor mean square
def rms(x):
  x = np.array(x)
  return np.sqrt(np.mean(np.square(x)))
# square root amplitude
def sra(x):
  x = np.array(x)
  return np.mean(np.sqrt(np.absolute(x)))**2
# peak to peak value
def ppv(x):
  x = np.array(x)
  return np.max(x)-np.min(x)
# crest factor
def cf(x):
  x = np.array(x)
  return np.max(np.absolute(x))/rms(x)
# impact factor
def ifa(x):
  x = np.array(x)
  return np.max(np.absolute(x))/np.mean(np.absolute(x))
# margin factor
def mf(x):
  x = np.array(x)
  return np.max(np.absolute(x))/sra(x)
# shape factor
def sf(x):
  x = np.array(x)
  return rms(x)/np.mean(np.absolute(x))
# kurtosis factor
def kf(x):
  x = np.array(x)
  return stats.kurtosis(x)/(np.mean(x**2)**2)

class  StatisticalTime(TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    return np.array([[rms(x), sra(x), stats.kurtosis(x), stats.skew(x), ppv(x), cf(x), ifa(x), mf(x), sf(x), kf(x)] for x in X])

