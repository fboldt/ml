from sklearn.base import TransformerMixin
import numpy as np
import scipy.stats as stats

# roor mean square
def rms(x):
  x = np.array(x)
  return np.sqrt(np.mean(x**2))
# peak to peak value
def ppv(x):
  x = np.array(x)
  return np.max(x)-np.min(x)
# crest factor
def cv(x):
  x = np.array(x)
  return max(np.absolute(x))/rms(x)
# shape factor
def sf(x):
  x = np.array(x)
  return rms(x)/np.mean(np.absolute(x))
# impact factor
def ip(x):
  x = np.array(x)
  return max(np.absolute(x))/np.mean(np.absolute(x))

class  ExtractionStatistical():
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    #return np.array([[rms(np.real(np.fft.fft(sample)))] for sample in X])
    return np.array([[stats.skew(sample),stats.kurtosis(sample),rms(sample),ppv(sample),cv(sample),sf(sample),ip(sample)] for sample in X])
