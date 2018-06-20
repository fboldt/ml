from sklearn.preprocessing import LabelBinarizer
import numpy as np
from numpy import linalg as la
import tensorflow as tf

class LinearMachine():
  def __init__(self):
    self.labelbinarizer = LabelBinarizer()
  def fit(self, samples, targets):
    self.labels = list(set(targets))
    #targets = self.labelbinarizer.fit_transform(y)
    #Xt = np.transpose(X)
    #self.W = np.matmul(linalg.pinv(X),targets)
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.constant(samples, dtype=tf.float32, name='X')
    Xt = tf.transpose(X)
    W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X)), Xt), y)
    with tf.Session() as sess:
      self.W = W.eval()
  def predict(self, samples, y=None):
    X = tf.constant(samples, dtype=tf.float32, name='X')
    W = tf.constant(self.W, dtype=tf.float32, name='W')
    py = tf.matmul(X,W)
    with tf.Session() as sess:
      pred = py.eval()
    idxs = np.argmax(pred, axis=1)
    resp = np.array([self.labels[i] for i in idxs])
    return resp

from sklearn import datasets
data = datasets.load_iris()
clf = LinearMachine()
clf.fit(data["data"],data["target"])
resp = clf.predict(data["data"])
print(resp)
