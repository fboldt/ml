from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

class LinearMachine():
  def __init__(self):
    self.labelbinarizer = LabelBinarizer()

  def fit(self, samples, targets):
    self.labels = list(set(targets))
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.constant(samples, dtype=tf.float32, name='X')
    bias = tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias')
    Xb = tf.concat([bias, X], axis=1, name='Xb')
    Xt = tf.transpose(Xb)
    W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, Xb)), Xt), y)
    with tf.Session() as sess:
      self.W = W.eval()

  def predict(self, samples, y=None):
    X = tf.constant(samples, dtype=tf.float32, name='X')
    bias = tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias')
    Xb = tf.concat([bias, X], axis=1)
    W = tf.constant(self.W, dtype=tf.float32, name='W')
    py = tf.matmul(Xb,W)
    with tf.Session() as sess:
      pred = py.eval()
    return np.array([self.labels[i] for i in np.argmax(pred, axis=1)])

#'''
from sklearn import datasets
data = datasets.load_iris()
clf = LinearMachine()
clf.fit(data["data"],data["target"])
resp = clf.predict(data["data"])
print(sum(resp==data["target"])/len(resp))
#'''
