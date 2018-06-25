from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

class LinearMachine():
  def __init__(self):
    self.labelbinarizer = LabelBinarizer()

  def fit(self, samples, targets):
    self.labels = list(set(targets))
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.concat([tf.ones([samples.shape[0],1], dtype=tf.float32), tf.constant(samples, dtype=tf.float32)], axis=1, name='X')
    Xt = tf.transpose(X)
    self.W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X)), Xt), y)
    with tf.Session() as sess:
      self.W.eval()

  def predict(self, samples, y=None):
    X = tf.concat([tf.ones([samples.shape[0],1], dtype=tf.float32), tf.constant(samples, dtype=tf.float32)], axis=1, name='X')
    prediction = tf.matmul(X,self.W)
    with tf.Session() as sess:
      pred = prediction.eval()
    if pred.ndim>1 and pred.shape[1]>1:
      return np.array([self.labels[i] for i in np.argmax(pred, axis=1)])
    else:
      return np.array([self.labels[int(i>0)] for i in pred])

'''
from sklearn import datasets
data = datasets.load_iris()
data = datasets.load_breast_cancer()
clf = LinearMachine()
clf.fit(data["data"],data["target"])
resp = clf.predict(data["data"])
print(sum(resp==data["target"])/len(resp))
#'''
