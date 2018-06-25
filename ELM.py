from sklearn.preprocessing import LabelBinarizer
import numpy as np
from numpy import linalg as la
import tensorflow as tf

class ELM():
  def __init__(self, n_hidden_nodes=None):
    self.n_hidden_nodes = n_hidden_nodes
    self.labelbinarizer = LabelBinarizer()

  def fit(self, samples, targets):
    self.labels = list(set(targets))
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.concat([tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias'),tf.constant(samples, dtype=tf.float32)], axis=1, name='X')
    if self.n_hidden_nodes == None:
      self.n_hidden_nodes = int(X.shape[1]*10) 
    self.input_weight = tf.random_uniform([int(X.shape[1]),self.n_hidden_nodes], minval=-1, maxval=1, name='input_weight')
    H = tf.sigmoid(tf.matmul(X,self.input_weight))
    Ht = tf.transpose(H)
    self.output_weight = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Ht, H)), Ht), y)
    with tf.Session() as sess:
      self.output_weight.eval()

  def predict(self, samples, y=None):
    X = tf.concat([tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias'),tf.constant(samples, dtype=tf.float32)], axis=1, name='X')
    H = tf.sigmoid(tf.matmul(X,self.input_weight))
    prediction = tf.matmul(H,self.output_weight)
    with tf.Session() as sess:
      pred = prediction.eval()
    if pred.ndim>1 and pred.shape[1]>1:
      return np.array([self.labels[i] for i in np.argmax(pred, axis=1)])
    else:
      return np.array([self.labels[int(i>0)] for i in pred])

'''
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
start_time = time.time()
data = datasets.load_iris()#datasets.load_breast_cancer()#
clf = ELM()
#clf = Pipeline([('scaler',StandardScaler()), ('ELM', ELM())])
clf.fit(data["data"],data["target"])
resp = clf.predict(data["data"])
elapsed_time = time.time() - start_time
print(sum(resp==data["target"])/len(resp),elapsed_time)
#'''
