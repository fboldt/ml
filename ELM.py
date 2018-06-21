from sklearn.preprocessing import LabelBinarizer
import numpy as np
from numpy import linalg as la
import tensorflow as tf

class ELM():
  def __init__(self, n_hidden_nodes=None):
    self.random_range = 2
    self.n_hidden_nodes = n_hidden_nodes
    self.labelbinarizer = LabelBinarizer()

  def fit(self, samples, targets):
    self.labels = list(set(targets))
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.constant(samples, dtype=tf.float32, name='X')
    bias = tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias')
    Xb = tf.concat([bias, X], axis=1, name='Xb')
    if self.n_hidden_nodes == None:
      self.n_hidden_nodes = int(Xb.shape[0])//3 
    self.Wh = tf.scalar_mul(self.random_range,tf.random_uniform([int(Xb.shape[1]),self.n_hidden_nodes], minval=-1, maxval=1, dtype=tf.float32))
    H1 = tf.sigmoid(tf.matmul(Xb,self.Wh), name="H1")
    H1t = tf.transpose(H1)
    self.Wo = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(H1t, H1)), H1t), y)
    with tf.Session() as sess:
      self.Wo.eval()

  def predict(self, samples, y=None):
    X = tf.constant(samples, dtype=tf.float32, name='X')
    bias = tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias')
    Xb = tf.concat([bias, X], axis=1)
    prediction = tf.matmul(tf.sigmoid(tf.matmul(Xb,self.Wh)),self.Wo)
    with tf.Session() as sess:
      pred = prediction.eval()
    return np.array([self.labels[i] for i in np.argmax(pred, axis=1)])

'''
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
start_time = time.time()
data = datasets.load_iris()
clf = ELM()
#clf = Pipeline([('scaler',StandardScaler()), ('ELM', ELM())])
clf.fit(data["data"],data["target"])
resp = clf.predict(data["data"])
elapsed_time = time.time() - start_time
print(sum(resp==data["target"])/len(resp),elapsed_time)
#'''
