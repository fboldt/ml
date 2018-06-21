from sklearn.preprocessing import LabelBinarizer
import numpy as np
from numpy import linalg as la
import tensorflow as tf

class ELM():
  def __init__(self, factor=5):
    self.factor = factor
    self.labelbinarizer = LabelBinarizer()

  def fit(self, samples, targets):
    self.labels = list(set(targets))
    y = tf.constant(self.labelbinarizer.fit_transform(targets), dtype=tf.float32, name='y')
    X = tf.constant(samples, dtype=tf.float32, name='X')
    bias = tf.ones([samples.shape[0],1], dtype=tf.float32, name='bias')
    Xb = tf.concat([bias, X], axis=1, name='Xb')
    #self.Wh = tf.random_uniform([int(Xb.shape[1]),int(Xb.shape[1]*self.factor)], minval=-0.5, maxval=0.5, dtype=tf.float32)
    self.Wh = tf.random_uniform([int(Xb.shape[1]),int(Xb.shape[1]*self.factor)], minval=-2, maxval=2, dtype=tf.float32)
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
