import tensorflow as tf
import numpy as np


class Hopfield():
  def __init__(self, max_iter=10, f=lambda x: x/abs(x)):
    self.max_iter = max_iter
    self.f = f
  
  def fit(self, samples, targets=None):
    U = tf.constant(samples, dtype=tf.complex64, name='U')
    Uc = tf.conj(U)
    Upinv = tf.matmul(tf.matrix_inverse(tf.matmul(Uc,U)),Uc)
    W = tf.matmul(U, Upinv)
    with tf.Session() as sess:
      self.W = np.transpose(W.eval())

  def predict(self, samples, targets=None):
    for it in range(self.max_iter):
      for x in samples:
        for i,xi in enumerate(x):
          v = np.dot(self.W[i],x)
          x[i] = self.f(v)
    return samples
        

X = np.array([[ 1,-1, 1],
              [-1, 1, 1],
              [ 1, 1,-1]])
print(X)
hf = Hopfield()
hf.fit(X)
X = np.array([[ 1,-1,-1],
              [-1,-1, 1],
              [-1, 1,-1]])
pred = hf.predict(X)
print(pred)

