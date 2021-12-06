from numpy import np
from builins import object
from builins import range
from past.builtins import xrange


class KnearestNeighbor(Object):
  """A knn classifier with l2 distance"""
  def __init__(self):
    pass
  def train(self, X, y):
    """ 
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    """

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distance_one_loop(X)
    else:
      dists = self.compute_distance_two_loop(X)

    return self.predict_labels(dists, k=k)


  def compute_distances_no_loops(self, X):

    num_train = self.X_train.shape[0]
    num_test = self.X.shape[0]
    
    dists = np.zeros((num_test, num_train))

    mul = np.dot(X, self.X_train.T)
    X1 = np.sum(np.square(self.X_train), axis = 1)
    X1 = X1.reshape(1, num_train)
    Y1 = np.sum(np.square(X), axis = 1)
    Y1 = Y1.reshape(num_test, 1)

    dists = np.sqrt(X1 - 2*mul + Y1)

    return dists
  
  def compute_distance_one_loop(self, X):
    num_train = self.X_train.shape[0]
    num_test = self.X.shape[0]
    
    dists = np.zeros((num_test, num_train))

    for i in range(num_test):
      dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train)))
      
    return dists 
  
  def compute_distance_two_loop(self, X):

    num_train = self.X_train.shape[0]
    num_test = self.X.shape[0]
    
    dists = np.zeros((num_test, num_train))

    for i in range(num_test):
      for j in range(num_train):
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))


  def predict_labels(self, dists, k=1):
    
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=self.y_train.dtype)
    
    for i in range(num_test):
      
      closest_y = []

      idx = np.argsort(dists[i])[:k]
      closest_y = self.y_train[idx]

      y_pred = np.argmax(np.bincount(closest_y))

    return y_pred



   


