import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import recall_score,fbeta_score,f1_score,precision_score,matthews_corrcoef



class MatthewsError(tf.keras.metrics.Metric):
  def __init__(self, name="matthews_score", **kwargs):
    super().__init__(name=name, ** kwargs)
    self.error = self.add_weight(name="ms")

  def update_state(self, y_true, y_pred, sample_weight=None):  
    m = matthews_corrcoef(y_true, y_pred)
    self.error.assign_add(m)

  def result(self):
    return self.error

  def reset_states(self):
    self.error.assign(0.)    



class FBetaScore(tf.keras.metrics.Metric):
  def __init__(self, name="fbeta_score", **kwargs):
    super().__init__(name=name, ** kwargs)
    self.error = self.add_weight(name="fbs")

  def update_state(self, y_true, y_pred, sample_weight=None):    
    m = fbeta_score(y_true, y_pred, average='weighted', beta=0.5) 
    self.error.assign_add(m)

  def result(self):
    return self.error

  def reset_states(self):
    self.error.assign(0.)    



class F1Score(tf.keras.metrics.Metric):
  def __init__(self, name="f1_score", **kwargs):
    super().__init__(name=name, ** kwargs)
    self.error = self.add_weight(name="f1s")

  
  def update_state(self, y_true, y_pred, sample_weight=None):   

    m = tf.keras.metrics.Precision()
    m.update_state(y_true,y_pred)  
    precision = m.result()._numpy()
    

    n = tf.keras.metrics.Recall()
    n.update_state(y_true,y_pred)
    recall = n.result().numpy()

    f1 = 2(precision * recall)/(precision+recall)
    self.error.assign_add(f1)

  def result(self):
    return self.error

  def reset_states(self):
    self.error.assign(0.)  



