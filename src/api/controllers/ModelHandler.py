import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

class ModelHandler(object):
    """
        Things the constructor will check for in named arguments 
        1. layers  - list of nodes in hidden layer 
        2. activations - list activation functions for the hidden layers
        3. learning rate - learning rate for the optimizer
        4. batch_size - size of batch
        5. epochs - number of epochs 
        6. problem - The problem based on Dataset 
        ?? loss function -> mse [binary_crossentropy, + others in possible development]
        ?? optimizer -> SGD [Adam + others in possible development]
    """
    def __init__(self, **kwargs) -> None:
      if "prob" not in kwargs.keys():
        self.prob = "regression"
      else: 
        self.prob = kwargs["prob"]

      # Checking for the layers 
      if "layers" not in kwargs.keys():
        self.layers = [1]
      else: 
        self.layers = kwargs["layers"]
      
        # Checking for the activation
      if "activations" not in kwargs.keys():
        self.activations = ["relu"]
      else: 
        self.activations = kwargs["activations"]

        # Checking for the batch_size
      if "batch_size" not in kwargs.keys():
        self.batch_size = 1
      else: 
        self.batch_size = kwargs["batch_size"]
      
        # Checking for the epochs
      if "epochs" not in kwargs.keys():
        self.epochs = 10
      else: 
        self.epochs = kwargs["epochs"]

        # Checking for the learning rate : lr
      if "lr" not in kwargs.keys():
        self.lr = 0.05
      else: 
        self.lr = kwargs["lr"]

    def createModel(self) -> None:
      options = []
      
      for i, num in enumerate(self.layers):
        options.append(Dense(num, activation=self.activations[i], name="dense-{}".format(i)))
        
      if self.prob == "heart_disease":
        options.append(Dense(1, activation="sigmoid", name="output"))
      elif self.prob == "iris":
        options.append(Dense(3, activation="softmax", name="output"))
      elif self.prob == "house_price":
        options.append(Dense(1, activation="relu", name="output"))

      self.model = Sequential(options)

      optimizer = SGD(self.lr)

      self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    
    def train(self, features, output) -> dict:
      hist = self.model.fit(features,
                            output, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            verbose=0)
      return {
          "epochs": hist.epoch,
          "loss_hist": hist.history['loss'],
          "acc_hist": hist.history['accuracy']
      }

    def evaluate(self, features, output) -> dict:
      eval = self.model.evaluate(features, 
                                  output)
      return {
          "loss": eval[0],
          "acc": eval[1]
      }

    def predict(self, features) -> pd.DataFrame:
      return self.model.predict(features)
    
    def clear(self):
      clear_session()
