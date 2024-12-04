import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class convkeras:
	def __init__(self):
		self.train_X = {}
		self.train_y = {}
		self.layers = []
		self.sgd_params = {'learning_rate':0.01}
		self.loss='squared_error'
		self.metrics = ['mae','mse']
		self.verbose = 0
		

	def adapt_scaler(self):
		self.scaler = tf.keras.layers.Normalization(axis=-1)
		self.scaler.adapt(np.array(self.test_X))

	def optimizer(self):
		return tf.keras.optimizers.SGD(**self.sgd_params)

	def specify_model(self,layers=None):
		if layers == None:
			layers = self.scaler + self.layers
		self.model = keras.Sequential(layers)
		self.model.compile(loss=self.loss,optimizer=self.optimizer())

	def fit_model(self,epochs,verbose=1,validation_split=0.05):
		self.fitted = self.model.fit(
			self.train_X,self.train_y,
			verbose=self.verbose,validation_split=validation_split,
			epochs=self.epochs
		)
