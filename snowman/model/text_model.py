# IMPORTS
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from snowman.model import text_cnn
from sklearn.metrics import roc_curve, auc
import random
import json

VERSION = "0.0.1"

MODEL_OUTPUT_FILEPATH = "../../fixtures/models/model_"+VERSION + "/"
MODEL_WEIGHTS_OUTPUT_FILEPATH = "../../fixtures/models/model_"+VERSION + "/weights"

class TextModel(object):
	def __init__(self):
		self.version = VERSION
		self.max_sequence_length = None
		self.max_char_index = None
		self.net = None
		self.prep_util = DataPrep()

	def train(self):

		# data preparation
		bl_strings = prep.load_url_file("../../fixtures/datasets/url_blacklist.txt", skip_lines=3)
		wl_strings = prep.load_url_file("../../fixtures/datasets/url_whitelist.txt", skip_lines=3)

		url_strings = bl_strings + wl_strings

		X = prep.to_one_hot_array(url_strings)
		Y = np.concatenate( [ np.ones(len(bl_strings)), np.zeros(len(wl_strings)) ])
		self.net = text_cnn(prep.max_index , prep.max_len)

		# model training
		(X_train, X_test,Y_train,Y_test) = prep.train_test_split(X,Y,.5)
		self.net.fit(X_train, Y_train, batch_size=128, epochs=5)

		#model evaluation
		Y_pred = self.net.predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
		auc_score = auc(fpr,tpr)
		print "\n AUC Score: " + str(auc_score) + "\n"


	def save(self):
		self.net.save(MODEL_WEIGHTS_OUTPUT_FILEPATH)
		model_cofiguration = {"max_sequence_length" : self.max_sequence_length,
		"max_char_index": self.max_char_index}
		with open(MODEL_OUTPUT_FILEPATH+"/config.json",'rw') as out_file:
			out_file.write(json.dumps(model_configuration))


	def load(self):
		model_configuration = json.load(MODEL_OUTPUT_FILEPATH+"/config.json")
		self.max_sequence_length = model_configuration["max_sequence_length"]
		self.max_char_index = model_configuration["max_char_index"]

	def predict(self, input_string):
		transformed = self.prep_util.to_one_hot(
			input_string, self.prep_util.max_index, self.prep_util.max_len)
		score = self.net.predict_proba(transformed)
		return score