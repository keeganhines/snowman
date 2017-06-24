# IMPORTS
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from snowman.model.text_cnn import text_cnn
from sklearn.metrics import roc_curve, auc
import random
import json
import os

VERSION = "0.0.2"

pwd  = os.path.dirname(__file__)

MODEL_OUTPUT_FILEPATH = os.path.join(pwd, "../../fixtures/models/model_"+VERSION + "/")
MODEL_WEIGHTS_OUTPUT_FILEPATH = os.path.join(pwd,"../../fixtures/models/model_"+VERSION + "/weights")
MODEL_CONFIG_OUTPUT_FILEPATH = os.path.join(pwd,"../../fixtures/models/model_"+VERSION + "/config.json")

TRAINING_DATA_BLACKLIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/url_blacklist.txt")
TRAINING_DATA_WHITELIST_FILEPATH = os.path.join(pwd, "../../fixtures/datasets/url_whitelist.txt")

class TextModel(object):
	def __init__(self):
		self.version = VERSION
		self.max_sequence_length = None
		self.max_char_index = None
		self.net = None
		self.prep = DataPrep()

	def train(self):

		# data preparation
		bl_strings = self.prep.load_url_file(TRAINING_DATA_BLACKLIST_FILEPATH, skip_lines=3)
		wl_strings = self.prep.load_url_file(TRAINING_DATA_WHITELIST_FILEPATH, skip_lines=3)

		url_strings = bl_strings + wl_strings

		X = self.prep.to_one_hot_array(url_strings)
		Y = np.concatenate( [ np.ones(len(bl_strings)), np.zeros(len(wl_strings)) ])
		self.net = text_cnn(self.prep.max_index , self.prep.max_len)

		# model training
		(X_train, X_test,Y_train,Y_test) = self.prep.train_test_split(X,Y,.5)
		self.net.fit(X_train, Y_train, batch_size=128, epochs=5)

		#model evaluation
		Y_pred = self.net.predict(X_test)
		fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
		auc_score = auc(fpr,tpr)
		print "\n AUC Score: " + str(auc_score) + "\n"


	def save(self):
		print "Saving model under directory: " + MODEL_OUTPUT_FILEPATH
		if not os.path.isdir(MODEL_OUTPUT_FILEPATH):
			os.mkdir(MODEL_OUTPUT_FILEPATH) 

		self.net.save(MODEL_WEIGHTS_OUTPUT_FILEPATH)
		model_configuration = {"max_sequence_length" : self.prep.max_len,
		"max_char_index": self.prep.max_index}
		with open(MODEL_OUTPUT_FILEPATH+"/config.json",'w+') as out_file:
			out_file.write(json.dumps(model_configuration))


	def load(self):
		with open(MODEL_CONFIG_OUTPUT_FILEPATH, "r") as in_file:
			model_configuration = json.load(in_file)
		print "Loaded model config: " + str(model_configuration)

		self.prep.max_len = model_configuration["max_sequence_length"]
		self.prep.max_index = model_configuration["max_char_index"]

		self.net = load_model(MODEL_WEIGHTS_OUTPUT_FILEPATH)

	def predict(self, input_string):
		transformed = self.prep.to_one_hot(
			input_string, self.prep.max_index, self.prep.max_len)
		score = self.net.predict(transformed)
		return score