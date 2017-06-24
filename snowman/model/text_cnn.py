# IMPORTS
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input, Dense
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from sklearn.metrics import roc_curve, auc
import random


# # obviously we want to make all of these topology decisions configurable...
# def text_cnn(max_seq_index, max_seq_length):
# 	# MODEL DEFINITION
# 	model = Sequential()
# 	model.add(Embedding(max_seq_index, 15, input_length=max_seq_length))
# 	model.add(Conv1D(15,3))
# 	model.add(Dropout(.2))
# 	model.add(GlobalMaxPooling1D())
# 	model.add(Dense(1))
# 	model.add(Activation('sigmoid'))

# 	model.compile(loss='binary_crossentropy',
# 	              optimizer='rmsprop',
# 	              metrics=['accuracy'])

# 	return model

def text_cnn(max_seq_index, max_seq_length):
	text_input = Input(shape = (max_seq_length,), name='text_input')
	x = Embedding(output_dim=15, 
			input_dim=max_seq_index, 
			input_length=max_seq_length)(text_input)
	conv_3 = Conv1D(15,3)(x)
	drop = Dropout(.2)(conv_3)
	pool = GlobalMaxPooling1D()(drop)
	dense = Dense(1)(pool)
	out = Activation("sigmoid")(dense)

	model = Model(inputs=text_input, outputs=out)

	model.compile(loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])

	return model


