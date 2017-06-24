# IMPORTS
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, Input, Dense, concatenate
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep
from sklearn.metrics import roc_curve, auc
import random

def text_cnn(max_seq_index, max_seq_length):
	text_input = Input(shape = (max_seq_length,), name='text_input')
	x = Embedding(output_dim=15, 
			input_dim=max_seq_index, 
			input_length=max_seq_length)(text_input)

	conv_a = Conv1D(15,2, activation='relu')(x)
	conv_b = Conv1D(15,4, activation='relu')(x)
	conv_c = Conv1D(15,6, activation='relu')(x)

	pool_a = GlobalMaxPooling1D()(conv_a)
	pool_b = GlobalMaxPooling1D()(conv_b)
	pool_c = GlobalMaxPooling1D()(conv_c)

	flattened = concatenate(
		[pool_a, pool_b, pool_c])

	drop = Dropout(.2)(flattened)

	dense = Dense(1)(drop)
	out = Activation("sigmoid")(dense)

	model = Model(inputs=text_input, outputs=out)

	model.compile(loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])

	return model


