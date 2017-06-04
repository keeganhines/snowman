# IMPORTS
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from .util.utilities import DataPrep

VERSION = "0.0.1"

def main():

	prep = DataPrep()

	bl_strings = prep.load_url_file("../fixtures/datasets/urls_blacklist.txt", skip_lines=3)
	wl_strings = prep.load_url_file("../fixtures/datasets/_url_whitelist.txt", skip_lines=3)

	url_strings = bl_strings + wl_strings

	X = prep.to_one_hot_array(url_strings)
	Y = np.concatenate( [ np.ones(len(bl_strings)), np.zeros(len(wl_strings)) ])

	(X_train, X_test,Y_train,Y_test) = prep.train_test_split(X,Y,.5)


	# MODEL DEFINITION
	model = Sequential()
	model.add(Embedding(prep.max_index, 20, input_length=prep.max_len))
	model.add( Conv1D(15,6))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=128, nb_epoch=25)

	model.save_weights("../fixtures/models/model_"+VERSION)



if __name__ == '__main__':
	main()