# IMPORTS
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np
from snowman.model.util.utilities import DataPrep

VERSION = "0.0.1"

def main():

	prep = DataPrep()

	bl_strings = prep.load_url_file("../../fixtures/datasets/url_blacklist.txt", skip_lines=3)
	wl_strings = prep.load_url_file("../../fixtures/datasets/url_whitelist.txt", skip_lines=3)

	url_strings = bl_strings + wl_strings

	X = prep.to_one_hot_array(url_strings)
	Y = np.concatenate( [ np.ones(len(bl_strings)), np.zeros(len(wl_strings)) ])

	(X_train, X_test,Y_train,Y_test) = prep.train_test_split(X,Y,.5)


	# MODEL DEFINITION
	model = Sequential()
	model.add(Embedding(prep.max_index, 20, input_length=prep.max_len))
	model.add( Conv1D(15,6))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=128, epochs=5)

	model.save("../../fixtures/models/model_"+VERSION)

if __name__ == '__main__':
	main()
