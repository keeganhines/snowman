import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

class DataPrep:
    def __init__(self):
        pass
    
    def load_url_file(self, file_path, skip_lines=0):
        with open(file_path) as file:
            lines = file.readlines()
        raw_url_strings = [line[:-2] for line in lines[skip_lines:]]
        return raw_url_strings
        
    def to_one_hot_array(self, string_list, max_index= 256):
        self.max_index = max_index
        x_one_hot = [one_hot(" ".join(list(sentence)), n = max_index) for sentence in string_list]
        self.max_len = max([len(s) for s in x_one_hot])
        X = np.array(pad_sequences(x_one_hot, maxlen=self.max_len))
        
        self.relevant_indices = np.unique(X)
        
        charset = set(list(" ".join(string_list)))
        self.charset = charset 
        
        encoding = one_hot(" ".join(charset),n=max_index)
        self.charset_map = dict(zip(charset,encoding) )
        self.inv_charset_map = dict(zip(encoding, charset) )
        
        return X
        
    def shuffle(self, X,Y):
        a = range(Y.size)
        np.random.shuffle(a)

        X = X[a]
        Y = Y[a]
        
        return(X,Y)
    
    def train_test_split(self, X,Y,proportion):
        (X,Y) = self.shuffle(X,Y)
        max_ind = int(proportion * X.shape[0])
        return(X[:max_ind,:],X[(max_ind+1):,:],Y[:max_ind,], Y[(max_ind+1):, ])