import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.metrics import cohen_kappa_score
import numpy as np

class Models:
    def __init__(self, input_shape=None):
        if input_shape != None:
          self.input_shape = input_shape
            # self.rnn_lstm = self.rnn(input_shape=input_shape, RNN=LSTM)
            # self.rnn_gru = self.rnn(input_shape=input_shape, RNN=GRU)
    
    def svm(self):
        return SVC()
    
    def random_forest(self):
        return RandomForestClassifier()
    
    def rnn(self, RNN):
        if RNN == 'LSTM': RNN = LSTM
        elif RNN == 'GRU': RNN = GRU
        
        """Define the model."""
        model = Sequential()
        model.add(Bidirectional(RNN(self.input_shape[-1], dropout=0.4, recurrent_dropout=0.4, input_shape=[1, self.input_shape[-1]], return_sequences=True)))
        model.add(Bidirectional(RNN(64, recurrent_dropout=0.4)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
        model.build(self.input_shape)
        model.summary()

        return model
    
class Train_Test_aes:
    def __init__(self):
        pass
    def test_acc(self, model, X_train, y_train):
        xx = X_train.squeeze(1)
        cv_results = cross_val_score(model, xx, y_train, cv=5, scoring='accuracy',error_score='raise')
        return cv_results.mean(), cv_results.std()
    
    def test_rnn_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred)

        result_aug = cohen_kappa_score(y_test,y_pred,weights='quadratic')
        print("Kappa Score: {}".format(result_aug))
    
    def eval_model_gen(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mean = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        return mean, mse
    