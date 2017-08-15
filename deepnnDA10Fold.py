'''Trains a convnet for the ECK face database.
Gets to ??.??% test accuracy after 50 epochs.
17 seconds per epoch on a Tesla C2070 GPU.
'''

from __future__ import print_function
import numpy as np
import collections

# fix random seed for reproducibility
seed = 137
np.random.seed(seed)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

batch_size = 128
nb_classes = 7
nb_epoch = 250
#number of folds for cross validation
n_folds = 10
# input image dimensions
img_rows, img_cols = 96, 96
# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
quadrant_pooling = (4,4)
# convolution kernel size
kernel_size = (5, 5)

# expresiones
expr = np.array([0,1,2,3,4,5,6])

# To stop the training based on val_loss value
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class EarlyStoppingByAccTrain(Callback):
    def __init__(self, monitor='acc', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

# Gives the index of the maximum element of an arraw
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx

if __name__ == "__main__":

    #Lee los datos
    X = np.load('/home/rosana/data/X.npy')
    y = np.load('/home/rosana/data/y.npy')
    subjs = np.load('/home/rosana/data/subjs.npy')

    #File to write the results
    results = open('/data2/rosana/results.txt', 'w')
    X=np.transpose(X,(0,2,3,1))

#Tiene que empezar en clase 0 o da error
    y = y - 1


# Augmentation - rotation & axis shifting
    datagen = ImageDataGenerator( 
        rotation_range=12,
        width_shift_range=0.22,
        height_shift_range=0.15,
#        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.1,
#        horizontal_flip = True,
        fill_mode='nearest')
# define data preparation
#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#skf = StratifiedKFold(y,10, shuffle=True)

#kf = KFold(n_splits=n_folds)
	
#kf = KFold(n_folds, shuffle=True, random_state=seed)

#Validation set
#    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#    print ("hola")
#else:
#    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#    print ("rama2")

    cvscores = []
#for i, (train, test) in enumerate(kfold):
    fold_number=0

#Saco cuantos subjects hay y primera posicion de cadas subject en el array
    u, indices = np.unique(subjs, return_index=True)

#Hago un numpy array de ceros, para ir sumando las confusion matrices
    confusion_acc = np.zeros((nb_classes,nb_classes), dtype=np.int)


    callbacks = [
#    EarlyStoppingByLossVal(monitor='val_loss', value=0.01, verbose=1),
    EarlyStoppingByAccTrain(monitor='acc', value=0.98, verbose=1),
#    ModelCheckpoint("/home/rosana/data/best.kerasModelWeights", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
#    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

#for train, test in kf.split(X):
    for s in range (10): # Hay 118 sujetos (len(indices)):
        if s < 9: 
            X_train = np.concatenate((X[:indices[11*s]], X[indices[11*s+11]:]))
            y_train = np.concatenate((y[:indices[11*s]], y[indices[11*s+11]:]))
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            X_test = X[indices[11*s]:indices[11*s+11]]
            y_test = y[indices[11*s]:indices[11*s+11]]
            Y_test = np_utils.to_categorical(y_test, nb_classes)
 #           X_val = X[indices[s+1]:indices[s+2]]
 #           y_val = y[indices[s+1]:indices[s+2]]
 #           Y_val = np_utils.to_categorical(y_val, nb_classes)
        else:
#Llegamos al ultimo elemento, y entonces testeo con el ultimo y valido con el primer elemento
            X_train = X[:indices[11*s+11]]
            y_train = y[:indices[11*s+11]]
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            X_test = X[indices[11*s+11]:]
            y_test = y[indices[11*s+11]:]
            Y_test = np_utils.to_categorical(y_test, nb_classes)
#            X_val = X[indices[0]:indices[1]]
#            y_val = y[indices[0]:indices[1]]
#            Y_val = np_utils.to_categorical(y_val, nb_classes)

        print(X_train.shape[0], 'train samples')
        results.write("Train samples: %d\n" % X_train.shape[0])
#        print(X_val.shape[0], 'val samples')
#        results.write("Validation samples: %d\n" % X_val.shape[0])
        print(X_test.shape[0], 'Test samples')
        results.write("Test samples: %d\n" % X_test.shape[0])
#    ii += 1
#    fold_number += 1
#    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#    Y_train = np_utils.to_categorical(y_train, nb_classes)
#    Y_test = np_utils.to_categorical(y_test, nb_classes)
#    Y = np_utils.to_categorical(y, nb_classes)
        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size)) #agr
        model.add(Convolution2D(2*nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Convolution2D(4*nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=quadrant_pooling))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(300))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)  
 

# fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=1, callbacks=callbacks)
#            validation_data=(X_val, Y_val), callbacks=callbacks)

#    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#    verbose=1, validation_data=(X_test, Y_test))
    
#        model.fit(X_train.astype('float32'), Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#          shuffle=True, verbose=1, validation_data=(X_val, Y_val),
#          callbacks=callbacks)
#    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#    verbose=1, validation_data=(X_val, Y_val))
   
        print('Numero de subject:', s)
        results.write("Numero de subject: %d\n" % s)
 
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        results.write("%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
        for sample_val in range(X_test.shape[0]):
            cvscores.append(scores[1] * 100)


        Preds = np.zeros(len(y_test))
        preds = model.predict(X_test)
        for i in range(len(preds)):
            Preds[i] = nanargmax(preds[i])[0]

        y_test_count = collections.Counter(y_test)
        print("Ocurrencias y_test\n:", y_test_count)
        
        cm = confusion_matrix(y_test, Preds, expr)
        
        print('Confusion matrix test:\n', cm)
        
        np.savetxt(results,
           cm,
           delimiter='\t',
           fmt=('%2u', '%2u', '%2u', '%2u', '%2u', '%2u', '%2u'),
           header='Confusion matrix test'
           )

        confusion_acc += cm
        print('Confusion matrix acumulada:\n', confusion_acc)
        np.savetxt(results,
           confusion_acc,
           delimiter='\t',
           fmt=('%2u', '%2u', '%2u', '%2u', '%2u', '%2u', '%2u'),
           header='Confusion matrix acumulada'
           )

        #Normalizo
        confusion_acc_norm = confusion_acc.astype('float') / confusion_acc.sum(axis=1)[:, np.newaxis]
        print('Confusion matrix acumulada normalizada:\n', confusion_acc_norm)
        np.savetxt(results,
           confusion_acc_norm,
           delimiter='\t',
           fmt=('%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f'),
           header='Confusion matrix acumulada normalizada'
           )

        print("Total muestras testeadas: %2u. Acc media y std: %.2f%% (+/- %.2f%%)\n\n" % (len(cvscores), np.mean(cvscores), np.std(cvscores)))
        results.write("Total muestras testeadas: %2u. Acc media y std: %.2f%% (+/- %.2f%%)\n\n" % (len(cvscores), np.mean(cvscores), np.std(cvscores)))

