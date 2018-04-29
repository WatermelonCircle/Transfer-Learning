import pickle
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
import numpy as np
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 40, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    y_one_hot_val = label_binarizer.fit_transform(y_val)
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_val= keras.utils.to_categorical(y_val, num_classes)
    # TODO: define your model and hyperparams here
    
    
    # make sure to adjust the number of classes based on
    # the dataset
    nb_class=len(np.unique(y_train))
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1:])))
    model.add(Dense(nb_class,activation='softmax'))
    print(X_train.shape[1:])
    
    # 10 for cifar10
    # 43 for traffic

    # TODO: train your model here
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_one_hot, nb_epoch=FLAGS.epochs,batch_size=FLAGS.batch_size,validation_data=(X_val, y_one_hot_val),shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
