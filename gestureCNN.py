from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical  # Updated import

from keras.models import Model
import tensorflow as tf


from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow
else:
    import theano

K.set_image_data_format('channels_first')

import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import cv2
import matplotlib.pyplot as plt

# input image dimensions
img_rows, img_cols = 200, 200
img_channels = 1  # For grayscale use 1 value

# Batch_size to train
batch_size = 32
nb_classes = 5  # Number of output classes
nb_epoch = 15  # Number of epochs to train
nb_filters = 32
nb_pool = 2
nb_conv = 3

# data paths
path = "./"
path1 = "./gestures"
path2 = './imgfolder_b'

output = ["OK", "NOTHING", "PEACE", "PUNCH", "STOP"]
jsonarray = {}

def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        cv2.line(plot, (0, y), (int(h * mul), y), (255, 0, 0), w)
        cv2.putText(plot, items, (0, y + 5), font, 0.7, (0, 255, 0), 2, 1)
        y = y + w + 30

    return plot

def debugme():
    import pdb
    pdb.set_trace()

def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 + '/' + file)
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' + file, "PNG")

def modlistdir(path, pattern=None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if pattern is None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)
    return retlist

def loadCNN(bTraining = False):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # Model summary
    model.summary()
    # Model conig details
    model.get_config()
    
    if not bTraining :
        #List all the weight files available in current directory
        WeightFileName = modlistdir('.','.hdf5')
        if len(WeightFileName) == 0:
            print('Error: No pretrained weight file found. Please either train the model or download one from the https://github.com/asingh33/CNNGestureRecognizer')
            return 0
        else:
            print('Found these weight files - {}'.format(WeightFileName))
        #Load pretrained weights
        w = int(input("Which weight file to load (enter the INDEX of it, which starts from 0): "))
        fname = WeightFileName[int(w)]
        print("loading ", fname)
        model.load_weights(fname)

    # refer the last layer here
    print(model)
    # layer = model.layers[-1]
    # get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    # Get the last layer of the model
    
    layer = model.layers[-1]
    # Create a new model to get the output from the last layer
    get_output = Model(inputs=model.inputs, outputs=layer.output)
    
    return model


def guessGesture(model, img):
    global output, get_output, jsonarray
    image = np.array(img).flatten()
    image = image.reshape(img_channels, img_rows, img_cols)
    image = image.astype('float32')
    image = image / 255
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    prob_array = get_output([rimage, 0])[0]
    
    d = {}
    for i, item in enumerate(output):
        d[item] = prob_array[0][i] * 100

    guess = max(d.items(), key=lambda x: x[1])[0]
    prob = d[guess]

    if prob > 60.0:
        jsonarray = d
        return output.index(guess)
    else:
        return 1

def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 + '/' + imlist[0]))  # open one image to get size
    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images
    
    immatrix = np.array([np.array(Image.open(path2 + '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype='f')

    label = np.ones((total_images,), dtype=int)
    samples_per_class = int(total_images / nb_classes)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    data, Label = shuffle(immatrix, label, random_state=2)
    X, y = data, Label
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
    Y_train = to_categorical(y_train, nb_classes)  # Updated to use to_categorical
    Y_test = to_categorical(y_test, nb_classes)    # Updated to use to_categorical
    return X_train, X_test, Y_train, Y_test

def trainModel(model):
    X_train, X_test, Y_train, Y_test = initializers()
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_split=0.2)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname, overwrite=True)
    else:
        model.save_weights("newWeight.hdf5", overwrite=True)
        
    visualizeHis(hist)

def visualizeHis(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history.get('val_loss', [])
    train_acc = hist.history.get('accuracy', [])
    val_acc = hist.history.get('val_accuracy', [])
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)

    plt.show()

def visualizeLayers(model):
    imlist = modlistdir('./imgs')
    if len(imlist) == 0:
        print('Error: No sample image file found under \'./imgs\' folder.')
        return
    else:
        print('Found these sample image files - {}'.format(imlist))

    img = int(input("Which sample image file to load (enter the INDEX of it, which starts from 0): "))
    layerIndex = int(input("Enter which layer to visualize. Enter -1 to visualize all layers possible: "))
    
    if img <= len(imlist):
        image = np.array(Image.open('./imgs/' + imlist[img]).convert('L')).flatten()
        print('Guessed Gesture is {}'.format(output[guessGesture(model, image)]))
        
        image = image.reshape(img_channels, img_rows, img_cols).astype('float32') / 255
        image = image.reshape(1, img_channels, img_rows, img_cols)
        
        print("Image after reshaping: ", image.shape)

        # Display layer outputs
        for layer in range(len(model.layers)):
            if layerIndex != -1 and layerIndex != layer:
                continue
            print("Layer {} - {}".format(layer, model.layers[layer].name))
            layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
            output = layer_output([image, 0])[0]
            print("Output shape: ", output.shape)

def main():
    model = loadCNN(bTraining=True)
    trainModel(model)
    visualizeLayers(model)

if __name__ == '__main__':
    main()
