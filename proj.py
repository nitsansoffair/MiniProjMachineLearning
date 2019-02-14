import random
import math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keras
import itertools

from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.datasets import load_iris

# Functions
# Rotate P.c by Angle
def rotate(pc, angle):
    pc[:] = pc[:] - 0.5 # Move to Center
    # Rotate Matrix
    Rx = np.array([(1, 0, 0),
                   (0, math.cos(angle), math.sin(angle)),
                   (0, -math.sin(angle), math.cos(angle))])
    pc = np.dot(pc, Rx) # Dot Production
    pc[:] = pc[:] + 0.5 # Move back to original coordinates
    return pc

# Input: <FactorAngle>, <PC>, <nAngles>
# Output: Volume after rotate P.C by <Agnle>
def pointcloud2volumeWithAngle(pc, angles_num, factor, dim = 32):
    angle = (360 / angles_num) # Angle
    angle = angle * factor * (math.pi / 180) # multiAngle
    vol = np.zeros(dim, dim, dim) # Volume
    temp = np.copy(pc) # Copy
    temp = rotate(temp, angle) # Rotate
    temp[:] *= (dim - 1) # Idx of Volume
    temp = temp.astype(int) # Int
    temp = np.clip(temp, 0, 31) # Limit Values {0, 31}
    vol[temp[:, 0], temp[:, 1], temp[:, 2]] = 1.0 # Volume
    return vol

# PC to Volume, With Random Angle
def pointcloud2volumeWithRandomAngle(pc, angles_num, factor, dim = 32):
    angle = (360 / angles_num) 
    angle = random.randint(angle * factor - angle, angle * factor) * math.pi / 180 # Rnd {angle * factor, angle * factor - angle}
    vol = np.zeros(dim, dim, dim)
    temp = np.copy(pc)
    temp = rotate(temp, angle) # Rotate
    temp[:] *= (dim - 1) # Idx of Volume
    temp = temp.astype(int) # Int
    temp = np.clip(temp, 0, 31) # Limit Values {0, 31}
    vol[temp[:, 0], temp[:, 1], temp[:, 2]] = 1.0 # Volume
    return vol

# PC to Volume
def pointcloud2volume(pc, dim=32):
    vol = np.zeros(dim, dim, dim)
    temp = np.copy(pc)
    temp[:] *= (dim - 1) # Idx of Volume
    temp = temp.astype(int) # Int
    temp = np.clip(temp, 0, 31) # Limit Values {0, 31}
    vol[temp[:, 0], temp[:, 1], temp[:, 2]] = 1.0 # Volume
    return vol

# PC to Volume, With One Angle
def pointcloud2volumeWithSimpleAngle(pc, angle_, dim = 32):
    angle = angle_ * math.pi / 180 # Radians
    vol = np.zeros(dim, dim, dim) # Volume
    temp = np.copy(pc) # Copy
    print (angle) # Debug ?
    
    if (angle != 0):
        temp = rotate(temp, angle) # Rotate

    temp[:] *= (dim - 1) # Volume
    temp = temp.astype(int) # Int
    temp = np.clip(temp, 0, 31) # Limit Values {0, 31}
    vol[temp[:, 0], temp[:, 1], temp[:, 2]] = 1.0 # Volume
    return vol

# Input: Volume
# Output: Depth map
def vol2depthmap(v, bg_val = 40.):
    vol = v.argmax(2)  # Maximum from {2} dim, Decrease by one dim
    vol[vol == 0] = bg_val # 0 => Background_Value
    return vol

# Input: Samples
# Output: Data set of Depth maps
def CreateDMDS_regular(samples):
    ln = len(samples)  # Samples Length
    DMDS = np.zeros(ln, 32, 32) # Initiate 3D array
    for i in range(len(samples)):
        v = pointcloud2volume(samples[i]) # Volume
        DMDS[i] = vol2depthmap(v) # Depth map
    return DMDS

# Creates data set with angles
def CreateDmdsWithAngle(samples, angles_num):
    ln = len(samples)
    DMDS = np.zeros(ln, angles_num, 32, 32) # One more dim
    DMDS_helper = np.zeros(angles_num, 32, 32) # One less dim
    for i in range(ln):
        for curr_angle in range(angles_num):
            v = pointcloud2volumeWithAngle(samples[i], angles_num, curr_angle) # Volume
            DMDS_helper[curr_angle] = vol2depthmap(v) # Depth map
        DMDS[i] = DMDS_helper # Depth maps
    return DMDS

# Output: Data set of depth maps, no rotate
def CreateDmdsWithNoAngle(samples, angles_num):
    ln = len(samples)
    DMDS = np.zeros(ln, angles_num, 32, 32)
    DMDS2_helper = np.zeros(angles_num, 32, 32)
    for i in range(ln):
        for curr_angle in range(angles_num):
            v = pointcloud2volumeWithAngle(samples[i], 1, 1) # No rotate
            DMDS2_helper[curr_angle] = vol2depthmap(v) # Depth map
        DMDS[i] = DMDS2_helper # Depth maps
    return DMDS

# Output: Data set of depth maps, random angles
def CreateDmdsWithRandomAngle(samples, angles_num):
    ln = len(samples)
    DMDS = np.zeros(ln, angles_num, 32, 32)
    DMDS_helper = np.zeros(angles_num, 32, 32)
    for i in range(ln):
        for curr_angle in range(angles_num):
            v = pointcloud2volumeWithRandomAngle(samples[i], angles_num, curr_angle)
            DMDS_helper[curr_angle] = vol2depthmap(v)
        DMDS[i] = DMDS_helper
    return DMDS

# Output: Data set of depth maps, one angle
def CreateDmdsWithSimpleAngle(samples, angle):
    ln = len(samples)
    DMDS = np.zeros(ln, 32, 32)
    for i in range(ln):
        v = pointcloud2volumeWithAngle(samples[i], 1, angle) # Volume
        DMDS[i] = vol2depthmap(v) # Depth map
    return DMDS

# Output: Data set of depth maps, with "mirror"
def CreateDmdsWithMirror(samples):
    ln = len(samples)
    DMDS = np.zeros(ln, 2, 32, 32)
    DMDS_helper = np.zeros(2, 32, 32)
    for i in range(ln):
        v_no_mirror = pointcloud2volume(samples[i])
        v_mirror = vol_mirror(v_no_mirror) # Volume, with "mirror"
        DMDS_helper[0] = vol2depthmap(v_no_mirror) # Volume, no "mirror"
        DMDS_helper[1] = vol2depthmap(v_mirror)
        DMDS[i] = DMDS_helper
    return DMDS

# Output: mirror to volume
def vol_mirror(v):
    v = np.flip(v, 1)
    return v

# Plot the depth map
def plot_depth_map(img_num, num_rot, is_mirror):
    vs = []
    for i in range(num_rot):
        vol = pointcloud2volumeWithAngle(samples[img_num], num_rot, i)
        if (is_mirror == 1): # With "mirror"
            vol = vol_mirror(vol)
        vs.append(vol2depthmap(vol))

    plt.imshow(np.vstack(vs), cmap = 'jet') # "Paint" depth map
    plt.colorbar() # Add color bar
    plt.show()

# Output: Paint confusion matrix
def conf_matrix(which, angle_param, random):
    iris = datasets.load_iris()
    labels = modelnet10_val['labels']
    samples = modelnet10_val['samples']
    mul_ang = 1
    
    if which == 'reg':
        samplesData = CreateDMDS_regular(samples) # Regular
        
    if which == 'mirror':
        samplesData = CreateDmdsWithMirror(samples) # Mirror
        mul_ang = angle_param
        
    if which == 'one_v_a':
        samplesData = CreateDmdsWithSimpleAngle(samples, angle_param) # With angle
    
    if which == 'mulVcnn':
        samplesData = CreateDmdsWithNoAngle(samples, angle_param) # Mul shape by <num_ang>
        mul_ang = angle_param
    
    if (which == 'WithRandom'):
        if (random=='true'): # with/out random m.v.Cnn, mul. shape by num_a.
            samplesData = CreateDmdsWithRandomAngle(samples, angle_param)
        else:
            samplesData = CreateDmdsWithAngle(samples, angle_param)
        mul_ang = angle_param

    (x_test, y_test) = (samplesData, labels)

    if K.image_data_format() == 'channels_first':   
        x_test = x_test.reshape(x_test.shape[0], 1*32*32*mul_ang) # , 32, 32)

    else:
        x_test = x_test.reshape(x_test.shape[0], 32*32*1*mul_ang) # , 32, 1)
    
    X = x_test[:908]
    y = labels
    
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = svm.SVC(kernel = 'linear', C = 0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision = 2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names,
                          title = 'Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = True,
                          title = 'Normalized confusion matrix')

    plt.show()
    
# this function is 'plot' the relevant confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize = True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Create the model for the build of the multi view model
def createCnn():    
    num_channels = 12
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(num_channels, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 1)))
    cnn.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    cnn.add(keras.layers.Conv2D(num_channels * 2, (3, 3), activation = 'relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Dropout(0.25))
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(128, activation = 'relu'))
    return cnn

# In[136]:
# Split layer class
import keras
import tensorflow as tf

class SplitLayer(keras.layers.Layer):
    """
    Layer expects a tensor (multi-dimensonal array) of shape (samples, views, ...)
    and returns a list of #views elements, each of shape (samples, ...)
    """
    def __init__(self, num_splits, **kwargs):
        self.num_splits = num_splits
        super(SplitLayer, self).__init__(**kwargs)
    
    def call(self, x):
        return [x[:, i] for i in range(self.num_splits)]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],) + input_shape[2:]]*self.num_splits

# In[137]:
# Load modelnet10_train samples & labels
modelnet10_train = np.load('model//modelnet10_train.npz')
modelnet10_val = np.load('model//modelnet10_val.npz')

samplesTrain = modelnet10_train['samples']
labelsTrain = modelnet10_train['labels']

samples = modelnet10_val['samples']
samplesTrain = modelnet10_train['samples']

# In[138]:
# Plots
plot_depth_map(4, 1, 0)

# In[139]:
plot_depth_map(4, 1, 1) # Mirror

# In[140]:
plot_depth_map(39, 3, 0)

# In[141]:
plot_depth_map(39, 3, 1)

# In[142]:
plot_depth_map(19, 6, 0)

# In[143]:
plot_depth_map(19, 6, 1)

# In[144]:
plot_depth_map(79, 12, 0)

# In[147]:
# Different "models"
def multiViewCnn(epochs, num_of_angles, random, conf_mat):
    num_classes = 10
    input = keras.layers.Input(num_of_angles, 32, 32, 1)
    views = SplitLayer(num_of_angles)(input)
    out = []
    cnn = createCnn() # Create model

    for v in views:
        out.append(cnn(v))

    single_vector = keras.layers.Maximum()(out)
    x = Dropout(0.25)(single_vector)
    x = keras.layers.Dense(128)(x)
    x = Dense(num_classes, activation = 'softmax')(x)
    model = keras.models.Model(input, x)

    # "Put" data to variables
    samples = modelnet10_val['samples']
    samplesTrain = modelnet10_train['samples']

    labels = modelnet10_val['labels']
    labelsTrain = modelnet10_train['labels']

    if random == 'true':
        samplesData = CreateDmdsWithRandomAngle(samples, num_of_angles)
        samplesDataTrain = CreateDmdsWithRandomAngle(samplesTrain, num_of_angles)
    else:
        samplesData = CreateDmdsWithAngle(samples, num_of_angles)
        samplesDataTrain = CreateDmdsWithAngle(samplesTrain, num_of_angles)
    print("samples shape: %s, labels shape: %s" % (samples.shape, labels.shape))

    (x_train, y_train) = (samplesDataTrain, labelsTrain)
    (x_test, y_test) = (samplesData, labels)
    batch_size = 128
    num_classes = 10

    # Input image dimensions
    img_rows, img_cols = 32, 32

    # The data, shuffled and split between train and test sets
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], num_of_angles, 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], num_of_angles, 1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], num_of_angles, img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], num_of_angles, img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1], '\ndone with', num_of_angles, 'angles')
    if random == 'false':
        print ('with regular angles \n')
    else:
        print('with random angles \n')
        
    if conf_mat == 'true':
        conf_matrix('WithRandom', num_of_angles, random) # plot the relevant conf. mat.

# In[148]:
# the split of the multiview, create the model, compile, run, and print the
# scores ...
# suppose to reaches to 50% and above with 12 + epochs
# this functions recieve an input of the epochs and the number
# of the of angles to rotate by.
# with option to random angles, if random == 'true' it will give random angles
def multiViewCnnMirror(epochs, conf_mat):

    num_classes = 10
    
    input = keras.layers.Input(2, 32, 32, 1)
    views = SplitLayer(2)(input)
    out = []
    cnn=createCnn() # create the model
    for v in views:
        out.append(cnn(v))

    single_vector = keras.layers.Maximum()(out)
    x = Dropout(0.25)(single_vector)
    x = keras.layers.Dense(128)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(input, x)


    # 'put' data to variables

    samples = modelnet10_val['samples']
    samplesTrain = modelnet10_train['samples']

    labels = modelnet10_val['labels']
    labelsTrain = modelnet10_train['labels']

    samplesData = CreateDmdsWithMirror(samples)
    samplesDataTrain = CreateDmdsWithMirror(samplesTrain)
    
    print('samples shape: %s, labels shape: %s' % (samples.shape, labels.shape))

    (x_train, y_train)=(samplesDataTrain, labelsTrain)
    (x_test, y_test) = (samplesData, labels)
    batch_size = 128
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the data, shuffled and split between train and test sets


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 2, 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 2, 1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], 2, img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], 2, img_rows, img_cols, 1)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1], '\ndone with mirror\n')

    if (conf_mat == 'true'):
        conf_matrix('mirror', 2, 'false') # plot the relevant conf. mat.

# In[149]:
# function
# input of number of angles
# and input of number of epochs
# to the code of assignment 1
def ass1multiArrayViewCnn(epochs, num_of_angles, conf_mat):

    num_classes = 10

    input = keras.layers.Input(num_of_angles, 32, 32, 1)
    views = SplitLayer(num_of_angles)(input)
    out = []
    cnn = createCnn() # create the model
    for v in views:
        out.append(cnn(v))

    single_vector = keras.layers.Maximum()(out)
    x = Dropout(0.25)(single_vector)
    x = keras.layers.Dense(128)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(input, x)

    # 'put' data to variables
    samples = modelnet10_val['samples']
    samplesTrain = modelnet10_train['samples']

    labels = modelnet10_val['labels']
    labelsTrain = modelnet10_train['labels']

    samplesData = CreateDmdsWithNoAngle(samples, num_of_angles)
    samplesDataTrain = CreateDmdsWithNoAngle(samplesTrain, num_of_angles)
    
    print("samples shape: %s, labels shape: %s" % (samples.shape, labels.shape))

    (x_train, y_train) = (samplesDataTrain, labelsTrain)
    (x_test, y_test) = (samplesData, labels)
    batch_size = 128
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the data, shuffled and split between train and test sets

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0],num_of_angles, 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], num_of_angles, 1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], num_of_angles, img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], num_of_angles, img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1], '\ndone with', num_of_angles,'angles')
    
    if (conf_mat == 'true'):
        conf_matrix('mulVcnn', num_of_angles, 'false') # plot the relevant conf. mat.

# In[150]:
# function of the assignment 1
# input: gets number of opochs (iteration)
# and it 'check' and print the results of iteration on 1 angle (no rotate)
def one_view_ass1_with_angle(epochs, angle, conf_mat):

    # data is already loaded
    labels = modelnet10_val['labels']
    labelsTrain = modelnet10_train['labels']
    
    # create depth map's and 'check them'
    samplesData = CreateDmdsWithSimpleAngle(samples, angle)
    samplesDataTrain = CreateDmdsWithSimpleAngle(samplesTrain, angle)

    print("samples shape- ", samples.shape, "labels shape- ", labels.shape) # changed
    batch_size = 128
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, shuffled and split between train and test sets
    (x_train, y_train) = (samplesDataTrain, labelsTrain)
    (x_test, y_test) = (samplesData, labels)

    if K.image_data_format() == 'channels_first':   
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3),
    activation = 'relu',
    input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adadelta(),
                  metrics = ['accuracy'])

    model.fit(x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('rotated by', angle*30, 'degrees')
    
    if (conf_mat == 'true'):
        conf_matrix('one_v_a', angle, 'false') # plot the relevant conf. mat.

# In[151]:
# function of assignment 1
# input: gets number of epochs (iteration)
# and it 'check' and print the results of iteration on 1 angle (no rotate)
def one_view_ass1(epochs, conf_mat):
    # data is already loaded
    labels = modelnet10_val['labels']
    labelsTrain = modelnet10_train['labels']

    samples = modelnet10_val['samples']
    samplesTrain = modelnet10_train['samples']
    
    # create dlabels = modelnet10_val['labels']epth map's and 'check them'
    samplesData = CreateDMDS_regular(samples)
    samplesDataTrain = CreateDMDS_regular(samplesTrain)

    print("samples shape- ", samples.shape, "labels shape- ", labels.shape) # changed
    batch_size = 128
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, shuffled and split between train and test sets
    (x_train, y_train) = (samplesDataTrain, labelsTrain)
    (x_test, y_test) = (samplesData, labels)

    if K.image_data_format() == 'channels_first':   
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3),
    activation = 'relu',
    input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adadelta(),
                  metrics = ['accuracy'])

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # a code to print a confusion matrix
    if (conf_mat == 'true'):
        conf_matrix('reg', 1, 'false') # plot the relevant conf. mat.

# In[140]:
# in this 'section' you can see our different 'checks'
# of different (4) kinds of 'models' with different (#) epochs
# different (#) angles, and with random angles or regular ones
# you also can see the results that we got if you 'open'
# the 'box' after each 'In[*]'

# put 'true' at the last parameter if you want to add a confusion matrix
# put 'true' at the middle param. if you want a random angles
# left param. is the number of epochs
# second left param. is the number of angles (views)

# this 'checks' below is without conf. mat. and were 'printed'
# in the 'middle' of our work, so we didn't touch them
# in order to keep the results
multiViewCnn(12, 6, 'true', 'false')

# In[74]:
multiViewCnn(12, 6, 'false', 'false')

# In[75]:
multiViewCnn(12, 3, 'true', 'false')

# In[76]:
multiViewCnn(12, 3, 'false', 'false')

# In[77]:
multiViewCnn(12, 12, 'true', 'false')

# In[81]:
multiViewCnn(12, 12, 'false', 'false')

# In[41]:
multiViewCnn(12, 24, 'false', 'false')

# In[42]:
multiViewCnn(12, 24, 'true', 'false')

# In[17]:
multiViewCnn(24, 3, 'true', 'false')

# In[18]:
multiViewCnn(24, 3, 'true', 'false')

# In[19]:
multiViewCnn(24, 6, 'false', 'false')

# In[20]:
multiViewCnn(24, 6, 'true', 'false')

# In[21]:
multiViewCnn(24, 12, 'false', 'false')

# In[22]:
multiViewCnn(24, 12, 'true', 'false')

# In[23]:
multiViewCnn(48, 24, 'false', 'false')

# In[162]:
# this is the a model with one angle rotated
# but with a large data set
# (with a 'confusion matrices')
ass1multiArrayViewCnn(12, 12, 'true')

# In[40]:
one_view_ass1(12, 'false') # will display the result of assignment 1

# In[92]:
one_view_ass1(48, 'false')

# In[98]:
# 12 view, model for each view (use the code of assignment 1)
for i in range(12):
    one_view_ass1_with_angle(12, i, 'false')

# In[44]:
multiViewCnn(96, 48, 'false', 'false')

# In[24]:
multiViewCnnMirror(12, 'false') # the 'mirror' bonus (12 epochs)

# In[25]:
multiViewCnnMirror(24, 'false') # the 'mirror' bonus

# In[26]:
multiViewCnnMirror(48, 'false')

# In[84]:
# now, 'check' for the 'confusion matrix', with visualization
# visualization of the 'normalized' matrix and the 'regular' matrix
one_view_ass1(12, 'true') # our check for plot conf. matrix

# In[156]:
# now, we will run some functions to see the results of the
# confusion matrix of different model
multiViewCnn(12, 3, 'false', 'true')

# In[160]:
multiViewCnn(12, 3, 'true', 'true')

# In[157]:
multiViewCnnMirror(12, 'true')

# In[158]:
ass1multiArrayViewCnn(12, 3, 'true')

# In[159]:
for i in range(12):
    one_view_ass1_with_angle(12, i, 'true')

