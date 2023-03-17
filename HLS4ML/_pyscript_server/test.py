import os
import tempfile
import zipfile
import hls4ml
import tensorflow as tf
import time
from datetime import datetime
from os import path
from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers       import Activation, Flatten, MaxPool2D, Dense, Conv2D, Input, Dropout
from tensorflow.keras.models import load_model, Sequential
import tensorflow_model_optimization as tfmot
import numpy as np
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from keras.utils.vis_utils import plot_model
#import hls4ml.model.profiling
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# create HLS_projects folder in current working directory
USER_PATH    = os.getcwd()
DATA_PATH    = USER_PATH + "/Data/"
MODEL_PATH   = USER_PATH + "/Models/"
PROJECT_PATH = USER_PATH + "/HLS_projects/"
PLOTS_PATH   = USER_PATH + "/Plots/"

#Training parameters
training_epochs	= 15
batch_size = 128
validation_split = 0.1 # 10% of training set will be used for validation set

# Model / data parameters
input_shape = (28,28,1)
kernel_size = (3,3)
filter_num	= 12
pool_size = (2,2)
num_classes = 10

hls4ml_description = """

        ╔╧╧╧╗────o
    hls ║ 4 ║ ml   - Machine learning inference in FPGAs
   o────╚╤╤╤╝
"""


#**************************MODELS DEFINITION************************************************************************************************************

def CNN_simple():
    model = Sequential()
    model.add(Input(input_shape, name='layer0'))
    model.add(Conv2D(filter_num, kernel_size, activation='relu', name='convolution_layer_1'))
    model.add(MaxPool2D(pool_size, name='max_pooling_layer_1'))
    model.add(Flatten(name='flatten_layer')),
    model.add(Dense(num_classes, name='dense_layer')),
    model.add(Activation('softmax', name="softmax"))
    #Print summary
    model.summary()
    # Return model
    return model

# define CNN model
def CNN_complex():
    model = Sequential()
    model.add(Input(input_shape, name='layer0'))
    model.add(Conv2D(filter_num, kernel_size, activation='relu', name='convolution_layer_1'))
    model.add(MaxPool2D(pool_size, name='max_pooling_layer_1'))
    model.add(Conv2D(64, kernel_size, activation="relu",  name='convolution_layer_2')),
    model.add(MaxPool2D(pool_size, name='max_pooling_layer_2')),
    model.add(Flatten(name='flatten_layer')),
    model.add(Dropout(0.5, name="dropout_layer")),
    model.add(Dense(num_classes, name='dense_layer')),
    model.add(Activation('softmax', name="softmax"))
    #Print summary
    model.summary()
    # Return model
    return model

# define LeNet-5 model
def LeNet5():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation='relu', input_shape=(28,28,1)))  # Convolutional Layer 1
    model.add(MaxPool2D(strides=(2,2)))  # Max Pooling Layer 1
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu')) # Convolutional Layer 2
    model.add(MaxPool2D(strides=(2,2)))  # Max Pooling Layer 2
    model.add(Flatten())  # Flatenning Layer
    model.add(Dense(units=256, activation='relu')) # Fully Connected Layer 1
    model.add(Dense(units=84, activation='relu'))   # Fully Connected Layer 2
    model.add(Dense(units=10))   # Fully Connected Layer 3
    model.add(Activation("softmax", name="softmax"))
    #Print summary
    model.summary()
    # Return model
    return model

#******************************************************************************************************************************************************

def pruned_CNN(model, train_images, train_labels):
    #Compute end step to finish pruning after 2 epochs.
    epochs = 2
 
    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    model_for_pruning.summary()


    callbacks = [ tfmot.sparsity.keras.UpdatePruningStep() ]

    model_for_pruning.fit(train_images, train_labels,
                    batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                    callbacks=callbacks)
    model_for_export= tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.save( MODEL_PATH + 'my_CNN/CNN_simple_pruned.h5')

    print("Size of the model: " + str(get_gzipped_model_size(MODEL_PATH + 'my_CNN/CNN_simple_pruned.h5' ))+" bytes")
   
    return model_for_export


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX,testX =prep_pixels(trainX, testX)
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY,10)
    testY = to_categorical(testY,10)
   
    print("Datasets loaded! \n")
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# Returns size of gzipped model in bytes
def get_gzipped_model_size(file):
  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)
  return os.path.getsize(zipped_file)


def layer_size (model):
        #Lets check if this model can be implemented completely unrolled (For the lowest possible latency, each layer should have a maximum number of trainable parameters of 4096)
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense' ]:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
            if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
                print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))
    print("Total number of parameters: ", model.count_params())

def train (trainX, trainY):
    if not path.isfile(MODEL_PATH + 'my_CNN/CNN_simple.h5'):
        print('Model not found: create and train it')
        #Define the model
        model = CNN_simple()
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        layer_size(model)
        with tf.device('/cpu:0'):
            start = time.time()
            model.fit(trainX, trainY, batch_size=batch_size, epochs=training_epochs, validation_split=validation_split, shuffle=True, verbose = 1)
            end = time.time()
            print('\n It took {} minutes to train!\n'.format( (end - start)/60.))
            model.save( MODEL_PATH + 'my_CNN/CNN_simple.h5')
    else:
        print('Found a Keras model (.h5): loading it \n')
        model = load_model( MODEL_PATH + 'my_CNN/CNN_simple.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        layer_size(model)
    
    print("Size of the Keras model: " + str(get_gzipped_model_size(MODEL_PATH + 'my_CNN/CNN_simple.h5') * 10 ** (-6))+" MB")
    return model




def config_hls4ml(model,testX):
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'], rounding_mode='AP_RND', saturation_mode='AP_SAT')
    Trace = True
    '''
    plots=hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=testX[:1000]) #-> Non funziona (dynamic library problem)
    for i,plot in enumerate(plots):
        plot.savefig(f'Plots/hls4mlPlots{i}.png')
    '''
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = 1
    config['Model']['Strategy'] = 'Resource'
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    
    #config['LayerName']['convolution_layer_2']['ReuseFactor'] = 6912

    config['LayerName']['layer0']['Precision'] = 'ap_ufixed<8,3>'

    config['LayerName']['dense_layer']['ReuseFactor'] = 13
    config['LayerName']['softmax']['Strategy']    = 'Stable'

    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True


    cfg = hls4ml.converters.create_config(board='alveo-u280', clock_period= 5, part='xcu280-fsvh2892-2L-e', backend='VivadoAccelerator')
    cfg['HLSConfig'] = config

    cfg['AcceleratorConfig']['Driver']    = 'python'
    cfg['AcceleratorConfig']['Board']     = 'alveo-u280'
    cfg['AcceleratorConfig']['Interface'] = 'axi_stream'
    cfg['AcceleratorConfig']['Precision']['Input']  = 'float'
    cfg['AcceleratorConfig']['Precision']['Output'] = 'float'

    cfg['IOType']= 'io_stream'
    cfg['KerasModel'] = model
    cfg['OutputDir'] = PROJECT_PATH + 'my_CNN6/hls4ml_prj'
   
    
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=[])
    #hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=testX[:1000])
    '''
    for i,plot in enumerate(plots):
        plot.savefig(f'Plots/hls4mlPlots{i}.png') 
    '''
    return hls_model


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def predict_image(model):
    #load txt image
    print("Loading the image...")
    img = load_image(DATA_PATH +"image_8.png")

    # predict the class
    predict_value = model.predict(img)
    digit = argmax(predict_value)
    print("The predicted value is: ", str(digit))
    print()


def _print_dt(timea, timeb, N):
    dt = (timeb - timea)
    dts = dt.seconds + dt.microseconds * 10 ** -6
    rate = N / dts
    print("Classified {} samples in {} seconds ({} inferences / s)".format(N, dts, rate))
    print("Or {} us / inferences".format(1 / rate * 1e6))
    return dts, rate



def main():
    
    #Download data from MNIST dataset
    trainX, trainY, testX, testY = load_dataset()
    print("Shapes of MNIST datasets:")
    print('trainX.shape = ', trainX.shape)
    print('trainY.shape = ', trainY.shape)
    print('testX.shape = ', testX.shape)
    print('testY.shape = ', testY.shape)
    print()
    
    '''
    # plot first 6 images of train dataset
    plt.figure(figsize=(20,2))
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(trainX[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(trainY[i])
    plt.savefig(DATA_PATH +"train_dataset_6_samples.jpg")
    '''

    #Define and train the model (if it doesn't exist) else use the pre-existing model 
    model= train(trainX,trainY)


    #Evaluate the Keras model
    timA= datetime.now()
    y_keras = model.predict(testX)
    print("Keras test Accuracy: {}".format(accuracy_score(np.argmax(testY, axis=1), np.argmax(y_keras, axis=1))))
    timB= datetime.now()
    dts, rate = _print_dt(timA,timB,testX.shape[0])
    '''
    #Create the file model.png to save the file 
    plot_model(model, PLOTS_PATH + 'Keras_model.png', show_layer_activations=True, show_shapes=True, show_dtype=True, rankdir='TB', dpi=100)
    

    #Classify the image in input
    predict_image(model)

    #Prune the model
    pruned_model=pruned_CNN(model, trainX, trainY)
    #plot_model(pruned_model, PLOTS_PATH + 'Keras_model_pruned.png', show_layer_activations=True, show_shapes=True, show_dtype=True, rankdir='TB', dpi=100)


    #Comparing between pruned and unpruned accuracy models
    model_ref = load_model(MODEL_PATH +'my_CNN/CNN_simple.h5')
    y_ref = model_ref.predict(testX)
    y_prune = pruned_model.predict(testX)

    print("Keras accuracy unpruned: {}".format(accuracy_score(np.argmax(testY, axis=1), np.argmax(y_ref, axis=1))))
    print("Keras accuracy pruned:   {}".format(accuracy_score(np.argmax(testY, axis=1), np.argmax(y_prune, axis=1))))
    '''

    #_______________________________HLS4ML____________________________________
   
    print(hls4ml_description)

    #hls4ml configuration
    hls_model=config_hls4ml(model, testX)

    #Compile hls4ml model
    hls_model.compile()

    # Predict hls4ml model
    y_hls = hls_model.predict(np.ascontiguousarray(testX))

    # Compare
    print("Keras Test Accuracy: {}".format(accuracy_score(np.argmax(testY, axis=1), np.argmax(y_keras, axis=1))))
    print("hls4ml Test Accuracy: {}".format(accuracy_score(np.argmax(testY, axis=1), np.argmax(y_hls, axis=1))))

    #hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=PLOTS_PATH+"HLS_model.png")
   
    # Synthesize
    hls_model.build(csim=False,synth=True,export=True)

    #Show the Vivado report
    hls4ml.report.read_vivado_report(PROJECT_PATH + 'my_CNN6/hls4ml_prj')

    #Bitstream Generation
    hls4ml.backends.VivadoAcceleratorBackend.make_xclbin(hls_model, 'xilinx_u280_xdma_201920_3')




# Call the Main function
if __name__ == '__main__':
	main()