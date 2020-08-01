import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, InputLayer, Dense, BatchNormalization, Flatten, ZeroPadding2D,Activation

#trial model containing (conv layer, batchnorm layer, activation , maxpool layer)x2 
def load_trial_model(inputshape, num_outputs):
    model = Sequential([InputLayer(input_shape= inputshape),
                    
                    Conv2D(32,(5,5),strides = (2,2)),
                    BatchNormalization(axis = 3),
                    Activation('relu'),
                    MaxPool2D((2, 2)),

                    Conv2D(64,(5,5), strides = (2,2)),
                    BatchNormalization(axis = 3),
                    Activation('relu'),
                    MaxPool2D((2, 2)),

                    Flatten(),
                    Dense(512,activation = "relu"),
                    Dense(num_outputs,activation = "softmax")
                   ])
    
    #Save the model architecture in an image form
    keras.utils.plot_model(model, to_file='model_architecture.png') #saving the model architecture in an image form
    
    # Compile the model
    model.compile(loss= "categorical_crossentropy",optimizer= "adam",metrics = ["accuracy"])
    

    return model




