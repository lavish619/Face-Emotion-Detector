import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, InputLayer, Dense, BatchNormalization, Flatten, ZeroPadding2D,Activation, Dropout

#trial model containing (conv layer, batchnorm layer, activation , maxpool layer)x2 
def trial_model(inputshape, num_outputs):
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
    keras.utils.plot_model(model, to_file='model architecture images/trial_model.png') #saving the model architecture in an image form
    
    # Compile the model
    model.compile(loss= "categorical_crossentropy",optimizer= "adam",metrics = ["accuracy"])
    

    return model

def alexnet_model(inputshape, num_outputs):
    alexnet_model = Sequential([InputLayer(input_shape= inputshape),
                                
                                #layer1
                                Conv2D(48,(11,11), strides = (4,4)),
                                BatchNormalization(),
                                Activation("relu"),
                                MaxPool2D(pool_size=(3, 3),strides=(2,2)),
                                
                                #layer2
                                Conv2D(128,(5,5),strides = (1,1), padding = "same"),
                                BatchNormalization(),
                                Activation("relu"),
                                MaxPool2D(pool_size=(3, 3), strides=(2,2)),
                                
                                #layer3
                                Conv2D(192,(3,3),strides = (1,1), padding = "same"),
                                #BatchNormalization(),
                                Activation("relu"),
                                
                                #layer4
                                Conv2D(192,(3,3),strides = (1,1), padding = "same"),
                                #BatchNormalization(),
                                Activation("relu"),
                                
                                #layer5
                                Conv2D(128,(3,3),strides = (1,1), padding = "same"),
                                #BatchNormalization(),
                                Activation("relu"),
                                MaxPool2D(pool_size= (3, 3),strides=(2,2)),
                                
                                Flatten(),
                                
                                #layer6
                                Dense(1024, ),
                                BatchNormalization(),
                                Activation("relu"),
                                #Dropout(0.9),
                                
                                #layer7
                                Dense(1024, ),
                                BatchNormalization(),
                                Activation("relu"),
                                #Dropout(0.9),
                                
                                #layer8
                                Dense(num_outputs, ),
                                BatchNormalization(),
                                Activation("softmax"),
                                

    ])

    keras.utils.plot_model(alexnet_model, to_file='model architecture images/alexnet.png') #saving the model architecture in an image form

    alexnet_model.compile(loss= "categorical_crossentropy",optimizer= "adam",metrics = ["accuracy"])
    return alexnet_model
    

# def alexnet_model(img_shape, n_classes=10, l2_reg=0.,
# 	weights=None):

# 	# Initialize model
# 	alexnet = Sequential()

# 	# Layer 1
# 	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
# 		padding='same', kernel_regularizer=l2(l2_reg)))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# 	# Layer 2
# 	alexnet.add(Conv2D(256, (5, 5), padding='same'))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# 	# Layer 3
# 	alexnet.add(ZeroPadding2D((1, 1)))
# 	alexnet.add(Conv2D(512, (3, 3), padding='same'))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# 	# Layer 4
# 	alexnet.add(ZeroPadding2D((1, 1)))
# 	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))

# 	# Layer 5
# 	alexnet.add(ZeroPadding2D((1, 1)))
# 	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# 	# Layer 6
# 	alexnet.add(Flatten())
# 	alexnet.add(Dense(3072))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(Dropout(0.5))

# 	# Layer 7
# 	alexnet.add(Dense(4096))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('relu'))
# 	alexnet.add(Dropout(0.5))

# 	# Layer 8
# 	alexnet.add(Dense(n_classes))
# 	alexnet.add(BatchNormalization())
# 	alexnet.add(Activation('softmax'))

#     keras.utils.plot_model(alexnet, to_file='alexnet.png') #saving the model architecture in an image form


# 	return alexnet





