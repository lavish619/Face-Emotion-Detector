#Importing Libraries and packages
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(r"./facial_expressions/image_emotion.csv")

print(dataset["emotion"].value_counts())

#Replacing emotion in capital to small
dataset['emotion'] = dataset['emotion'].replace({'NEUTRAL':'neutral','HAPPINESS':'happiness','DISGUST':'disgust',
                                                 'SADNESS':'sadness','ANGER':'anger','SURPRISE':'surprise',
                                                 'FEAR':'fear'})

num_outputs = len(dataset["emotion"].value_counts()) #number of classes
print(dataset["emotion"].value_counts())

#Splitting into train test data
x_train, x_test, y_train, y_test = train_test_split(dataset['image'], dataset['emotion'])

train_data = pd.concat([x_train, y_train],axis = 1)  #training dataframe
test_data = pd.concat([x_test, y_test],axis = 1)  #test dataframe

# Using ImageDataGenerator 
datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2) # rescaling the pixel values between 0 to 1.
test_datagen = ImageDataGenerator(rescale=1./255) # rescaling the pixel values between 0 to 1.

#Access training images from directory marked to their class labels in dataframe
train_generator = datagen.flow_from_dataframe(dataframe = train_data,
                                              directory= "./facial_expressions/images",
                                              x_col="image", y_col = "emotion",
                                              target_size = (350,350),
                                              subset = "training")  

#Access validation images from directory marked to their class labels in dataframe
validation_generator = datagen.flow_from_dataframe(dataframe = train_data,
                                                   directory= "./facial_expressions/images",
                                                   x_col="image", y_col = "emotion",
                                                   target_size = (350,350),
                                                   subset= "validation")

#Access test images from directory marked to their class labels in dataframe
test_generator = test_datagen.flow_from_dataframe(dataframe = test_data,
                                                  directory= "./facial_expressions/images",
                                                  x_col="image", y_col = "emotion",
                                                  target_size = (350,350))

#traget size is chosen (350,350) because maximum images have this dimension.

label_indices = train_generator.class_indices #get indices of each class in one hot encoding done during flow from dataframe
print(label_indices)