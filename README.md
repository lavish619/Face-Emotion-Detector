# Face-Emotion-Detector using Keras, OpenCV

## Dataset
Face emotion detector model trained on a dataset containing **8 emotions** :
```
['anger','happiness','disgust','contempt','fear','neutral','sadness','surprise']
``` 
### The *facial_expressions* directory contains the images and the csv file of dataset consisting of around 13690 images.



## Processing the Dataset
The Dataset is processed using **ImageDataGenerators** in Keras to acces the images from raw form to be trained.  
Dataset contains images from all categories, however the images of each class label are not equal so the model is little bit more accurate towards 
those labels which are dominant.

## Description of Files

The **load_model.py** file loads the csv file as a dataframe.  

**models.py** file consists of various model architectures that are used to build the model.  

And mostly work is done inside the **model_trainer.ipynb** notebook, from preprocessing of data to evaluating the model on test set.  

The file **trial_model.h5** contain weights of trained model on the training data (75% of total dataset) after 20 epochs which gave an 
accuracy of 88% on training set and 79% on test set.  

The file **main.py** takes the video feed using Opencv. It uses the trail_model.h5 to get a pretrained model. and predicts the face emotion in real time. 


## Run program
```
Clone the repository.  
Run the file main.py to see face emotion predictions on your face.
```

