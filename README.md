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

And all the preprocessing of data is done inside the **image_generator.py** and is used to load the dataset images for training.  

The file **trial_model.h5** contain weights of trained model on the training data (75% of total dataset) after 20 epochs which gave an 
accuracy of 88% on training set and 79% on test set.  

The file **alexnet_model.h5** contain weights of trained model on the training data (75% of total dataset) after 20 epochs which gave an 
accuracy of 78% on training set and 80% on test set.  

The file **main.py** takes the video feed using Opencv. It uses the trail_model.h5 to get a pretrained model. and predicts the face emotion in real time.You can also use alexnet_model.h5.  

If you wish to train the train the model again , separate file for each model is available (trialmodel.py, alexnet.py), running those files will start training the model and new weights will be saved to the .h5 file of that particulat model.  

 
## Run program
```
Clone the repository.  
Run the file main.py to see face emotion predictions on your face.
```

