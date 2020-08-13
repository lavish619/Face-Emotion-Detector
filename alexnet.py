from models import alexnet_model
from image_generator import *

#load the model
model = alexnet_model((350,350,3), num_outputs)

# Train the model
model.fit(train_generator,
          steps_per_epoch=100,
          epochs = 20,
          validation_data = validation_generator
          )

# get accuracy on test set
pred = model.evaluate(test_generator)
print("Accuracy on test set:", pred[1])

# save the trained weights
model.save_weights("alexnet_model.h5")