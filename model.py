# Root cause analysis

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# read the data
data = pd.read_csv("root_cause_analysis.csv")

# Convert the categorical target variable to numerical
label_encoder = preprocessing.LabelEncoder()
data['ROOT_CAUSE'] = label_encoder.fit_transform(
                                data['ROOT_CAUSE'])

# Convert dataframe to numpy array
data = data.to_numpy().astype(float)

# Divide the data to features and target
data_features = data[:, 1:8]
data_target = data[:, 8]

# Convert traget to one-hot encoding
data_target = tf.keras.utils.to_categorical(data_target, 3)

# Split the data into train and test
#Split training and test data
X_train,X_test,Y_train,Y_test = train_test_split( 
    data_features, data_target, test_size=0.10)

# Create the deep learning model
# No of classes in the target variable
NB_CLASSES = 3

# Create a sequential model in keras
model = tf.keras.models.Sequential()

# Add the 1st hidden layer
model.add(keras.layers.Dense(128,                  # No of nodes
                             input_shape=(7,),      # No of input variables
                             name = "Hidden-Layer-1", # Logical name
                             activation="relu"))        # activation function
# Add the 2nd hidden layer
model.add(keras.layers.Dense(128, 
                             name = "Hidden-Layer-2",
                             activation = "relu"))

# Output Layer
model.add(keras.layers.Dense(NB_CLASSES,
                             name = "Output-Layer",
                             activation="softmax"))

# Compile the model with loss and metrics
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Make the model Verbose so we can see the progress
VERBOSE = 1

# Setup Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2

print("\n Training in progress :\n-----------------------------------")

# Train the model
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs = EPOCHS, verbose=VERBOSE, 
                    validation_split=VALIDATION_SPLIT)

print("\n accuracy during training :\n-----------------------------------")

# Plot the accurcay changes
pd.DataFrame(history.history)["accuracy"].plot(figsize = (9,6))
plt.title("Accuracy improvements with epochs")
plt.show()

# Evaluate against test data
print("\nEvaluation against Test Dataset :\n------------------------------------")
model.evaluate(X_test,Y_test)

#Pass individual flags to Predict the root cause
import numpy as np

CPU_LOAD=1
MEMORY_LOAD=0
DELAY=0
ERROR_1000=0
ERROR_1001=1
ERROR_1002=1
ERROR_1003=0

prediction=np.argmax(model.predict(
    [[CPU_LOAD,MEMORY_LOAD,DELAY,
      ERROR_1000,ERROR_1001,ERROR_1002,ERROR_1003]]), axis=1 )

print(label_encoder.inverse_transform(prediction))

#Predicting as a Batch
print(label_encoder.inverse_transform(np.argmax(
        model.predict([[1,0,0,0,1,1,0],
                                [0,1,1,1,0,0,0],
                                [1,1,0,1,1,0,1],
                                [0,0,0,0,0,1,0],
                                [1,0,1,0,1,1,1]]), axis=1 )))


# Save the model for future 
#Saving a model
    
model.save("root_cause")
    
#Loading a Model 
loaded_model = keras.models.load_model("root_cause")

#Print Model Summary
loaded_model.summary()






