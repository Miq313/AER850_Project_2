import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.optimizers import Adam

############################################################
################# Step 1: Data Processing ##################
############################################################

# Defining input image shape
input_shape = (100, 100, 3)

# Defining data directories
train_data_dir = './Data/Train'
validation_data_dir = './Data/Validation'
test_data_dir = './Data/Test'

# Setting up data augmentation for training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling validation and test data
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Setting up train, validation, and test datasets using image_dataset_from_directory
batch_size = 32

# Train dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=input_shape[:2],
    batch_size=batch_size,
    label_mode='categorical'
)

# Validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    image_size=input_shape[:2],
    batch_size=batch_size,
    label_mode='categorical'
)

# Test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=input_shape[:2],
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False  # To keep the order of images for evaluation
)

############################################################
###### Step 2: Neural Network Architecture Design ##########
############################################################

# Creating a Sequential model
model = Sequential()

# Convolutional layers with LeakyReLU activation
model.add(Conv2D(32, (3, 3), activation='LeakyReLU', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='LeakyReLU'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='LeakyReLU'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening layer to transition from convolutional to fully connected layers
model.add(Flatten())

# Fully connected layers with ReLU activation
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Optional dropout for regularization

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer with 4 neurons (one for each class) and softmax activation for multi-class classification
model.add(Dense(4, activation='softmax'))

############################################################
############# Network Hyperparameter Analysis ##############
############################################################

# Experiment with hyperparameters
activation_conv = 'relu'
activation_dense = 'relu'  # or 'elu'
filters = 64  # (32, 64, 128, etc.)
neurons_dense = 128  #  (32, 64, 128, etc.)

# Compile the model with hyperparameters
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the summary of the model architecture
model.summary()