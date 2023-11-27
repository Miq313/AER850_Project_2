
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
import matplotlib.pyplot as plt

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
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling validation and test data
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Setting up train, validation, and test datasets using image_dataset_from_directory
batch_size = 32

# Train dataset
train_dataset = image_dataset_from_directory(
    train_data_dir,
    image_size=input_shape[:2],
    batch_size=batch_size,
    label_mode='categorical'
)

# Validation dataset
validation_dataset = image_dataset_from_directory(
    validation_data_dir,
    image_size=input_shape[:2],
    batch_size=batch_size,
    label_mode='categorical'
)

# Test dataset
test_dataset = image_dataset_from_directory(
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

model.add(Conv2D(256, 2, strides=(1, 1), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, 2, strides=(1, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(64, 4, activation='relu')),  # Changed activation to 'relu
model.add(MaxPooling2D())
model.add(Dropout(0.35))

model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

############################################################
######### Step 3: Network Hyperparameter Analysis ##########
############################################################

# Experimenting with hyperparameters
activation_conv = 'relu'
activation_dense = 'relu'  # or 'elu'
filters = 64  # (32, 64, 128, etc.)
neurons_dense = 128  #  (32, 64, 128, etc.)

# Compiling the model with hyperparameters
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the summary of the model architecture
model.summary()

############################################################
################# Step 4: Model Evaluation #################
############################################################

# Training
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Plotting training history (loss and accuracy over epochs)
plt.figure(figsize=(12, 4))

# Plotting training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plotting training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()