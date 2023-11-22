from keras.preprocessing.image import ImageDataGenerator

############################################################
################# Step 1: Data Processing ##################
############################################################

# Defining input image shape
input_shape = (100, 100, 3)

# Defining data directories
train_data_dir = './Data/Train'
validation_data_dir = './Data/Validation'
test_data_dir = './Data/Test'

# Seting up data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling validation and test data
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Setting up train, validation, and test generators
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # To keep the order of images for evaluation
)

############################################################
#########  Neural Network Architecture Design ##############
############################################################
