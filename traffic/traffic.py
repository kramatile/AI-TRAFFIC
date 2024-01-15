import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # declare the lists to return : 
    images = []
    label = []
    # loop over the directories : 
    for i in range(NUM_CATEGORIES) :
        images_dir = os.listdir(os.path.join('.',data_dir,str(i)))
        for image in images_dir :
            # read the img using cv2
            img = cv2.imread(os.path.join('.',data_dir,str(i),image))
            if img.shape[0] != IMG_HEIGHT or img.shape[1] != IMG_WIDTH :
                img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH), interpolation = cv2.INTER_AREA)
            # stock it as a multi-dimetionnal array the type of the img is already a multidimentional array 
            images.append(img)
            label.append(i)
    # return the final lists (hopefully)
    return (images,label)



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([  
        # normalize the data
        tf.keras.layers.Rescaling( 
            1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            ),

        # convolution layer to find patterns
        tf.keras.layers.Conv2D(
            32, (3,3), activation='relu', input_shape= (IMG_HEIGHT,IMG_WIDTH,3)
            ),
        
        # reduce the size
        tf.keras.layers.MaxPooling2D(
            pool_size=(3,3)
            ),

        tf.keras.layers.Conv2D(
            64, (3,3), activation='relu', input_shape= (IMG_HEIGHT,IMG_WIDTH,3)
            ),
        
        # reduce the size
        tf.keras.layers.MaxPooling2D(
            pool_size=(3,3)
            ),    

        # deploy the units
        tf.keras.layers.Flatten(),


        #another layer
        tf.keras.layers.Dense(128,activation='relu'),

        #DROPOUTS SO WE DONT RELY ON ANY PARTICULAR NEURONE
        tf.keras.layers.Dropout(0.2),

        # FINAL LAYER
        tf.keras.layers.Dense(NUM_CATEGORIES,activation='softmax')    
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    return model
    raise NotImplementedError


if __name__ == "__main__":
    main()
