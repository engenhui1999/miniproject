import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import cv2

while True:
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    try:
        image = Image.open('faces/face.jpg')


        #turn the image into a numpy array
        image_array = np.asarray(image)
        image_array = np.array([image_array, image_array, image_array])
        image_array = image_array.reshape(224, 224, 3)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        print(prediction)

        os.unlink("faces/face.jpg")

        if prediction[0, 0] > 0.8:
            print("Welcome En Hui")
        else:
            print("Stand in frame")

    except:
        print("No image found")
        pass

