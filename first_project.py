import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    #image = Image.open('test_photo.jpg')
    image = frame
    #print("1" + str(image))

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    #size = (224, 224)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = cv2.resize(image, (224, 224))
    # gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image', gray2)
    #print("2" + str(image))
    #image.save('test_photo.jpg')

    #turn the image into a numpy array
    #image_array = np.asarray(image)
    #print("3" + str(image_array))
    #image_array = image

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image.astype(np.float32) / 127.0) - 1
    #normalized_image_array = (image / 127.0) - 1
    #print("4" + str(normalized_image_array))

    # Load the image into the array
    data[0] = normalized_image_array
    #print("5" + str(data[0]))


    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if prediction[0,0] > 0.8:
        print("Hello En Hui")
    elif prediction[0,1] > 0.8:
        print("Hello Bottle")

cap.release()
cv2.destroyAllWindows()