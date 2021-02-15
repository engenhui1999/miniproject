import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
anterior = 0


def face_checker(n):
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image_array = np.asarray(n)
    image_array = np.array([image_array, image_array, image_array])
    image_array = image_array.reshape(224, 224, 3)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model(data)
    print(prediction)
    if prediction[0, 0] > 0.8:
        return print("Hello En Hui")
    elif prediction[0, 1] > 0.8:
        return print("Hello Bottle")


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (200, 255, 0), 2)
        roi = frame[y-10:y + h + 10, x - 10:x + w + 10]
        if roi is not [] and x > 10 and y > 10:
            cv2.imshow('Bounding Box', roi)
            face_detected = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            face_detected = Image.fromarray(face_detected)
            face_detected = face_detected.resize((224, 224))
            face_checker(face_detected)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()