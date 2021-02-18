from appJar import gui
import tensorflow.keras
from PIL import Image
import numpy as np
import os


def retrievedata():
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
            person = "en_hui"
            return person
        else:
            print("Stand in frame")

    except:
        print("No image found")
        person = False
        return person


# handle button events
def press(button):
    if button == "End":
        app.stop()
    elif button == "Check Info":
        person = retrievedata()
        if not person:
            app.infoBox("Information", "No Image Found")
        else:
            info = {}
            with open(f'data/{person}.txt') as input_file:
                for line in input_file:
                    newinfo = line.split(":", 1)
                    i = 0
                    for item in newinfo:
                        newinfo[i] = item.rstrip("\n")
                        i += 1
                    info[newinfo[0]] = newinfo[1]
            app.infoBox("Information", f"Name: {info['name']} \n "
                                       f"Blood: {info['blood']} \n"
                                       f"Allergies: {info['allergies']} \n"
                                       f"Medical Conditions: {info['medical conditions']} \n"
                                       f"Height: {info['height']} \n"
                                       f"Weight: {info['weight']}")


# create a GUI variable called app
app = gui("Login Window", "400x200")
app.setBg("orange")
app.setFont(18)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Retrieval of Information")
app.setLabelBg("title", "blue")
app.setLabelFg("title", "orange")


# link the buttons to the function called press
app.addButtons(["Check Info"], press)
app.addButtons(["End"], press)

# start the GUI
app.go()