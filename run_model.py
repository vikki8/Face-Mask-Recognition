# installing the required packages
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# The important OpenCV, Face detection, and Mask detection instances that are required
video_capture = cv2.VideoCapture(0)
mask_model = load_model("mask_detector.model")
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


# The mask detection operation is handled by the predictions() function, which takes the detected face co-ordinates
# (detected using OpenCV) and the captured frame and returns a tuple containing the mask and non-mask scores.

def predictions(face_given, frame):
    (x, y, w, h) = face_given

    real_face = frame[y:y + h, x:x + w]

    real_face = cv2.cvtColor(real_face, cv2.COLOR_BGR2RGB)
    real_face = cv2.resize(real_face, (224, 224))
    real_face = img_to_array(real_face)
    real_face = preprocess_input(real_face)
    real_face = np.expand_dims(real_face, axis=0)

    prediction = mask_model.predict(real_face)

    return prediction


# The detect faces() method uses OpenCV to do face detection and returns the co-ordinates of the discovered face.
# It takes a frame as an argument and returns the co-ordinates of the detected face.


def detect_faces(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    real_faces = cascade_face.detectMultiScale(gray_image, 1.1, 4)

    return real_faces

# The main while loop is handled by the main method, which is also the driver function.
def main():
    while True:

        frame = video_capture.read()[1]
        danger = 0
        no_risk = 0

        # Scaling is a technique for increasing the number of frames per second (FPS).
        scale_percent = 80  # The percentage of the original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        h_f, w_f = frame.shape[0:2]

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        faces = detect_faces(frame)

        outer_frame_color = (0, 255, 50)

        for face in faces:
            (x, y, w, h) = face

            # get the predictions
            pred = predictions(face, frame)

            [[clear, risk]] = pred

            # generate the labels
            label = "Clear !" if clear > risk else "Possible Risk !"
            color = (0, 255, 0) if label == "Clear !" else (0, 0, 255)

            if label == "Possible Risk !":
                outer_frame_color = (0, 0, 255)
                cv2.putText(frame, "Possible Threat !", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                danger += 1
            else:
                no_risk += 1

            # Taking care of the face rectangle and the colour of the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # dealing with the frame rectangle as well as other statistics data
        cv2.rectangle(frame, (0, 0), (w_f - 130, h_f - 100), outer_frame_color, 6)
        cv2.putText(frame, "Detected Faces : " + str(len(faces)), (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 222), 2)
        cv2.putText(frame, "Positive Threat Count : " + str(danger), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Negative Threat Count : " + str(no_risk), (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 2)

        cv2.imshow("Face Mask Detector - 17206277", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
