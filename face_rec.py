import face_recognition
import os
import cv2

KNOWN_DIR = "G:\\OpenCV\\Face_Rec\\known"
UNKNOWN_DIR = "G:\\OpenCV\\Face_Rec\\unknown"
TOLL = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

print("loading known faces")

known_face = []
known_names = []

for name in os.listdir(KNOWN_DIR):
    for filename in os.listdir(f"{KNOWN_DIR}\\{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_DIR}\\{name}\\{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face.append(encoding)
        known_names.append(name)

print("loading unknown faces")

for filename in os.listdir(UNKNOWN_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_DIR}\\{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_face, face_encoding, TOLL)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print("Match Found: ", match)

            top_left = (face_location[3], face_location[0])
            bot_right = (face_location[1], face_location[2])
            color = [255,0,0]
            cv2.rectangle(image, top_left, bot_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bot_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bot_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(155,155,155),FONT_THICKNESS)
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)