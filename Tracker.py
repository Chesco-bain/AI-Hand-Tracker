import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
#https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
#the above link was used to download the hand tracking model
#creating the base configuration
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#configuring the hand landmarker with these specific settings:
# it should detect 2 hands (you can drop it to one if you want)
#80% confidence  required to make a detection
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2, 
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

#hand detector object
detector = vision.HandLandmarker.create_from_options(options)

#opening a camera
cap = cv2.VideoCapture(0)


while cap.isOpened():
    #reading the frames
    ret, frame = cap.read()
    if not ret:
        break  #if the reading fails, stop

    # Create a mirror (by flipping the camera)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape #get the height and width of the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting image format from BGR to RGB for mediapipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    #runs the AI model on the image
    results = detector.detect(mp_image)

    #if hands were detected
    if results.hand_landmarks:
        #loops though each detected hand
        for hand_landmarks in results.hand_landmarks:
            # Getting the index finger (check mediapip documentation for landmark points)
            index_tip = hand_landmarks[8]

            # Getting the x and y co-ords 
            x= int(index_tip.x * w)
            y=int(index_tip.y * h)
            #drawing a circle on the index finger(s)
            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
    #show the processed frame with drawings
    cv2.imshow("index finger track", frame)

    # Quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




