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
            #https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker 
            index_tip = hand_landmarks[8]
            # Getting the x and y co-ords 
            x= int(index_tip.x * w)
            y=int(index_tip.y * h)
            #drawing a circle on the index finger(s)
            cv2.circle(frame, (x, y), 10, (0, 0, 0), -1)

            #index dip
            index_dip=hand_landmarks[7]
            index_dip_x=int(index_dip.x*w)
            index_dip_y=int(index_dip.y*h)
            cv2.circle(frame, (index_dip_x, index_dip_y), 10, (0, 0, 0), -1)

            #index pip
            index_pip=hand_landmarks[6]
            index_pip_x=int(index_pip.x*w)
            index_pip_y=int(index_pip.y*h)
            cv2.circle(frame, (index_pip_x, index_pip_y), 10, (0, 0, 0), -1)

            #index mcp
            index_mcp=hand_landmarks[5]
            index_mcp_x=int(index_mcp.x*w)
            index_mcp_y=int(index_mcp.y*h)
            cv2.circle(frame, (index_mcp_x, index_mcp_y), 10, (0, 0, 0), -1)

            #thumb tip
            thumb_tip=hand_landmarks[4]
            thumb_tip_x=int(thumb_tip.x*w)
            thumb_tip_y=int(thumb_tip.y*h)
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 0, 0), -1)

            #thumb ip
            thumb_ip=hand_landmarks[3]
            thumb_ip_x=int(thumb_ip.x*w)
            thumb_ip_y=int(thumb_ip.y*h)
            cv2.circle(frame, (thumb_ip_x, thumb_ip_y), 10, (0, 0, 0), -1)

            #thumb mcp
            thumb_mcp=hand_landmarks[2]
            thumb_mcp_x=int(thumb_mcp.x*w)
            thumb_mcp_y=int(thumb_mcp.y*h)
            cv2.circle(frame, (thumb_mcp_x, thumb_mcp_y), 10, (0, 0, 0), -1)

            #thumb cmc
            thumb_cmc=hand_landmarks[1]
            thumb_cmc_x=int(thumb_cmc.x*w)
            thumb_cmc_y=int(thumb_cmc.y*h)
            cv2.circle(frame, (thumb_cmc_x, thumb_cmc_y), 10, (0, 0, 0), -1)

            #wrist
            wrist=hand_landmarks[0]
            wrist_x=int(wrist.x*w)
            wrist_y=int(wrist.y*h)
            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 0), -1)


            #middle tip
            middle_tip=hand_landmarks[12]
            middle_tip_x=int(middle_tip.x*w)
            middle_tip_y=int(middle_tip.y*h)
            cv2.circle(frame, (middle_tip_x, middle_tip_y), 10, (0, 0, 0), -1)

            #middle dip
            middle_dip=hand_landmarks[11]
            middle_dip_x=int(middle_dip.x*w)
            middle_dip_y=int(middle_dip.y*h)
            cv2.circle(frame, (middle_dip_x, middle_dip_y), 10, (0, 0, 0), -1)

            #middle pip
            middle_pip=hand_landmarks[10]
            middle_pip_x=int(middle_pip.x*w)
            middle_pip_y=int(middle_pip.y*h)
            cv2.circle(frame, (middle_pip_x, middle_pip_y), 10, (0, 0, 0), -1)

            #middle mcp
            middle_mcp=hand_landmarks[9]
            middle_mcp_x=int(middle_mcp.x*w)
            middle_mcp_y=int(middle_mcp.y*h)
            cv2.circle(frame, (middle_mcp_x, middle_mcp_y), 10, (0, 0, 0), -1)

            #ring tip
            ring_tip=hand_landmarks[16]
            ring_tip_x=int(ring_tip.x*w)
            ring_tip_y=int(ring_tip.y*h)
            cv2.circle(frame, (ring_tip_x, ring_tip_y), 10, (0, 0, 0), -1)

            #ring dip
            ring_dip=hand_landmarks[15]
            ring_dip_x=int(ring_dip.x*w)
            ring_dip_y=int(ring_dip.y*h)
            cv2.circle(frame, (ring_dip_x, ring_dip_y), 10, (0, 0, 0), -1)

            #ring pip
            ring_pip=hand_landmarks[14]
            ring_pip_x=int(ring_pip.x*w)
            ring_pip_y=int(ring_pip.y*h)
            cv2.circle(frame, (ring_pip_x, ring_pip_y), 10, (0, 0, 0), -1)

            #ring mcp
            ring_mcp=hand_landmarks[13]
            ring_mcp_x=int(ring_mcp.x*w)
            ring_mcp_y=int(ring_mcp.y*h)
            cv2.circle(frame, (ring_mcp_x, ring_mcp_y), 10, (0, 0, 0), -1)






            #connecting the hand landmarks (refer to the documentation for the numbers)
            #thumb
                            #(x,y of thumb tip)        (x,y of thumb ip)
            cv2.line(frame, (thumb_tip_x, thumb_tip_y), (thumb_ip_x, thumb_ip_y), (0, 255, 0), 2)  #4 - 3
            cv2.line(frame, (thumb_ip_x, thumb_ip_y), (thumb_mcp_x, thumb_mcp_y), (0, 255, 0), 2)  #3 - 2
            cv2.line(frame, (thumb_mcp_x, thumb_mcp_y), (thumb_cmc_x, thumb_cmc_y), (0, 255, 0), 2)  #2 - 1
            cv2.line(frame, (thumb_cmc_x, thumb_cmc_y), (wrist_x, wrist_y), (0, 255, 0), 2)  #1 - 0
            cv2.line(frame, (thumb_cmc_x, thumb_cmc_y), (index_mcp_x, index_mcp_y), (0, 255, 0), 2)  #1 - 5

            #index
            cv2.line(frame, (x, y), (index_dip_x, index_dip_y), (0, 255, 0), 2)  #8 - 7
            cv2.line(frame, (index_dip_x, index_dip_y), (index_pip_x, index_pip_y), (0, 255, 0), 2)  #7 - 6
            cv2.line(frame, (index_pip_x, index_pip_y), (index_mcp_x, index_mcp_y), (0, 255, 0), 2)  #6 - 5
            cv2.line(frame, (index_mcp_x, index_mcp_y), (wrist_x, wrist_y), (0, 255, 0), 2)  #5 - 0

            #middle
            cv2.line(frame, (middle_tip_x, middle_tip_y), (middle_dip_x, middle_dip_y), (0, 255, 0), 2)  #12 - 11
            cv2.line(frame, (middle_dip_x, middle_dip_y), (middle_pip_x, middle_pip_y), (0, 255, 0), 2)  #11 - 10
            cv2.line(frame, (middle_pip_x, middle_pip_y), (middle_mcp_x, middle_mcp_y), (0, 255, 0), 2)  #10 - 9
            cv2.line(frame, (index_mcp_x, index_mcp_y), (middle_mcp_x, middle_mcp_y), (0, 255, 0), 2)  #5 - 9
            cv2.line(frame, (wrist_x, wrist_y), (middle_mcp_x, middle_mcp_y), (0, 255, 0), 2)  #0 - 9

            #ring
            cv2.line(frame, (ring_tip_x, ring_tip_y), (ring_dip_x, ring_dip_y), (0, 255, 0), 2)  #16 - 15
            cv2.line(frame, (ring_pip_x, ring_pip_y), (ring_dip_x, ring_dip_y), (0, 255, 0), 2)  #14 - 15
            cv2.line(frame, (ring_pip_x, ring_pip_y), (ring_mcp_x, ring_mcp_y), (0, 255, 0), 2)  #14 - 13
            cv2.line(frame, (middle_mcp_x, middle_mcp_y), (ring_mcp_x, ring_mcp_y), (0, 255, 0), 2)  #9 - 13
            cv2.line(frame, (wrist_x, wrist_y), (ring_mcp_x, ring_mcp_y), (0, 255, 0), 2)  #0 - 13









    #show the processed frame with drawings
    cv2.imshow("index finger track", frame)

    # Quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




