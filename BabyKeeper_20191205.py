# coding=UTF-8
import os
import sys
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import mrcnn.config
import mrcnn.utils

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

from mrcnn.model import MaskRCNN
from pathlib import Path
from skimage import io


def check_mouth(check_x):
    if not check_x.any():
        print(check_x[0])
    # return the mouth aspect ratio
    mar = 1
    return mar

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    # print mouth value
    # 注意前面縮排的格式, 需要依照當前程式的慣例, 選擇四個小空格, 或是一個大空格
    # print(mouth[0:10])
    return mar

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat', help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.79

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
predictor_path = 'shape_predictor_68_face_landmarks.dat' 
#predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor(predictor_path)

detector = dlib.get_frontal_face_detector()

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)
(xStart, xEnd) = (37, 48)

# start the video stream thread
#print("[INFO] starting video stream thread...")
#vs = VideoStream(src=args["webcam"]).start()
#time.sleep(1.0)

#frame_width = 640
#frame_height = 360

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
#time.sleep(1.0)

#-----------------------------------------------------------------------------------------------------------------------------
no_face = 0
IoU_count = 0
#-----------------------------------------------------------------------------------------------------------------------------

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.98


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [1]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
#VIDEO_SOURCE = "parking.mp4"

print('before-0')
# Load the video file we want to run detection on
#video_capture = cv2.VideoCapture(VIDEO_SOURCE)
video_capture = cv2.VideoCapture(0)
print('after-0')

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None

# How many frames of video we've seen in a row with a parking space open
free_space_frames = 0

# Have we alerted that a space is free yet?
notified_free_space = False


# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
    # Get Video Frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)


    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
        mouth = shape[mStart:mEnd]
        check_x = shape[xStart:xEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        xMAR = check_mouth(check_x)
        mar = mouthMAR
        # compute the convex hull for the mouth, then
        # visualize the mouth
        mouthHull = cv2.convexHull(mouth)		
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        #cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # visualize the eyes
        #xHull = cv2.convexHull(check_x)		
        #cv2.drawContours(frame, [xHull], -1, (0, 255, 0), 1)
        # Draw text if mouth is open
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "(#1) Mouth is Open!", (30,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    
    # Write the frame into the file 'output.avi'
    #out.write(frame)
    # --------------------------------------------

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    if parked_car_boxes is None:
        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
        parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    else:
        # We already know where the parking spaces are. Check if any are currently unoccupied.

        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])

        # See how much those cars overlap with the known parking spaces
        overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

        # Assume no spaces are free until we find one that is free
        free_space = False

        # Loop through each known parking space box
        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

            # For this parking space, find the max amount it was covered by any
            # car that was detected in our image (doesn't really matter which car)
            max_IoU_overlap = np.max(overlap_areas)

            # Get the top-left and bottom-right coordinates of the parking area
            y1, x1, y2, x2 = parking_area

            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU
            if max_IoU_overlap < 0.5:
                # Parking space not occupied! Draw a green box around it
                IoU_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"(#2) Baby is Leaving!", (x1 + 6, y2 - 6), font, 0.7, (255, 255, 0), 2)
                # Flag that we have seen at least one open space
                free_space = True
            #elif max_IoU_overlap < 0.5 and IoU_count<1:
                # Parking space not occupied! Draw a green box around it
            #    IoU_count += 1
            #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            else:
                # Parking space is still occupied - draw a red box around it
                IoU_count = 0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Write the IoU measurement inside the box
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
            print(IoU_count)

        # If there is no face landmark on screen, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
        if rects:
            no_face = 0
        else:
            no_face += 1

        if no_face > 7:
            cv2.putText(frame, f"(#3) No Face!!!", (10, 150), font, 0.7, (0, 255, 0), 2, cv2.FILLED)

        # If at least one space was free, start counting frames
        # This is so we don't alert based on one frame of a spot being open.
        # This helps prevent the script triggered on one bad detection.
        if free_space:
            free_space_frames += 0
        else:
            # If no spots are free, reset the count
            free_space_frames = 0

        # If a space has been free for several frames, we are pretty sure it is really free!
        if free_space_frames > 10:
            # Write SPACE AVAILABLE!! at the top of the screen
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 2, cv2.FILLED)

            # If this is the first time we've found a free space,
            # alert the user!
            if not notified_free_space:
                print("Open parking space - go go go!")
                notified_free_space = True

        # Show the frame of video on the screen
        cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video_capture.release()
cv2.destroyAllWindows()
