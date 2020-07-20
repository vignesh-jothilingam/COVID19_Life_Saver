import numpy as np
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from social_dist_confg import social_distancing_config as config
from social_dist_confg.detection import detect_people
from scipy.spatial import distance as dist
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import imutils
from imutils.video import VideoStream
from playsound import playsound
from tensorflow.keras.models import load_model


def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.8:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)
# for mask detection
print("Face detection model and weight file loading")
prototxtPath = 'deploy.prototxt'
weightsPath = 'res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
print("Loanding the Face Mask Detection Pre-Trained Model")
maskNet = load_model('.\mask_detector.model')
# for social distance ditection
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
yolo_weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
yolo_configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(yolo_configPath, yolo_weightsPath)
# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# load our YOLO object detector trained on COCO dataset (80 classes)
print("Loading the YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(yolo_configPath, yolo_weightsPath)
# to strore the output Covid19 Mask and Social distancing Detections

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialize the video stream and allow the camera sensor to warm up
print("getting video data from CCTV/Webcam")
vs = VideoStream(src=0).start()
#print('vs',vs)
time.sleep(2.0)
writer = None
# loop over the frames from the video stream
cur_path = os.getcwd()
i_faces = 0
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (100, 23, 10) if label == "Mask" else (0, 0, 255)
		if (withoutMask*100) > 70:
			i_faces += 1
			s = r'.\{}.jpg'.format(cur_path,str(i_faces))
			crop_img = frame[startY:endY, startX:endX]
			cv2.imwrite(s, crop_img)
			playsound(r'.\sound.mp3')
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	#frame = imutils.resize(frame, width=700)
	results = detect_people(frame,net,ln,personIdx=LABELS.index("person")) 
	# initialize the set of indexes that violate the minimum social distance
	violate = set()
	display=1
	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)
	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)
		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10,10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 3,bottomLeftOrigin=False)
	# check to see if the output frame should be displayed to our
	# screen
	if display > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()