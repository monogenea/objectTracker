#%% Imports and constants
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

# Define target class and thresholds
TARGET_CLASS = 'cat' # try pottedplant, vase, book, diningtable, bottle
OBJ_THRESH = .5
P_THRESH = .5

#%% Load YOLOv3 COCO weights, configs and class IDs

# Import class names
with open('yolov3/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them
cfg = 'yolov3/yolov3.cfg'
weights = 'yolov3/yolov3.weights'
# Load model
model = cv2.dnn.readNetFromDarknet(cfg, weights)
# Extract names from output layers
layersNames = model.getLayerNames()
outputNames = [layersNames[i[0] - 1] for i in model.getUnconnectedOutLayers()]

#%% Define function to extract object coordinates if successful in detection
def where_is_it(target_class, frame, outputs):
    assert target_class in classes
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    probs, bboxes = [], []
    class_idx = classes.index(target_class) + 5

    for scl in outputs: # different detection scales
        obj_prob = scl[:, class_idx]
        pred = scl[np.argmax(obj_prob), :]
        # Save prob and bbox coordinates if both objectness and probability pass respective thresholds
        if pred[4] > OBJ_THRESH and pred[class_idx] > P_THRESH:
            center_x = int(pred[0] * frame_w)
            center_y = int(pred[1] * frame_h)
            width = int(pred[2] * frame_w)
            height = int(pred[3] * frame_h)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            probs.append(float(pred[class_idx]))
            bboxes.append([left, top, width, height])

    # Simply take the hit with max probability
    if len(probs) > 0:
        return True, bboxes[np.argmax(probs)]
    else:
        return False, None

#%% Load video capture and init VideoWriter 
vid = cv2.VideoCapture('input/input.mp4')
vid_w, vid_h = int(vid.get(3)), int(vid.get(4))
out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                      vid.get(cv2.CAP_PROP_FPS), (vid_w, vid_h))

# Check if capture started successfully
assert vid.isOpened()

#%% Init and execute

# Initialize variables - no need for non-maximum suppression
x, y, w, h = 0, 0, 0, 0
count = 0

# Create new window
cv2.namedWindow('stream')

# Create KCF tracker
tracker = cv2.TrackerMedianFlow_create() # fast but fails under occlusion

while(vid.isOpened()):
    # Perform detection every 60 frames
    perform_detection = count % 60 == 0
    ok, frame = vid.read()

    if ok:
        if perform_detection: # perform detection
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0,0,0], 1, crop=False)
            # Pass blob to model
            model.setInput(blob)
            # Execute forward pass
            outputs = model.forward(outputNames)
            obj_found, coords = where_is_it(TARGET_CLASS, frame, outputs)
            
            if obj_found:
                x, y, w, h = [int(v) for v in coords]
                # Mark detection frame with blue color
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Switch to tracking and increase counter
                init_tracking = tracker.init(frame, (x, y, x+w, y+h))
                perform_detection = False
                count += 1
            else: # declare failure
                cv2.putText(frame, 'Detection failed', (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        else: # perform tracking
            is_tracking, coords = tracker.update(frame)
            if is_tracking:
                x, y, _w, _h = [int(v) for v in coords]
                # Mark tracking frame with green color, write class name on top
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, TARGET_CLASS, (x, y-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # Increase counter
                count += 1
            else: # declare failure
                cv2.putText(frame, 'Tracking failed', (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                detected = False
        # Display the resulting frame
        cv2.imshow('stream', frame)
        out.write(frame)
        # Press ESC to exit
        if cv2.waitKey(25) & 0xFF == 27:
            break
    # Break if capture read does not work
    else:
        print('Exhausted / cannot read video capture.')
        break
out.release()
cv2.destroyAllWindows()