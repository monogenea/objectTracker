#%% Imports and constants
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

# Define objectness, prob and NMS thresholds
OBJ_THRESH = .75
P_THRESH = .75
NMS_THRESH = .5

# Set random seed
np.random.seed(999)

#%% Load YOLOv3 COCO weights, configs and class IDs

# Import class names
with open('yolov3/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
colors = np.random.randint(0, 255, (len(classes), 3))
# Give the configuration and weight files for the model and load the network using them
cfg = 'yolov3/yolov3.cfg'
weights = 'yolov3/yolov3.weights'
# Load model
model = cv2.dnn.readNetFromDarknet(cfg, weights)
# Extract names from output layers
layersNames = model.getLayerNames()
outputNames = [layersNames[i[0] - 1] for i in model.getUnconnectedOutLayers()]

#%% Define function to extract object coordinates if successful in detection
def where_is_it(frame, outputs):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    bboxes, probs, class_ids = [], [], []
    for preds in outputs: # different detection scales
        hits = np.any(preds[:, 5:] > P_THRESH, axis=1) & (preds[:, 4] > OBJ_THRESH)
        # Save prob and bbox coordinates if both objectness and probability pass respective thresholds 
        for i in np.where(hits)[0]:
            pred = preds[i, :]
            center_x = int(pred[0] * frame_w)
            center_y = int(pred[1] * frame_h)
            width = int(pred[2] * frame_w)
            height = int(pred[3] * frame_h)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            # Append all info
            bboxes.append([left, top, width, height])
            probs.append(float(np.max(pred[5:])))
            class_ids.append(np.argmax(pred[5:]))
    return bboxes, probs, class_ids
    
#%% Load video capture and init VideoWriter 
vid = cv2.VideoCapture('input/input.mp4')
vid_w, vid_h = int(vid.get(3)), int(vid.get(4))
out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                      vid.get(cv2.CAP_PROP_FPS), (vid_w, vid_h))

# Check if capture started successfully
assert vid.isOpened()

#%% Initiate processing

# Init count
count = 0

# Create new window
cv2.namedWindow('stream')

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
            bboxes, probs, class_ids = where_is_it(frame, outputs)
            
            if len(bboxes) > 0:
                # Init multitracker
                mtracker = cv2.MultiTracker_create()
                # Apply non-max suppression and pass boxes to the multitracker
                idxs = cv2.dnn.NMSBoxes(bboxes, probs, P_THRESH, NMS_THRESH)
                for i in idxs:
                    bbox = [int(v) for v in bboxes[i[0]]]
                    x, y, w, h = bbox
                    mtracker.add(cv2.TrackerMedianFlow_create(), frame, (x, y, w, h))
                    # Mark detection frame with blue color
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Increase counter
                count += 1
            else: # declare failure
                cv2.putText(frame, 'Detection failed', (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        else: # perform tracking
            is_tracking, bboxes = mtracker.update(frame)
            if is_tracking:
                for i, bbox in enumerate(bboxes):
                    x, y, w, h = [int(val) for val in bbox]
                    class_id = classes[class_ids[idxs[i][0]]]
                    col = [int(c) for c in colors[class_ids[idxs[i][0]], :]]
                    # Mark tracking frame with corresponding color, write class name on top
                    cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
                    cv2.putText(frame, class_id, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                # Increase counter
                count += 1
            # If tracking fails, reset count to trigger detection
            else:
                count = 0

        # Display the resulting frame
        cv2.imshow('stream', frame)
        out.write(frame)
        # Press ESC to exit
        if cv2.waitKey(25) & 0xFF == 27:
            break
    # Break if capture read does not work
    else:
        print('Exhausted video capture.')
        break
out.release()
cv2.destroyAllWindows()