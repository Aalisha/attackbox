import torchvision
import cv2
from torchvision import transforms as T
from PIL import Image
import matplotlib
import pylab as plt
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def obj_prediction(img,model, threshold=0.6):
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  pred = model([img]) # Pass the image to the model
  #print(pred)
  pred_labels = [i for i in list(pred[0]['labels'])]
  pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'])] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_classes = pred_classes[:pred_t+1]
  
  ''' 
  pred_boxes_list = []; pred_classes_list = []
  for pred_box in pred_boxes:
  	pred_boxes_list.append(np.array(pred_box))

  for pred_class in pred_classes:
        pred_classes_list.append(np.array(pred_class))
  '''
  print('Object detection')
  print(pred_boxes)      
  #pred_boxes, pred_labels = non_max_suppression_fast(np.array(pred_boxes), np.array(pred_labels), 0.8 )
  print('Object prediction')
  print(pred_boxes) 
  pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred_labels] # Get the Prediction Score

  return pred_boxes, pred_classes, pred_labels

def get_prediction(img_path, threshold):
  img = Image.open(img_path) # Load the image
  pred_boxes, pred_class = obj_prediction(img, threshold)
  return pred_boxes, pred_class

def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
 
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    cv2.imwrite('/home/aalishadalal/obj-detection/two_cats-detected.jpg',img)
    plt.figure(figsize=(20,30)) # display the output image
    print('hello')
    return img

# Malisiewicz et al.
def non_max_suppression_fast(boxes, pred_classes, overlapThresh):
	# if there are no boxes, return an empty list
    
	if len(boxes) == 0:
		return []
	boxes = boxes.reshape((boxes.shape[0], boxes.shape[1]*boxes.shape[2]))
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	print(pred_classes[pick])
	print(boxes[pick])
	return boxes[pick].astype("int"), pred_classes[pick]




