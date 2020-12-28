import cv2
import numpy as np
import sys

manually_annotated = \
    {
        'dart0.jpg': [(440, 20, 150, 180)],
        'dart1.jpg': [(195, 135, 190, 190)],
        'dart2.jpg': [(97, 92, 100, 100)],
        'dart3.jpg': [(325, 150, 65, 70)],
        'dart4.jpg': [(200, 100, 200, 200)],
        'dart5.jpg': [(440, 140, 95, 95)],
        'dart6.jpg': [(210, 120, 60, 60)],
        'dart7.jpg': [(255, 170, 145, 145)],
        'dart8.jpg': [(845, 215, 120, 120), (60, 255, 60, 90)],
        'dart9.jpg': [(201, 51, 231, 231)],
        'dart10.jpg': [(90, 103, 96, 107), (585, 129, 56, 81), (918, 150, 35, 63)],
        'dart11.jpg': [(175, 102, 51, 73)],
        'dart12.jpg': [(155, 80, 62, 134)],
        'dart13.jpg': [(270, 125, 130, 129)],
        'dart14.jpg': [(108, 97, 140, 140), (980, 90, 140, 140)],
        'dart15.jpg': [(154, 55, 130, 137)]
    }

def tpr_and_f1score(auto, manual, detected):
    if manual == 0 or detected == 0 or auto == 0:
        return 0, 0

    recall = detected / manual  # recall == tpr
    if recall > 1:
        recall = 1

    precision = detected / auto
    # the program would not reach this section: make error message
    if precision > 1:
        print("check your precision score")
        precision = 1

    f1 = (2 * recall * precision) / (recall + precision)

    return recall, f1


# the actual order in which these parameters are supplied to intersection_over_union does not matter
def intersection_over_union(manual_cor, auto_cor, threshold):

    number_of_auto_faces = len(manual_cor)
    number_of_manual_faces = len(auto_cor)

    darts_counter = 0

    for i in range(number_of_auto_faces):
        for j in range(number_of_manual_faces):

            xA = max(manual_cor[i][0], auto_cor[j][0])  # top left x
            yA = max(manual_cor[i][1], auto_cor[j][1])  # top left y
            xB = min(manual_cor[i][2], auto_cor[j][2])  # bottom right x
            yB = min(manual_cor[i][3], auto_cor[j][3])  # bottom right y

            # intersected area
            intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # left and right are needed to compute union of the rectangles
            left  = (manual_cor[i][2] - manual_cor[i][0] + 1) * (manual_cor[i][3] - manual_cor[i][1] + 1)
            right = (auto_cor[j][2] - auto_cor[j][0] + 1) * (auto_cor[j][3] - auto_cor[j][1] + 1)

            # think IOU as a Venn diagram.
            # intersection : A and B
            # union : A or B which is float(left + right - intersection)
            iou = intersection / float(left + right - intersection)

            if iou > threshold:
                darts_counter = darts_counter + 1

    return darts_counter

# Load trained yolo weights and configuration file
read_Net_getter = cv2.dnn.readNet("yolov3_dartboard_detection_weights.weights", "yolov3_dartboard_detection.cfg")

# ========== setting for iou and f1-score ==========
manual_coordinates_list = []
auto_coordinates_list = []
# ===================================================

# We would like to detect dartboard
single_class = ["dart"]

# Pre processing
filename = sys.argv[1]
images_path = '../dart_images/' + filename
layers = read_Net_getter.getLayerNames()
final_output = [layers[i[0] - 1] for i in read_Net_getter.getUnconnectedOutLayers()]

# Load dart_.jpg images
img = cv2.imread(images_path)
row, col, _ = img.shape

for (x, y, width, height) in manually_annotated[filename]:
    img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
    manual_coordinates = [x, y, x + width, y + height]
    manual_coordinates_list.append(manual_coordinates)

# object detection
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

read_Net_getter.setInput(blob)
_final_ = read_Net_getter.forward(final_output)

# result visualisation
class_idx, confidence_list, boxes = [], [], []
for detections in _final_:
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # be sure to think about confidence threshold
        if confidence > 0.3:
            # detected object.
            center_x, center_y  = int(detection[0] * col), int(detection[1] * row)

            # get location of the box
            w = int(detection[2] * col)
            h = int(detection[3] * row)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # append location of the box
            boxes.append([x, y, w, h])
            confidence_list.append(float(confidence))
            class_idx.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidence_list, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        auto_coordinates = [x, y, x + w, y + h]
        auto_coordinates_list.append(auto_coordinates)
        # print(auto_coordinates_list)
        label = str(single_class[class_idx[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y + 55), font, 1, (0, 255, 0), 2)

successfully_detected_darts = intersection_over_union(manual_coordinates_list, auto_coordinates_list, 0.5)
print("successfully_detected_darts (YOLO + MANUAL) -->", successfully_detected_darts)

# auto darts, manual darts, detected darts
tpr, f1_score = tpr_and_f1score(len(indexes), len(manually_annotated[filename]), successfully_detected_darts)
print("tpr(recall) --> ", tpr)
print("f1_score -->", f1_score)

img_save_location = 'img_after_detection/' + 'detected_' + filename

cv2.imshow("Image", img)
cv2.imwrite(img_save_location, img)
key = cv2.waitKey(0)

cv2.destroyAllWindows()
