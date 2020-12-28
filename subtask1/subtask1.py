import cv2
import sys
import numpy as np

# Pre-annotated ground truths
manually_annotated = \
    {
        'dart0.jpg': np.array([]),
        'dart1.jpg': np.array([]),
        'dart2.jpg': np.array([]),
        'dart3.jpg': np.array([]),
        'dart4.jpg': [(344, 101, 120, 169)],
        'dart5.jpg': [(59, 134, 61, 71), (54, 244, 59, 72), (188, 211, 61, 72), (248, 160, 50, 65), (300, 238, 52, 69),
                      (379, 190, 60, 60), (429, 228, 54, 70), (511, 180, 59, 63), (562, 241, 56, 76),
                      (652, 184, 56, 64),
                      (681, 241, 52, 73)],
        'dart6.jpg': [(288, 117, 38, 39)],
        'dart7.jpg': [(341, 199, 80, 81)],
        'dart8.jpg': np.array([]),
        'dart9.jpg': [(89, 214, 100, 110)],
        'dart10.jpg': np.array([]),
        'dart11.jpg': [(330, 80, 50, 65)],
        'dart12.jpg': np.array([]),
        'dart13.jpg': [(425, 115, 115, 139)],
        'dart14.jpg': [(471, 216, 81, 99), (736, 191, 88, 95)],
        'dart15.jpg': np.array([])
    }

cascade_name = "frontalface.xml"
cascade = cv2.CascadeClassifier()


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

    faces_counter = 0

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
                faces_counter = faces_counter + 1

    return faces_counter


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(frame_gray, frame_gray)

    faces = cascade.detectMultiScale(frame_gray, 1.1, 1, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))

    # !!! faces length print하기 !!!

    for (x, y, width, height) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # return number of faces
    return faces


def main():
    # ========== setting for iou and f1-score ==========
    manual_coordinates_list = []
    auto_coordinates_list = []
    # ===================================================

    filename = sys.argv[1]
    filename_path = '../dart_images/' + filename

    frame = cv2.imread(filename_path, cv2.IMREAD_COLOR)

    if not cascade.load(cascade_name):
        print("--(!)Error loading")
        exit(0)

    faces = detectAndDisplay(frame)
    print("face detection by viola jones -->", len(faces))

    # get list of manually annotated faces
    for (x, y, width, height) in manually_annotated[filename]:
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        manual_coordinates = [x, y, x + width, y + height]
        manual_coordinates_list.append(manual_coordinates)

    # draw + get list of automatically detected faces
    for (x, y, w, h) in faces:
        auto_coordinates = [x, y, x + w, y + h]
        auto_coordinates_list.append(auto_coordinates)

    successfully_detected_faces = intersection_over_union(manual_coordinates_list, auto_coordinates_list, 0.5)
    print("successfully_detected_faces -->", successfully_detected_faces)

    # auto faces, manual faces, detected faces
    tpr, f1_score = tpr_and_f1score(len(faces), len(manually_annotated[filename]), successfully_detected_faces)
    print("tpr(recall) --> ", tpr)
    print("f1_score -->", f1_score)

    img_save_location = 'img_after_detection/' + 'detected_' + filename
    cv2.imshow("frame", frame)
    cv2.imwrite(img_save_location, frame)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()