import cv2
import sys

# Pre-annotated ground truths
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

cascade_name = "cascade.xml"
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

    number_of_auto_darts = len(manual_cor)
    number_of_manual_darts = len(auto_cor)

    darts_counter = 0

    for i in range(number_of_auto_darts):
        for j in range(number_of_manual_darts):

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


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(frame_gray, frame_gray)

    darts = cascade.detectMultiScale(frame_gray, 1.03, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))

    print("number of dartboards", len(darts))

    for (x, y, width, height) in darts:
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # return number of darts
    return darts


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

    darts = detectAndDisplay(frame)
    print("face detection by viola jones -->", len(darts))

    # draw + get list of manually annotated darts
    for (x, y, width, height) in manually_annotated[filename]:
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        manual_coordinates = [x, y, x + width, y + height]
        manual_coordinates_list.append(manual_coordinates)

    # get list of automatically detected darts
    for (x, y, w, h) in darts:
        auto_coordinates = [x, y, x + w, y + h]
        auto_coordinates_list.append(auto_coordinates)

    successfully_detected_darts = intersection_over_union(manual_coordinates_list, auto_coordinates_list, 0.5)
    print("successfully_detected_darts -->", successfully_detected_darts)

    # auto darts, manual darts, detected darts
    tpr, f1_score = tpr_and_f1score(len(darts), len(manually_annotated[filename]), successfully_detected_darts)
    print("tpr(recall) --> ", tpr)
    print("f1_score -->", f1_score)

    img_save_location = 'img_after_detection/' + 'detected_' + filename
    cv2.imshow("frame", frame)
    cv2.imwrite(img_save_location, frame)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
