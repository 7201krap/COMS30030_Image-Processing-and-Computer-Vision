import cv2
import sys
import numpy as np

import my_lib as lib


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

def detectAndDisplay(frame, hough_circle_centers, hough_line_centers, darts):
    dart_counter_VJ_HT = 0

    for (x, y, width, height) in darts:
        flag = 0
        for c_center in hough_circle_centers:
            if x < c_center[0] < x+width and y < c_center[1] < y+height:
                flag = flag + 1
                dart_counter_VJ_HT = dart_counter_VJ_HT + 1
                frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # GREEN SQUARE --> denominator of precision.
                break   # move to next detected dart image

        for l_center in hough_line_centers:
            if x < l_center[0] < x+width and y < l_center[1] < y+height:
                if flag == 1:
                    break
                dart_counter_VJ_HT = dart_counter_VJ_HT + 1
                frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # GREEN SQUARE --> denominator of precision.
                break   # move to next detected dart image

    # return number of darts
    return darts, dart_counter_VJ_HT

def get_possible_radius(frame):

    darts_radius = []

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(frame_gray, frame_gray)

    # note that hyper-parameters are tuned
    darts = cascade.detectMultiScale(frame_gray, 1.03, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (50, 50), (500, 500))
    print("VJ detected darts: ", len(darts))

    for (x, y, width, height) in darts:
        darts_radius.append(np.maximum(width, height) // 2)

    radius_mean = np.mean(darts_radius)
    # return number of darts
    return radius_mean, darts

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

    avg_radius, darts_iter = get_possible_radius(frame)

    center_coordinates_for_hough_circle = lib.hough_circle_and_space(filename_path, int(avg_radius), 0.9)
    centers_coordinates_for_hough_line  = lib.hough_line_and_space(filename_path, 0.8)

    darts, counter_VJ_HT = detectAndDisplay(frame, center_coordinates_for_hough_circle,
                                            centers_coordinates_for_hough_line, darts_iter)
    print("Viola Jones and HT detected darts (Denominator of precision) -->", counter_VJ_HT)
    print("Denominator of recall == number of red squares")

    # draw + get list of manually annotated darts
    for (x, y, width, height) in manually_annotated[filename]:
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
        # RED SQUARE --> denominator of Recall
        manual_coordinates = [x, y, x + width, y + height]
        manual_coordinates_list.append(manual_coordinates)

    for (x, y, width, height) in darts:
        flag = 0
        auto_coordinates = [x, y, x + width, y + height]

        for c_center in center_coordinates_for_hough_circle:
            if x < c_center[0] < x+width and y < c_center[1] < y+height:
                auto_coordinates_list.append(auto_coordinates)
                flag = flag + 1
                break

        for l_center in centers_coordinates_for_hough_line:
            if x < l_center[0] < x+width and y < l_center[1] < y+height:
                if flag == 1:
                    break
                auto_coordinates_list.append(auto_coordinates)
                break

    successfully_detected_darts = lib.intersection_over_union(manual_coordinates_list, auto_coordinates_list, 0.5)
    print("Viola Jones and Hough detected darts + IOU. Numerator -->", successfully_detected_darts)

    # auto darts, manual darts, detected darts
    tpr, f1_score = lib.tpr_and_f1score(counter_VJ_HT, len(manually_annotated[filename]), successfully_detected_darts)
    print("tpr(recall) --> ", tpr)
    print("f1_score -->", f1_score)

    img_save_location = 'img_after_detection/' + 'detected_' + filename
    cv2.imshow("Final detection", frame)
    print("Showing total 4 images")
    cv2.imwrite(img_save_location, frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
