import numpy as np
import cv2
import math


def degree_to_radian(x):
    return x * 3.141592 / 180


def edge_detector(_input_):
    mag_val = np.zeros(_input_.shape)
    row = _input_.shape[0]
    col = _input_.shape[1]

    # sobel edge detector itself contains Gaussian Blur
    sobel_dx = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    sobel_dy = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

    dx_dir = cv2.filter2D(_input_, -1, sobel_dx)
    dy_dir = cv2.filter2D(_input_, -1, sobel_dy)

    for y in range(row):
        for x in range(col):
            mag_val[y, x] = math.floor(math.sqrt(dx_dir[y, x] ** 2 + dy_dir[y, x] ** 2))

    # Threshold is now 0.7 which means we are going to get top 30%
    threshold_value = 0.7 * np.max(mag_val)
    output = np.zeros(mag_val.shape)
    for y in range(mag_val.shape[0]):
        for x in range(mag_val.shape[1]):
            if mag_val[y, x] > threshold_value:
                output[y, x] = 255
            else:
                output[y, x] = 0

    cv2.imshow("edge detected image. Image will show up soon", output)
    # cv2.waitKey(0)

    return output


def hough_line_and_space(_image_, th):

    # pre-processing
    image = cv2.imread(_image_, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(grayscale, grayscale)
    edge = edge_detector(grayscale)

    # pre-compute cos and sin therefore enhancing further calculation when voting on the hough line space
    angle = np.arange(0, 180, 1)
    cos = np.cos(degree_to_radian(angle))
    sin = np.sin(degree_to_radian(angle))

    # initialise variable #
    row = image.shape[0]
    col = image.shape[1]

    # ------------------------ main algorithm begins here ------------------------
    # voting : voting variable will be used for later hough line space
    distance_range = round(math.sqrt(edge.shape[0]**2 + edge.shape[1]**2))
    voting         = np.zeros((2 * distance_range, len(angle)), dtype=np.uint8)

    # Threshold to get edges pixel location (x,y)
    thr_255     = np.where(edge == 255)
    candidates  = list(zip(thr_255[0], thr_255[1]))

    # loop through all the edges computed by thr_255.
    # Calculate distance value for every location
    # Consider all of the possible angle which ranges from 0 - 180
    for idx in range(len(candidates)):
        for ang in range(len(angle)):
            # precomputed cos and sin are used in here
            # compute distance from origin to the line
            distance = int(round(candidates[idx][1] * cos[ang] + candidates[idx][0] * sin[ang]))
            voting[distance, ang] = voting[distance, ang] + 2
        print("Accumulating hough line .. {} % finished".format((idx / len(candidates)) * 100))

    print("Finalising calculation. Please wait ,,,\n")

    _max_ = np.max(voting)
    # print("============= max value =============", _max_)
    voting = voting / _max_

    # resize hough space
    resized = cv2.resize(voting, (500, 500), interpolation=cv2.INTER_AREA)
    print("Showing hough space for line detection\n")
    cv2.imshow("Hough line space (resized). Image will show up soon", resized)
    # print("Enter space bar to show detected hough line on the image \n")
    # cv2.waitKey(0)

    # threshold voting matrix. Otherwise it will draw too many line increasing computation as well as ruining the program
    # This threshold (th) is necessary
    thr_255 = np.where(voting > th)
    # distance: thr_255[0]
    # angle:    thr_255[1]
    candidates = list(zip(thr_255[0], thr_255[1]))

    final_image, center_list = draw_hough_line(candidates, row, col, image)

    # cv2.imshow("img", final_image)
    # cv2.waitKey(0)

    return center_list

def draw_hough_line(candidate_array, _row_, _col_, _image_):

    center_list = []
    final_grayscale_img = np.zeros((_row_, _col_))

    # Loop through all the candidate coordinates
    # Accumulate the line to find the center of the dart board
    for i in range(0, len(candidate_array)):
        grayscale_img = np.zeros((_row_, _col_))
        a, b = np.cos(degree_to_radian(candidate_array[i][1])), np.sin(degree_to_radian(candidate_array[i][1]))  # angle
        valid_x, valid_y = a * candidate_array[i][0], b * candidate_array[i][0]  # distance
        x1, y1 = int(valid_x + 1000 * (-b)), int(valid_y + 1000 * a)
        x2, y2 = int(valid_x - 1000 * (-b)), int(valid_y - 1000 * a)

        line_img = cv2.line(grayscale_img, (x1, y1), (x2, y2), 0.25, 2)
        cv2.accumulate(line_img, final_grayscale_img)  # accumulate lines to the final_grayscale_img
        # cv2.imshow("line_img", final_grayscale_img)

    for r in range(_row_):
        for c in range(_col_):
            if final_grayscale_img[r][c] >= 1:
                # notice that (r,c) should be changed to (c,r) to correctly position accumulated coordinate
                center_list.append([c, r])
                cv2.circle(_image_, (c, r), 2, (0, 255, 0), 2)

    return _image_, center_list

def hough_circle_and_space(image, r, threshold):

    # pre-processing and initialise values
    frame = cv2.imread(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    cols = gray.shape[1]
    cv2.equalizeHist(gray, gray)
    edge = edge_detector(gray)
    # ----------------------------------------

    # Set radius range
    # Radius range crucially determines the accuracy of the detection e.g. recall, precision, and f1-score
    # Usually it is better to have large range of radius to make a robust detection program
    min_rad = r - 10
    max_rad = r + 70

    print("radius range: ", min_rad, " to ", max_rad)

    accumulator = np.zeros((int(max_rad), rows, cols))
    get_centers = []

    # iterate through all radius range from min-rad to max-rad
    for r in range(min_rad, max_rad):
        for row in range(rows):
            for col in range(cols):
                if edge[row][col] == 255:
                    for theta in range(0, 360, 10):
                        # draw circle
                        x = int(row - r * math.cos(theta * math.pi / 180))
                        y = int(col - r * math.sin(theta * math.pi / 180))
                        if 0 < x < rows and 0 < y < cols:
                            accumulator[r][x][y] += 1

        print("Finding appropriate hough circle radius for the image. Radius:", r)

    print("Finalising calculation. Please wait ,,,\n")

    # find
    max_val = np.max(accumulator)

    # generate hough circle space and make it to numpy form
    hough_circle_space = accumulator.sum(axis=0)
    hough_circle_space = np.array(hough_circle_space)

    # find the maximum of hough circle space. Otherwise the hough circle space will be covered with white pixels
    hough_circle_space_max = np.max(hough_circle_space)
    hough_circle_space = hough_circle_space / hough_circle_space_max

    # divide accumulator with max value of accumulator
    # This enables the program to set specific percentage of thresholding
    accumulator = accumulator / max_val

    for r in range(min_rad, max_rad):
        for row in range(rows):
            for col in range(cols):
                if accumulator[r][row][col] > threshold:
                    # notice that (row, col) should be changed to (col, row)
                    get_centers.append([col, row])
                    __image = cv2.circle(frame, (col, row), r, (0, 255, 0), 2)    # circle
                    __image = cv2.circle(frame, (col, row), 2, (255, 0, 0), 2)    # centre

    # cv2.imshow("Circle Detected Image", __image)

    cv2.imshow("Hough circle space. Image will show up soon", hough_circle_space)
    # print("Enter space bar for hough line detection")

    # cv2.waitKey(0)

    # return coordinates of the center detected by hough circle transform
    return get_centers

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


# Actually, manual_cor and auto_cor could be changed. This function does not effected by the order
# e.g. intersection_over_union(auto_cor, manual_cor, threshold) is fine
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
