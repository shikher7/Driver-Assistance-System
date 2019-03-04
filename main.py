import pickle
from sklearn import svm
import cv2
import numpy as np
from math import sqrt
import os
from training import getSign


#   Types of Road signs which are present in our machine learning model
SIGNS = ["OTHER",
         "Horn Prohibited",
         "Stop",
         "No Parking",
         "Compulsory Keep left",
         "Pedestrian Crossing Ahead",
         "Slippery Road Ahead",
         "Round About",
         "Speed Limit",
         "No Entry",
         "Hospital Ahead",
         "OTHER"
         ]


#   Removing small components which are of no importance
def Small_Components_removal(image, threshold):
    #   Finding all your connected components in the image
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    img = np.zeros((output.shape), dtype=np.uint8)

    #   Defining acceptance threshold size of a connected component

    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img[output == i + 1] = 255
    return img


#   Checking if the contour has shape of sign (Ex. Circle,Hexagon,Triangle etc..)
def check_is_contour_Sign(contour, centroid, threshold):
    result = []
    for cnt in contour:
        cnt = cnt[0]
        distance = sqrt((cnt[0] - centroid[0]) ** 2 + (cnt[1] - centroid[1]) ** 2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]
    temp = sum(s for s in signature)
    temp = temp / len(signature)

    #   Condition for checking the shape

    if temp > threshold or len(cv2.approxPolyDP(contour, 0.25 * (cv2.arcLength(contour, True)), True)) == 3:
        return True, max_value + 2
    else:
        return False, max_value + 2


#   Finding the Greatest sign among all the detected signs
def find_Greatest_Sign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = check_is_contour_Sign(c, [cX, cY], threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 2, bottom + 2)]

            width = image.shape[1]
            height = image.shape[0]
            top = max([int(coordinate[0][1]), 0])
            bottom = min([int(coordinate[1][1]), height - 1])
            left = max([int(coordinate[0][0]), 0])
            right = min([int(coordinate[1][0]), width - 1])
            sign = image[top:bottom, left:right]
    return sign, coordinate

#Finding contours of connected components of the image
#Contour are array of pixel cordinates of all connected components respectively
def find_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    return cnts

def Locailizing_ROI(image, min_size_components, circle_detection_threshold, model, count, current_sign_type):
    original_image = image.copy()

    #   Red/Blue colored masking to remove unwanted areas from the image

    image = cv2.bitwise_and(original_image, original_image, mask=red_blue_masking(original_image))
    cv2.imshow("Colored feature extraction", image)

    #   Preprocessing the image for effecient detection Starts here..

    #   LaplacianOfGaussian of image helps finding borders of the objects present in image

    LoG_image = cv2.medianBlur(image, 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    #    cv2.imshow("LoG", LoG_image)
    image = LoG_image

    #   Binarization
    #   Converting image into binary colors (Black/white) for effecient detecting

    thresh = cv2.threshold(image, 22, 255, cv2.THRESH_BINARY)[1]
    #   Adaptive Thresholding is more efficient but slow
    # thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # cv2.imshow("Threshold",thresh)
    binary_image = thresh

    #   Preprocessing the image for effecient detection Ends here..

    #   Remove small sized components from the image which are of no intrest

    binary_image = Small_Components_removal(binary_image, min_size_components)
    # cv2.imshow('BINARY', binary_image)

    # binary_image = cv2.bitwise_and(binary_image,binary_image, mask=red_blue_masking(image))
    cv2.imshow('BINARY IMAGE', binary_image)

    #   Finding contours of connected components of the image
    #   Contour are array of pixel cordinates of all connected components respectively

    contours = find_contours(binary_image)

    #   Finding the Greatest sign among all the detected signs
    sign, coordinate = find_Greatest_Sign(original_image, contours, circle_detection_threshold, 25)

    text = ""
    sign_type = -1

    if sign is not None:
        sign_type = getSign(model, sign)
        print(sign_type)
        sign_type = sign_type if sign_type <= 10 else 0
        text = SIGNS[sign_type]
        cv2.imwrite(str(count) + '_' + text + '.png', sign)
    else:
        sign_type = 0

    if sign_type > 0 and sign_type != current_sign_type:
        cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2,
                    cv2.LINE_4)
    return coordinate, original_image, sign_type, text


def red_blue_masking(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #    define range of blue color in HSV
    lower_blue = np.array([101, 50, 28])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0, 0, 125])
    upper_white = np.array([255, 255, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    #    upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    #   join masks
    mask_red = mask0 + mask1
    # mask_red = cv2.inRange( hsv, lower_red, upper_red)

    mask_1 = cv2.bitwise_or(mask_blue, mask_red)
    mask_2 = cv2.bitwise_or(mask_1, mask_white)
    # return cv2.bitwise_or(mask_red, mask_blue)
    return mask_1


def main():
    #   Clean previously saved images
    file_list = os.listdir('./')
    for file_name in file_list:
        if '.png' in file_name:
            os.remove(file_name)

    print('Loading Model...')
    with open('svm_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Starting Detection...")
    vidcap = cv2.VideoCapture(0)

    #    For IP Webcam 'http://192.168.43.63:8080/video'

    #   Codec For Saving Output Video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (640, 480))
    circle_detection_threshold = 0.8  # parameter
    count = 0
    current_sign = None
    while True:
        success, frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        frame = cv2.resize(frame, (640, 480))
        print("Frame:{}".format(count))
        min_size_components = 200
        coordinate, image, sign_type, text = Locailizing_ROI(frame, min_size_components, circle_detection_threshold,
                                                             model, count, current_sign)
        if coordinate is not None and sign_type == 0:
            cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
        if sign_type != -1:
            print("Sign: " + SIGNS[sign_type])
        cv2.imshow('Result', image)
        count = count + 1

        #        Write to video

        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Finish {} frames".format(count))
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()