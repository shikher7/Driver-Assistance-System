import cv2
import numpy as np
from math import sqrt
import os
import math

from classification import getLabel,training

SIGNS = ["OTHER",
        "STOP",
        "TURN LEFT",
        "TURN RIGHT",
        "DO NOT TURN LEFT",
        "DO NOT TURN RIGHT",
        "ONE WAY",
        "SPEED LIMIT",
        "NOT FOUND"
        ]

# Clean all previous file
def clean_images():
	file_list = os.listdir('./')
	for file_name in file_list:
		if '.png' in file_name:
			os.remove(file_name)


### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
#    cv2.imshow("CL", img_hist_equalized)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.medianBlur(image,3)           # paramter 
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
#    cv2.imshow("LoG", LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,22,255,cv2.THRESH_BINARY)[1]
#    thresh,a=cv2.threshold(image,22,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#    Adaptive Thresholding is more efficient but slow
#    cv2.imshow("Threshold",thresh)
    return thresh

def preprocess_image(image):
#    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
#    cv2.imshow("RSC",img2)
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    return cnts

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
#        print (p)
#        cv2.waitKey(0)
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum(s for s in signature)
    temp = temp / len(signature)
    if temp > threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 
def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-5,top-5),(right+5,bottom+5)]
            sign = cropSign(image,coordinate)
    return sign, coordinate


def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)
#    cv2.imshow('BINARY', binary_image)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))
    cv2.imshow('BINARY IMAGE', binary_image)
    
    # Preprocesing Done
    contours = findContour(binary_image)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    text = ""
    sign_type = -1

    if sign is not None:
        sign_type = getLabel(model, sign)
        print(sign_type)
        sign_type = sign_type if sign_type <= 8 else 0
        text = SIGNS[sign_type]
        cv2.imwrite(str(count)+'_'+text+'.png', sign)
    else:
        sign_type = 0

    if sign_type > 0 and sign_type != current_sign_type:        
        cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
    return coordinate, original_image, sign_type, text

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([101,50,38])
    upper_blue = np.array([110,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,125])
    upper_white = np.array([255,255,255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
 
    lower_red = np.array([160,20,70])
    upper_red = np.array([190,255,255])
    mask_red = cv2.inRange( hsv, lower_red, upper_red)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask_2 = cv2.bitwise_or(mask_1, mask_red)

    return mask_2

def main():
	#Clean previous image    
    clean_images()
#    x = training()
#    cv2.show(x)
#    if cv2.waitKey(0):
#        cv2.destroyAllWindows()
    
    #Training phase
    model = training()
#    model = cv2.ml.SVM_load('data_svm.dat')
    vidcap = cv2.VideoCapture(0)
#    'http://192.168.43.63:8080/video' for IP Webcam
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = .75   # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
#        width = frame.shape[1]
#        height = frame.shape[0]
        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        frame = cv2.resize(frame, (640,480))

        print("Frame:{}".format(count))
        min_size_components = 200
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        coordinate, image, sign_type, text = localization(frame, min_size_components, similitary_contour_with_circle, model, count, current_sign)
        if coordinate is not None and sign_type==0:
            cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
        if sign_type!=-1:
            print("Sign: " + SIGNS[sign_type])
            
        #Tracking
#        if sign_type > 0 and (not current_sign or sign_type != current_sign):
#            current_sign = sign_type
#            current_text = text
#            top = int(coordinate[0][1])
#            left = int(coordinate[0][0])
#            bottom = int(coordinate[1][1])
#            right = int(coordinate[1][0])
#
#            position = [count, sign_type if sign_type <= 8 else 8, coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]]
##            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
##            font = cv2.FONT_HERSHEY_PLAIN
##            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
#
#            tl = [left, top]
#            br = [right,bottom]
##            print(tl, br)
#            current_size = math.sqrt(math.pow((tl[0]-br[0]),2) + math.pow((tl[1]-br[1]),2))
#            # grab the ROI for the bounding box and convert it
#            # to the HSV color space
#            roi = frame[tl[1]:br[1], tl[0]:br[0]]
#            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
##            cv2.imshow("ROI",roi)
#            
#            # compute a HSV histogram for the ROI and store the bounding box
#            roiHist = cv2.calcHist([roi], [0], None, [180], [0, 180])
##            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
#            
#            roiBox = (tl[0], tl[1], br[0], br[1])
#
#        elif current_sign:
#            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
##            cv2.imshow("BPRO",backProj)
#
#            # apply cam shift to the back projection, convert the
#            # points to a bounding box, and then draw them
#            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
#            pts = np.int0(cv2.boxPoints(r))
#            s = pts.sum(axis = 1)
#            tl = pts[np.argmin(s)]
#            br = pts[np.argmax(s)]
#            size = math.sqrt(pow((tl[0]-br[0]),2) +pow((tl[1]-br[1]),2))
#
#            if  current_size < 1 or size < 1 or size / current_size > 30 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) > 10 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) < 0.5:
#                current_sign = None
#                print("Stop tracking")
#            else:
#                current_size = size
#
#            if sign_type > 0:
#                top = int(coordinate[0][1])
#                left = int(coordinate[0][0])
#                bottom = int(coordinate[1][1])
#                right = int(coordinate[1][0])
#
#                position = [count, sign_type if sign_type <= 8 else 8, left, top, right, bottom]
#                cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
#                font = cv2.FONT_HERSHEY_PLAIN
#                cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
#            elif current_sign:
#                position = [count, sign_type if sign_type <= 8 else 8, tl[0], tl[1], br[0], br[1]]
#                cv2.rectangle(image, (tl[0], tl[1]),(br[0], br[1]), (0, 255, 0), 1)
#                font = cv2.FONT_HERSHEY_PLAIN
#                cv2.putText(image,current_text,(tl[0], tl[1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
#
#        if current_sign:
#            sign_count += 1
#            coordinates.append(position)

        cv2.imshow('Result', image)
        count = count + 1
        #Write to video
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.write("{}".format(sign_count))
    for pos in coordinates:
        file.write("\n{} {} {} {} {} {}".format(pos[0],pos[1],pos[2],pos[3],pos[4], pos[5]))
    print("Finish {} frames".format(count))
    cv2.destroyAllWindows()
    file.close()
    return 


if __name__ == '__main__':
    main()
