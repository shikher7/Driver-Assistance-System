import cv2
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import time
from sklearn import svm
import numpy as np
from os import listdir
DIMENSION = 32
NO_OF_CLASSES = 12




def load_dataset():
    signDataset = []
    signLabels = []
    for type in range(NO_OF_CLASSES):
        list = listdir("./dataset/{}".format(type))
        for file in list:
            if '.png' in file:
                path = "./dataset/{}/{}".format(type,file)
                print(file)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (DIMENSION, DIMENSION))
                img = np.reshape(img, [DIMENSION, DIMENSION])
                signDataset.append(img)
                signLabels.append(type)
                time.sleep(0.01)
    return np.array(signDataset), np.array(signLabels)





def get_HOGDescriptor() :
    # Various Parameters used for converting image to HOG format
    derivAperture = 1
    winSigma = -1.
    signedGradient = True
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    cellSize = (10,10)
    nbins = 9
    hog_descriptor = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog_descriptor



if __name__ == '__main__':
    print('Training data..')
    time.sleep(1)
    data, labels = load_dataset()
    print(data.shape)

    print('Getting HoG parameters ...')

    hog = get_HOGDescriptor()

    print('Computing HoG descriptors... ')
    hog_descriptors = []
    for img in data:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    # print('Randomizing and Spliting dataset into training(60%) and test data(40%).. ')
    hog_descriptors_train, hog_descriptors_test, labels_train, labels_test = train_test_split(hog_descriptors, labels, test_size=0.4, random_state=4)
    # train_n = int(0.99 * len(hog_descriptors))
    # data_train, data_test = np.split(data, [train_n])
    # hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    # labels_train, labels_test = np.split(labels, [train_n])


    model = svm.SVC(C=12.5, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.50625, kernel='rbf', probability=False )
    # model.train(hog_descriptors_train, labels_train)
    model.fit(hog_descriptors_train, labels_train)
    print('SVM model training completed ...')
    print('Testing model..')

    response = model.predict(hog_descriptors_test)
    # print(response)

    # err = (labels_test != response).mean()
    # print(err)
    # print('Accuracy: %.2f %%' % ((1 - err) * 100))



    print("Confusion Matrix:")
    print(classification_report(labels_test, response))
    print("Accuracy: " + str(accuracy_score(labels_test, response) * 100))
    # print('Saving SVM model ...')
    # with open('svm_model.pkl', 'wb') as file:
    #     pickle.dump.0
    #     2(model, file)



# Used in main file for predicting sign using trained SVM model
def getSign(model, data):
    # Converting to recieved image to grayscale
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # Converting Dimension of the image
    img = [cv2.resize(gray,(DIMENSION,DIMENSION))]
    # Calcualting HOG Descriptor for the image
    hog = get_HOGDescriptor()
    hog_descriptors = np.array([hog.compute(img[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    # Predicting the class of the image or sign Type
    pred = model.predict(hog_descriptors)
    return int(pred)

#
# class Model(object):
#     def load(self, fn):
#         self.model.load(fn)
#     def save(self, fn):
#         self.model.save(fn)

# class SVM(Model):
#     # Definig SVM model with paramenter respective to training of images
#     def __init__(self):
#         self.model = cv2.ml.SVM_create()
#         self.model.setGamma(0.50625)
#         self.model.setC(12.5)
#
#         # RBF Kernel is used because it is better in dealing with images as input
#
#         self.model.setKernel(cv2.ml.SVM_RBF)
#         self.model.setType(cv2.ml.SVM_C_SVC)
#
#     def train(self, samples, responses):
#
#         self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#
#     def predict(self, samples):
#
#         return self.model.predict(samples)[1].ravel()