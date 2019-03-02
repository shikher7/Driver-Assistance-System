import cv2
import numpy as np
from os import listdir
DIMENSION = 32
NO_OF_CLASSES = 12


class Model(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(Model):
    # Definig SVM model with paramenter respective to training of images
    def __init__(self):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(0.50625)
        self.model.setC(12.5)

        # RBF Kernel is used because it is better in dealing with images as input

        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):

        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()



def load_dataset():
    signDataset = []
    signLabels = []
    for type in range(NO_OF_CLASSES):
        list = listdir("./dataset/{}".format(type))
        for file in list:
            if '.png' in file:
                path = "./dataset/{}/{}".format(type,file)
                print(path)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (DIMENSION, DIMENSION))
                img = np.reshape(img, [DIMENSION, DIMENSION])
                signDataset.append(img)
                signLabels.append(type)
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
    print('Loading road sign data..')
    data, labels = load_dataset()
    print(data.shape)

    print('Getting HoG parameters ...')

    hog = get_HOGDescriptor()

    print('Computing HoG descriptors... ')
    hog_descriptors = []
    for img in data:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors, labels)

    print('Saving SVM model ...')
    model.save('SVM_model.dat')

