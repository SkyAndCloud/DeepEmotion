"""
The Slave.
Module for processing the images.
"""
import cv2
import matplotlib.pyplot as plt

def test():
    img_path = raw_input("Please input the image path: ")
    img = cv2.imread(img_path)
    pixel = process_image(img)
    plt.imshow(pixel, cmap='gray')
    plt.show()

def process_image(img):
    """
	Extracts faces from the image using haar cascade, resizes and applies filters.
	:param img: image matrix. Must be grayscale
	::returns faces:: list contatining the cropped face images
	"""
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    tmp = face_cascade.detectMultiScale(img, 1.3, 5)
    rect = tmp[0]
    face = img[rect[1]:(rect[1]+rect[2]), rect[0]:(rect[0]+rect[3])]
    try:
        i = cv2.resize(face, (48, 48))
    except:
        exit(1)
    i = cv2.bilateralFilter(i,15,10,10)
    i = cv2.fastNlMeansDenoising(i,None,4,7,21)
    return i