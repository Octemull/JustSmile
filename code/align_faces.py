# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
# import numpy as np

def align(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
        print("face it!")
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)

    return  faceAligned
