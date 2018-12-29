
# coding: utf-8

import cv2
import dlib
import numpy as np
from time import sleep
import sys
import cognitive_face as CF
from FaceSDKCopy import face_detect
from scipy.misc import imread,imsave
from Face_Cord import face_cord

def face_swapping(image_name, processed_image, n,neutral_face_list, neutral_face_loc_list):
    """
    input: name of the original image, the processed image, the number of face you want to change (Example: "orig.jpg","procd.jpg", 0)
    automatically imsaved the output image which replaced the face in the original image with the processed image
    """
    # 把表情变换后的图片大小变成原图片大小
#    neutral_face_list, neutral_face_loc_list = face_detect(image_name)
    print(neutral_face_loc_list)
    im1 = neutral_face_list[n]
    im2 = cv2.imread(processed_image)
    
    w = im1.shape[1]
    h = im1.shape[0]
    im2 = cv2.resize(im2, (w, h))
    
    # 将scipy.misc的转成cv2
    imsave('ori_n.jpg',im1)
    im1 = cv2.imread('ori_n.jpg')
    
    # use 68 face landmarks to detect face
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

    JAW_POINTS = list(range(0, 17))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 35))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_POINTS = list(range(48, 61))
    FACE_POINTS = list(range(17, 68))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # 人脸检测
    face1 = detector(im1, 1)
    face2 = detector(im2, 1)
    
    # 关键点识别
    landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face1[0]).parts()])
    landmarks2 = np.matrix([[p.x, p.y] for p in predictor(im2, face1[0]).parts()])
    
    # 得到要对齐的点，并转换成浮点数
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
    
    points1 = landmarks1[ALIGN_POINTS].astype(np.float64)
    points2 = landmarks2[ALIGN_POINTS].astype(np.float64)
    
    # 归一化
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    
    # 普氏（Procrustes ）分析得到变换矩阵
    """
    Return an affine transformation [s * R | T] such that:
        min sum ||s*R*p1,i + T - p2,i||^2
    """
    #奇异值分解得到sR
    U, S, V = np.linalg.svd(points1.T * points2)
    sR = (U * V).T
    hs = np.hstack(((s2 / s1) * sR, c2.T - (s2 / s1) * sR * c1.T))
    transferM = np.vstack([hs, np.matrix([0., 0., 1.])])
    
    # 定义仿射变换
    def warp_im(im, transferM, w,h):
        output_im = np.zeros(im1.shape, dtype=im1.dtype)
        cv2.warpAffine(im,transferM[:2],(w, h),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
        return output_im
    
    # 将图2转成图1位置
    warp_im2 = warp_im(im2, transferM, w,h)
    
    # 高斯模糊纠正图2颜色
    # 设定高斯内核大小：0.6*瞳距
    blur_amount = int(0.6 * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)))
    if blur_amount % 2 == 0:
            blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(warp_im2, (blur_amount, blur_amount), 0)
    #im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    
    im1 = im1.astype(np.float64)
    im2 = warp_im2.astype(np.float64)
    im1_blur = im1_blur.astype(np.float64)
    im2_blur = im2_blur.astype(np.float64)
    
    im2_after = warp_im2 * im1_blur / im2_blur
    
    # 把第二张图像的特征混合在第一张图像中
    OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]
    FEATHER_AMOUNT = 11
    
    # 得到遮罩
    def get_face_mask(im, landmarks, color):
        im = np.zeros(im.shape[:2], dtype=np.float64)
        for group in OVERLAY_POINTS:
            points = cv2.convexHull(landmarks[group])
            cv2.fillConvexPoly(im, points, color=color) 
        
        im = np.array([im, im, im]).transpose((1, 2, 0))
        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
        return im

    # 得到图2的mask,并仿射变换到图1位置
    mask2 = get_face_mask(im2, landmarks2, 1)
    warped_mask2 = warp_im(mask2, transferM, w,h)
    # 得到图1的mask
    mask1 = get_face_mask(im1, landmarks1, 1)

    combined_mask = np.max([mask1, warped_mask2], axis=0)
    output_im = im1 * (1.0 - combined_mask) + im2_after * combined_mask
    
    face = neutral_face_loc_list['Neutral_Face '+ str(n+1)]
    converted_face_cord = face_cord(face)
    raw_img = cv2.imread(image_name)
    raw_img[converted_face_cord[0] - 30:converted_face_cord[1]+10,converted_face_cord[2]-20:converted_face_cord[3]+20,:] =  output_im
    cv2.imwrite(image_name, raw_img)

