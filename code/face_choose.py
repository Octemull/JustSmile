# -*- coding: utf-8 -*-
import cv2

global img, point1, point2, face_loc

def on_mouse(event, x, y, flags, param):
    global img, point1, point2, face_loc,cut_img
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:   #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)

        # window size
        cv2.HoughLinesP
        cv2.namedWindow("image",0);
        cv2.resizeWindow("image", 800, 800);
        cv2.imshow('image', img2)
        # cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):    #按住左键拖曳
        # window size
        cv2.HoughLinesP
        cv2.namedWindow("image",0);
        cv2.resizeWindow("image", 800, 800);
        cv2.imshow('image', img2)

        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        # window size
        cv2.HoughLinesP
        cv2.namedWindow("image",0);
        cv2.resizeWindow("image", 800, 800);
        cv2.imshow('image', img2)

        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('./cut_1.jpeg', cut_img)
        top = min_y
        bot = min_y+height
        left=min_x
        right=min_x+width
        face_loc=(top,bot,left,right)


def choose_face(im1):
    global img,cut_img
    img = cv2.imread(im1)
    # window size
    cv2.HoughLinesP
    cv2.namedWindow("image",0);
    cv2.resizeWindow("image", 800, 800);
    

    # cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (face_loc,cut_img)

