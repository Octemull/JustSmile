
# coding: utf-8

import cognitive_face as CF
import numpy as np
from scipy.misc import imread,imsave
import matplotlib.pyplot as plt
from Face_Cord import face_cord
import json
import os  
import time

def face_detect(image_name):
    '''
    input: name of the image(Example: "test.jpg")
    Modifies: save the cropped neutral face, all the emotions
    Return: the cropped neutral faces, the locations of the neutral faces
    '''
    KEY = '90ad75d4de8f49d5be69410e6b5b6c6d'
    BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0' 


    # KEY = 'c3ea0ed4935f4a1689a64738bf147e00'
    CF.Key.set(KEY)
    # BASE_URL = 'https://eastasia.api.cognitive.microsoft.com/face/v1.0' 
    CF.BaseUrl.set(BASE_URL)
    image_prefix = image_name.split('.')[0] #get image prefix
    face_out = CF.face.detect(image_name,attributes = 'emotion')
    emotion_list = {}
    neutral_face_loc_list = {}
    neutral_face_count = 1
    neutral_face_list = []
    for i in range(len(face_out)):
        face = face_out[i]
        face_emotion = face['faceAttributes']['emotion']
        emotion_list['Person %d' %(i+1)] = face_emotion
        emotion_tuple = list(face_emotion.items())
        emotion_tuple.sort(key = lambda expression: expression[1],reverse=True)
        expected_emotion = emotion_tuple[0][0]
        if (expected_emotion == 'neutral'):
            # print(face_out)
            face_loc= face['faceRectangle']
            neutral_face_loc_list['Neutral_Face %d' %(neutral_face_count)] = face_loc
            neutral_face_count += 1
            img = imread(image_name)
            neutral_cord = face_cord(face_loc)
            img_dim = img.shape
            height = img_dim[0]
            width  = img_dim[1]
            if (((neutral_cord[0] - 30) >=0) and ((neutral_cord[1]+10) < height) and ((neutral_cord[2]-20) >=0) 
                and ((neutral_cord[3]+20) < width)):
                neutral_face = img[neutral_cord[0]-30:neutral_cord[1]+10,neutral_cord[2]-20:neutral_cord[3]+20,:]
            else:
                neutral_face = img[neutral_cord[0]:neutral_cord[1],neutral_cord[2]:neutral_cord[3],:]
            neutral_face_list.append(neutral_face)
            imsave("%s_neutral_%d.png"%(image_prefix,i),neutral_face)
    with open("./%s_expression.json"%(image_prefix),'w',encoding='utf-8') as json_file:
     json.dump(emotion_list,json_file,ensure_ascii=False) #save emotion as a json file
    return neutral_face_list, neutral_face_loc_list
