#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import skimage.io as io
import matplotlib.pyplot as plt
import time
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

from skimage.transform import resize as skresize
from scipy.misc import imread,imsave
from FaceSDKCopy import face_detect
from facial_change_v1 import *
from align_faces import align
from faceSwapping import face_swapping

parser = argparse.ArgumentParser(description='You can try is with default setting. input_image_path is the path of your image. TYPE = 1 or 2.')
parser.add_argument('--input_image_path', type=str, default = "../test_images/p3/training_04158.jpeg")
parser.add_argument('--mouth_alpha', type=float, default=1)
parser.add_argument('--eye_alpha', type=float, default=1)
parser.add_argument('--type', type=int, default=1)

args = parser.parse_args()

filename    = args.input_image_path
mouth_alpha = args.mouth_alpha
eye_alpha   = args.eye_alpha
type        = args.type

print("input_image_path:",args.input_image_path)
if type==1:
    print("            type:",args.type)
    print("     mouth_alpha:",args.mouth_alpha)
elif type==2:
    print("            type:",args.type)
    print("     mouth_alpha:",args.mouth_alpha)
    print("       eye_alpha:",args.mouth_alpha)
else:
    print("Wrong input!")

print("***************************************************************")



start = time.process_time()
# 1. Load image
# input_path = "../test_images/"

#filename = input_path+"training_07211.jpeg"  #no change
#filename = input_path + "p3/training_04158.jpeg" # one boy/4 people -- check
# filename = input_path + "p3/training_05497.jpeg" #baby + dad (both) -- check
#------------------------------------------------------------------------------
imagename = os.path.splitext(filename)[0].split('/')[-1] #muiltiple persons
image = io.imread(filename)

print("# Load image ...")

plt.figure()
plt.imshow(image)
plt.axis("off")
plt.title("Original image")

# 2. Face Recognition
### ZTC
neutral_face_list, neutral_face_loc_list = face_detect(filename)

print("# Face recognition ...")
plt.figure()
num_of_faces = len(neutral_face_list)
print("num_of_faces:",len(neutral_face_list))
if num_of_faces==0:
    print("# No faces need to change!")
else:
    for i in range(num_of_faces):
        plt.subplot(1,num_of_faces,i+1)
        plt.imshow(neutral_face_list[i])
        plt.title("Face "+str(i+1))
        plt.axis("off")
    
    io.imsave('input.jpg',image)
    
    for i in range(num_of_faces):
        
        ### 3. Change Facial Expression
        print("# Smiling ...")
        ### YY: traditional method
        '''
        def facial_expression_change(face, *args):
            return new_face
        '''    
        face = neutral_face_list[i]
        face = align(face)

        if type==1:
            imsave('output1.jpg',image)
            new_face = facial_change(face,alpha=mouth_alpha,process=False)
            new_face1 = new_face
            imsave('im2_'+str(i+1)+'_type1.jpg', new_face)
        elif type==2:
            imsave('output2.jpg',image)
            new_face = facial_change(face,alpha=mouth_alpha,process=False)
        #    new_face = facial_change(new_face, component="left_eye", change_function=bigger_eye, alpha=eye_alpha,process=False)
        #    new_face = facial_change(new_face, component="right_eye", change_function=bigger_eye, alpha=eye_alphapha,process=False)
            new_face = facial_change(new_face, component="left_eye", change_function=smiling_eye, alpha=eye_alpha,process=False)
            new_face = facial_change(new_face, component="right_eye", change_function=smiling_eye, alpha=eye_alpha,process=False)
            new_face2 = new_face
            imsave('im2_'+str(i+1)+'_type2.jpg', new_face)

        ## 4. Restore Faces
        print("# Generate new image  ...")
        #### JLY
        if type==1:
            face_swapping('output1.jpg', 'im2_'+str(i+1)+'_type1.jpg', i,neutral_face_list, neutral_face_loc_list)
            print("# Type 1 saved!")
        elif type==2:
            face_swapping('output2.jpg', 'im2_'+str(i+1)+'_type2.jpg', i,neutral_face_list, neutral_face_loc_list)
            print("# Type 2 saved!") 
    

    # move all output images and json file into a new folder
    outputs_filename = imagename
    print("# Create new folder:", outputs_filename,"...")
    
    path1 = "./"
    path2 = "../outputs/"+outputs_filename+"/"
    os.mkdir(path2)

    filelist = os.listdir(os.getcwd())
    for files in filelist:
        filename1 = os.path.splitext(files)[1]  # suffix
        filename0 = os.path.splitext(files)[0]  # filename
    #     print()
        m = (filename1 == '.jpg')|(filename1 =='.png')|(filename1 =='.jpeg')|(filename1 =='.json')
        if filename1:
            if m:
                print("move ",filename0+filename1,"; status:",m)
                despath = path2 + filename0+filename1
                shutil.move(path1+filename0+filename1, despath)
    print("# Organize finish!")
    
    # plot results
    output = io.imread('../outputs/'+outputs_filename+'/output'+str(type)+'.jpg')
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Before')
    plt.subplot(122)
    plt.imshow(output)
    plt.axis('off')
    plt.title('After')
    plt.show()

# add log file
f=open('../outputs/'+outputs_filename+'/log.txt',"w")
f.write("input_image_path: "+args.input_image_path+"\n")
if type==1:
    f.write("            type: "+str(args.type)+"\n")
    f.write("     mouth_alpha: "+str(args.mouth_alpha)+"\n")
elif type==2:
    f.write("            type: "+str(args.type)+"\n")
    f.write("     mouth_alpha: "+str(args.mouth_alpha)+"\n")
    f.write("       eye_alpha: "+str(args.mouth_alpha)+"\n")
else:
    f.write("Wrong input!")
f.close()

os.rename('../outputs/'+outputs_filename, '../outputs/'+outputs_filename+'_type'+str(type)+'_'+str(int(time.time())))

elapsed = time.process_time()-start
print("# Time used:",elapsed,'s')

