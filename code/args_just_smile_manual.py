import sys
import argparse
import skimage.io as io
import matplotlib.pyplot as plt
import time
import os
import shutil
import cv2
from skimage.transform import resize as skresize
from scipy.misc import imread,imsave
from facial_change_v1 import *
from face_choose import choose_face
from align_faces import align
from faceSwapping_manual import face_swapping_manual

parser = argparse.ArgumentParser(description='You can try is with default setting. input_image_path is the path of your image. TYPE = 1 or 2.')
parser.add_argument('--input_image_path', type=str, default = "../test_images/p4/training_07267.jpeg")
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
    print("Wrong Type! Use default settings!")
    type=1

print("***************************************************************")


start = time.process_time()
# 1. Load image
# input_path = "../test_images/"
# filename = input_path+"training_07267.jpeg" #no change


image = io.imread(filename)
imsave("input.jpg",image)

print("# Load image ...")

plt.figure()
plt.imshow(image)
plt.axis("off")
plt.title("Original image")

# 2. Face Choose
### 
face_loc, face=choose_face(filename)
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

# 3. Change Facial Expression

# new_face = align(face)
if type==1:
    new_face = facial_change(face,alpha=mouth_alpha,process=False)
    imsave('im2_type1.jpg', new_face)
elif type==2:
    new_face = facial_change(face,alpha=mouth_alpha,process=False)
    #    new_face = facial_change(new_face, component="left_eye", change_function=bigger_eye, alpha=eye_alpha,process=False)
    #    new_face = facial_change(new_face, component="right_eye", change_function=bigger_eye, alpha=eye_alpha,process=False)
    new_face = facial_change(new_face, component="left_eye", change_function=smiling_eye, alpha=eye_alpha,process=False)
    new_face = facial_change(new_face, component="right_eye", change_function=smiling_eye, alpha=eye_alpha,process=False)
    imsave('im2_type2.jpg', new_face)

## 4. Restore Faces
print("# Generate new image  ...")
if type==1:
    face_swapping_manual(filename,'im2_type1.jpg',face,face_loc)
    print("# Type 1 saved!")
elif type==2:
    face_swapping_manual(filename,'im2_type2.jpg',face,face_loc)
    print("# Type 2 saved!") 


# move all output images and json file into a new folder
imagename = os.path.splitext(filename)[0].split('/')[-1] #muiltiple persons
outputs_filename = imagename
path1 = "./"
path2 = "../outputs/"+outputs_filename+"/"
os.mkdir(path2)

filelist = os.listdir(os.getcwd())
for files in filelist:
    filename1 = os.path.splitext(files)[1]  # suffix
    filename0 = os.path.splitext(files)[0]  # filename
#     print()
    m = (filename1 == '.jpg')|(filename1 =='.png')|(filename1 =='.jpg')|(filename1 =='.jpeg')
    if filename1:
        if m:
            print(filename0,filename1,m)
            despath = path2 + filename0+filename1
            shutil.move(path1+filename0+filename1, despath)
# plot results
output = io.imread('../outputs/'+outputs_filename+'/output.jpg')
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