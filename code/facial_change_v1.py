#!/usr/bin/env python
# coding: utf-8

import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import transform as tf
import warnings
warnings.filterwarnings('ignore')

def plot_landmarks(image,face_landmarks):
    '''plot the landmarks on the image
    INPUT: 
          face_landmarks - dict({"key1":(X1,Y1), "key2":(X2,Y2), ...})
    OUTPUT:
          the face landmarks plot of input image
    '''
#     image = face_recognition.load_image_file(filename)
    plt.imshow(image)
    for key, value in face_landmarks.items():
#         print(key)
        plt.scatter(value[0],value[1],s=1)

def get_boundary(face_landmarks,facial_feature,start,end):
    '''
    return the lower or upper boundary of face components
    '''
    X = np.array(face_landmarks[facial_feature]).T[0]
    X = np.hstack((X,X))[start:end]
    Y = np.array(face_landmarks[facial_feature]).T[1]
    Y = np.hstack((Y,Y))[start:end]
    
    if X[0]>X[1]: #flip coordinates if it's not ascending
        X = X[::-1]
        Y = Y[::-1]
    return X, Y
    
def mouth_middle_part_boundary(image):
    '''return the middle part of the face
    INPUT:
        image - array
    OUTPUT: 
        dict(upper boundary of upper lip, lower boundary of bottom lip)
    '''
    # extract facial landmarks
    face_landmarks = face_recognition.face_landmarks(image,model='large')[0]
    
    middle_part_boundary = dict({"top_lip_boundary": get_boundary(face_landmarks,"top_lip",0,7),
                                 "bottom_lip_boundary":get_boundary(face_landmarks,"bottom_lip",0,7)})
    
    xu_min, xu_max = middle_part_boundary["bottom_lip_boundary"][0].min(),middle_part_boundary["bottom_lip_boundary"][0].max()
    H = middle_part_boundary["bottom_lip_boundary"][1] - middle_part_boundary["top_lip_boundary"][1]
    H_mean = H.mean().astype(int)
#     ## quadratic fit top lip boundary and replace
#     x, y = middle_part_boundary["top_lip_boundary"]
#     z = np.polyfit(x, y, 2)#quadratic fitting
#     y = (z[0]*x**2 + z[1]*x + z[2]).astype(int)
#     middle_part_boundary["top_lip_boundary"] = (x,y)
    
    ## modify lower lip boundary
#     x, y = middle_part_boundary["bottom_lip_boundary"]
#     y[3] = y[3]+H_mean/2
#     z = np.polyfit(x[[0,3,-1]], y[[0,3,-1]], 2)#quadratic fitting
#     y = (z[0]*x**2 + z[1]*x + z[2]).astype(int)
#     middle_part_boundary["bottom_lip_boundary"] = (x,y)
    return middle_part_boundary

def skin_mouth_upper_lower_boundary(m_b, d):
    ''' return the upper and lower boundary based on middle_part_boundary
    INPUT:
            d   - num of blocks
            m_b - the output of mouth_middle_part_boundary
    OUTPUT:
            dict("top_skin_bounday", "bottom_skin_bounday")
    parameters:
            L   - horizontal length
            H   - original height of middle part
            r   - transformed height of middle part
            r   - ratio of h to H
    '''
    #upper
    x = m_b["bottom_lip_boundary"][0]
    y = m_b["bottom_lip_boundary"][1]
    z1 = np.polyfit(x, y, 2)#quadratic fitting
    p1 = np.poly1d(z1)
    
    xu_min, xu_max = m_b["bottom_lip_boundary"][0].min(),m_b["bottom_lip_boundary"][0].max()
    H = m_b["bottom_lip_boundary"][1] - m_b["top_lip_boundary"][1]
    H_mean = H.mean().astype(int)
    L = (xu_max-xu_min)*1.2

    x_u = np.linspace(xu_min-0.1*L,xu_max+0.1*L, d+1).astype(int)
    y_u = (z1[0]*x_u**2 + z1[1]*x_u + z1[2]-2.2*H_mean).astype(int)

    ##lower
    x_l = np.linspace(xu_min-0.1*L,xu_max+0.1*L, d+1).astype(int)
    y_l = (z1[0]*x_l**2 + z1[1]*x_l + z1[2]+2.2*H_mean).astype(int)
    return dict({"top_skin_bounday": (x_u,y_u), "bottom_skin_bounday": (x_l, y_l)})


def nearest_upper_row(src_rows, row):
    '''return the nearest upper row of row in src_rows'''
    row_min = row.min()
    distance = src_rows - row_min
    idx = np.max(np.where(distance <= 0)[0])
    return src_rows[idx]

def nearest_lower_row(src_rows, row):
    '''return the nearest lower row of row in src_rows'''
    row_max = row.max()
    distance = src_rows - row_max
    idx = np.max(np.where(distance <= 0)[-1])+1
    return src_rows[idx]

def nearest_left_col(src_cols, col):
    '''return the nearest left col of col in src_cols'''
    col_min = col.min()
    distance = src_cols - col_min
    idx = np.max(np.where(distance <= 0)[0])
    return src_cols[idx] 

def nearest_right_col(src_cols, col):
    '''return the nearest right col of col in src_cols'''
    col_max = col.max()
    distance = src_cols - col_max
    idx = np.max(np.where(distance <= 0)[-1])+1
    return src_cols[idx]

def remove_morphing_part(src, skin_b):
    '''remove the square grid where the morhping part exists
    src     - (nparray) the original mesh grid 
    skin_b  - (dict)    the boundary of skin
    '''
    src_cols, src_rows = np.unique(src[:,0]), np.unique(src[:,1])
    
    col, row = skin_b["top_skin_bounday"]
    u_row = nearest_upper_row(src_rows, row).astype(int)
    l_col = nearest_left_col(src_cols, col).astype(int)
    
    col, row = skin_b["bottom_skin_bounday"]
    b_row = nearest_lower_row(src_rows, row).astype(int)
    r_col = nearest_right_col(src_cols, col).astype(int)
#     print("row: (", u_row, b_row,"), col:(", l_col, r_col,")", sep="")
    print("row: ({},{}); col: ({},{})".format(u_row, b_row,l_col, r_col))
    
    idx = []
    for i in range(len(src)):
        col, row = src[i,0], src[i,1]
        #print(col,row)
        if np.logical_and(np.logical_and(col>=l_col,col<=r_col),(np.logical_and(row<=b_row, row>=u_row))):
            #print("Delete")
            idx.append(i)
    src_cut = np.delete(src, idx, axis = 0)
    print(len(idx),"points have been removed.")
    return src_cut

def area_flatten(area_dict):
    '''flatten area corrdinates the same form as src'''
    x = np.array([])
    y = np.array([])
    for key in area_dict:
        x = np.hstack((x,area_dict[key][0]))
        y = np.hstack((y,area_dict[key][1]))
    return np.vstack((x.astype(int), y.astype(int))).T


# # mouth

def smiling_mouth(ms_b, alpha=1):

    ms_b_smile = ms_b
    x1, y1 = ms_b_smile["bottom_lip_boundary"]
    xu_min, xu_max = ms_b_smile["bottom_lip_boundary"][0].min(),ms_b_smile["bottom_lip_boundary"][0].max()
    H = ms_b_smile["bottom_lip_boundary"][1] - ms_b_smile["top_lip_boundary"][1]
    H_mean = H.mean().astype(int)
    
    #check alpha coefficient
    def alpha_ratio(ms_b_smile):
        x1, y1 = ms_b_smile["bottom_lip_boundary"]
        H_lower = y1.max()-y1.min()
        
        x2, y2 = ms_b_smile["top_lip_boundary"]
        H_upper = y2.max()-y2.min()
        
        if H_upper<H_lower:
            a=1
        else:
            a=0.7
        return a
    alpha = alpha_ratio(ms_b_smile)*alpha
    
    y1[0] = y1[0] + alpha*H_mean #col
    y1[-1] = y1[-1] + alpha*H_mean
#    x1[0] = x1[0] - alpha*H_mean #row
#    x1[-1] = x1[-1] + alpha*H_mean
    z1 = np.polyfit(x1[[0,3,-1]], y1[[0,3,-1]], 2)#quadratic fitting
    
#     print("# x1:",x1, type(x1))
#     print("# y1:",y1, type(y1))

    
    x2, y2 = ms_b_smile["top_lip_boundary"]
    xx2 = np.hstack((x2[0],x2[3],x2[-1]))
    yy2 = np.hstack((y1[0],y2[3],y1[-1]))
    z2 = np.polyfit(xx2, yy2, 2)#quadratic fitting
    
    # alter upper lip
    y1 = (z1[0]*x1**2 + z1[1]*x1 + z1[2]).astype(int)
    y2 = (z2[0]*x2**2 + z2[1]*x2 + z2[2]).astype(int)
    y2[1:6] = y2[1:6]-0.2*alpha*H_mean
    y2[[2,4]] = y2[[2,4]] - 0.05*alpha*H_mean
 
    # uniform step for x
#     x1_l = np.linspace(x1[0],x1[3], 4).astype(int)
#     x1_r = np.linspace(x1[3],x1[-1], 4).astype(int)
#     x1 = np.hstack((x1_l,x1_r[1:]))
    

#     print("# x1:",x1, type(x1))
#     print("# y1:",y1, type(y1))
    
    ms_b_smile["bottom_lip_boundary"]=(x1,y1)
    ms_b_smile["top_lip_boundary"]=(x2, y2)
    
    return ms_b_smile


# # eye


def eye_middle_part_boundary(image, key="left_eye"):
    '''return the middle part of the face
    (1) upper boundary of upper eye
    (2) lower boundary of bottom eye
    '''    
    # extract facial landmarks
    face_landmarks = face_recognition.face_landmarks(image,model='large')[0]

    middle_part_boundary = dict({"top_eye_boundary": get_boundary(face_landmarks,key,0,4),
                                 "bottom_eye_boundary":get_boundary(face_landmarks,key,3,7)})
    return middle_part_boundary

def skin_eye_upper_lower_boundary(m_b, d):
    ''' return the upper and lower boundary based on middle_part_boundary
    d   - num of blocks
    m_b - the output of mouth_middle_part_boundary
    L   - horizontal length
    H   - original height of middle part
    r   - transformed height of middle part
    r   - ratio of h to H
    '''
    ##upper
    x = m_b["top_eye_boundary"][0]
    y = m_b["top_eye_boundary"][1]
    z1 = np.polyfit(x, y, 2)#quadratic fitting
    p1 = np.poly1d(z1)
    
    xu_min, xu_max = m_b["top_eye_boundary"][0].min(),m_b["top_eye_boundary"][0].max()
    H = m_b["bottom_eye_boundary"][1] - m_b["top_eye_boundary"][1]
    H_mean = H.mean().astype(int)
    L = (xu_max-xu_min)*1.2

    x_u = np.linspace(xu_min-0.1*L,xu_max+0.1*L, d+1).astype(int)
    y_u = (z1[0]*x_u**2 + z1[1]*x_u + z1[2]-2.2*H_mean).astype(int)
    
    #lower
    x = m_b["bottom_eye_boundary"][0]
    y = m_b["bottom_eye_boundary"][1]
    z1 = np.polyfit(x, y, 2)#quadratic fitting
    p1 = np.poly1d(z1)
    
    xu_min, xu_max = m_b["bottom_eye_boundary"][0].min(),m_b["bottom_eye_boundary"][0].max()
    H = m_b["bottom_eye_boundary"][1] - m_b["top_eye_boundary"][1]
    H_mean = H.mean().astype(int)
    L = (xu_max-xu_min)*1.2

    x_l = np.linspace(xu_min-0.1*L,xu_max+0.1*L, d+1).astype(int)
    y_l = (z1[0]*x_u**2 + z1[1]*x_u + z1[2]+2.2*H_mean).astype(int)

    
    return dict({"top_skin_bounday": (x_u,y_u), "bottom_skin_bounday": (x_l, y_l)})




def smiling_eye(es_b, alpha=1):
    es_b_smile = es_b
    x1, y1 = es_b_smile["bottom_eye_boundary"]
    xu_min, xu_max = es_b_smile["bottom_eye_boundary"][0].min(),es_b_smile["bottom_eye_boundary"][0].max()
    H = es_b_smile["bottom_eye_boundary"][1] - es_b_smile["top_eye_boundary"][1]
    H_mean = H.mean().astype(int)

    x2, y2 = es_b_smile["top_eye_boundary"]
    
    y1[1:3] = y1[1:3] + alpha/2*H_mean

    es_b_smile["bottom_eye_boundary"]=(x1,y1)
    es_b_smile["top_eye_boundary"]=(x2,y2)
    
    return es_b_smile

def bigger_eye(es_b, alpha=1):
    boundary = es_b
    x1, y1 = boundary["bottom_eye_boundary"]
    xu_min, xu_max = boundary["bottom_eye_boundary"][0].min(),boundary["bottom_eye_boundary"][0].max()
    H = boundary["bottom_eye_boundary"][1] - boundary["top_eye_boundary"][1]
    H_mean = H.mean().astype(int)

    x2, y2 = boundary["top_eye_boundary"]
    
    y1 = y1 - alpha*H_mean

    boundary["bottom_eye_boundary"]=(x1,y1)
    boundary["top_eye_boundary"]=(x2,y2)
    
    return boundary


# # expression change


def facial_expression_change(image, component, change_function, alpha=1, show_process=False):
    ''' return the changed image
    INPUT:  image, component, change_function, alpha (change degree), show_process
    OUTPUT: out
    '''
    rows, cols = image.shape[0], image.shape[1]
    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 20)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0] #original corrdinates

    src_cut = remove_morphing_part(src, component) #remove_morphing_part
    area_coord = area_flatten(component)
    src = np.vstack((src_cut,area_coord))

    changed_component = change_function(component,alpha)

    # mapping & transform
    area_coord = area_flatten(changed_component)
    dst = np.vstack((src_cut, area_coord ))

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = warp(image, tform, output_shape=(rows, cols))
    out = (out*255).astype('uint8') #convert into RGB image
    
    def plot_process(image, src, tform, out, process):#plot the whole process
        rows, cols = image.shape[0], image.shape[1]
        
        if process=="be": #begin & end
            plt.figure(figsize=(6,3))
            plt.subplot(121)
            plt.imshow(image)
            plt.title("Before")
#             plt.axis("off")

            plt.subplot(122)
            plt.imshow(out)
            plt.axis((0, cols, rows, 0))
#             plt.axis("off")
            plt.title("After")
            plt.show()
        elif process=="all":
            plt.figure(figsize=(12,3))
            plt.subplot(141)
            plt.imshow(image)
            plt.title("Before")
#             plt.axis("off")

            plt.subplot(142)
            plt.imshow(image)
            plt.scatter(src[:, 0], src[:, 1], c="blue", s=1)
            plt.axis((0, cols, rows, 0))
            plt.title("Before: grid")
#             plt.axis("off")

            plt.subplot(143)
            plt.imshow(out)
            plt.scatter(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], c="blue", s=1)
            plt.axis((0, cols, rows, 0))
#             plt.axis("off")
            plt.title("After: grid transform")

            plt.subplot(144)
            plt.imshow(out)
            plt.axis((0, cols, rows, 0))
#             plt.axis("off")
            plt.title("After")
            plt.show()
        else:
            print("Error: The wrong type of output images was selected! Choose 'all' or 'end'.")
    
    if show_process:
        plot_process(image, src, tform, out, show_process)
        
    return out

def component_and_skin(image, component_type):
    if component_type=="mouth":
        mounth_b = mouth_middle_part_boundary(image)
        skin_b = skin_mouth_upper_lower_boundary(mounth_b, d=6)
        boundary  = dict(**mounth_b,**skin_b) #combine two dicts
    elif component_type=="left_eye":
        eye_b = eye_middle_part_boundary(image) #left eye
        skin_eye_b = skin_eye_upper_lower_boundary(eye_b, d=6)
        boundary  = dict(**eye_b,**skin_eye_b) #combine two dicts
    elif component_type=="right_eye":
        eye_b = eye_middle_part_boundary(image,"right_eye") #right eye
        skin_eye_b = skin_eye_upper_lower_boundary(eye_b, d=6)
        boundary  = dict(**eye_b,**skin_eye_b) #combine two dicts
    else:
        print("ERROR: Wrong component type. Choose from 'mouth', 'left_eye', 'right_eye'.")
    return boundary  
    

def facial_change(image, component="mouth", change_function=smiling_mouth, alpha=1, process="all"):
    boundary = component_and_skin(image, component)
    out = facial_expression_change(image, boundary, change_function,alpha, process)
    return out


# #  Test


#load image
# # filename = "./test_images/cropped/test0.jpeg"
# # filename = "./test_images/cropped/test1.jpg" #awful example
# # filename = "./test_images/cropped/test2.png"
# # filename = "./test_images/cropped/test3.jpg"
# filename = "./test_images/cropped/test4.jpg"


# image = io.imread(filename)
# image = align(image)
# # out = facial_change(image, alpha=1)
# # # out = facial_change(out, component="left_eye", change_function=bigger_eye, alpha=1,process="all")
# # # out = facial_change(out, component="right_eye", change_function=bigger_eye, alpha=1,process="all")
# # out = facial_change(out, component="left_eye", change_function=smiling_eye, alpha=1,process="all")
# # out = facial_change(out, component="right_eye", change_function=smiling_eye, alpha=1,process="all")


# out = facial_change(image, alpha=1,process=False)
# # out = facial_change(out, component="left_eye", change_function=bigger_eye, alpha=1,process="all")
# # out = facial_change(out, component="right_eye", change_function=bigger_eye, alpha=1,process="all")
# out1 = facial_change(out, component="left_eye", change_function=smiling_eye, alpha=1,process=False)
# out1 = facial_change(out1, component="right_eye", change_function=smiling_eye, alpha=1,process=False)

# plt.figure(figsize=(10,4))
# plt.subplot(131)
# plt.imshow(image);plt.axis("off");plt.title("Original")
# plt.subplot(132)
# plt.imshow(out);plt.axis("off");plt.title("Type1: mouth")
# plt.subplot(133)
# plt.imshow(out1);plt.axis("off");plt.title("Type2: mouth+eyes")
# plt.suptitle(filename[-9:-4])
# plt.show()


