#!/usr/bin/env python
# coding: utf-8
def face_cord(faceRectangle):
    top = faceRectangle["top"]
    left = faceRectangle["left"] 
    width = faceRectangle["width"] 
    height = faceRectangle["height"]
    bot = top + height
    right = left + width
    return (top,bot,left,right)