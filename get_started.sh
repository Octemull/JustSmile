#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUPUT_DIR=$HEAD_DIR/outputs
CODE_DIR=$HEAD_DIR/code

mkdir -p $OUPUT_DIR

# Download shape_predictor_68_face_landmarks for dlib
cd $CODE_DIR
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Creates the environment
conda create -n justsmile python=3.7

# Activates the environment
source activate justsmile

# pip install into environment
cd $HEAD_DIR
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run two examples
cd $CODE_DIR
python args_just_smile_manual.py   #manual example, use mouse to choose face then press enter twice
python args_just_smile_auto_v1.py  #auto example
