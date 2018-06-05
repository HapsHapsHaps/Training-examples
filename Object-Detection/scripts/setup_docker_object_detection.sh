#!/bin/sh

cd /root

echo copying models folder
cp -r /root/sharedfolder/tensorflow-repo/models-master /tensorflow/tensorflow/models

echo installing dependencies

#Repo with newer protobuf packages
add-apt-repository -y ppa:maarten-fonville/protobuf

#Install dependencies
apt update
apt install -y protobuf-compiler python-pil python-lxml python-tk cython

echo installing python dependencies with pip
pip install Cython
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib

#COCO API and 
echo setting up the COCO api stuff
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
cp -r pycocotools /tensorflow/tensorflow/models/research/

cd /tensorflow/tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

#Needed in path
echo adding to python path
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#Validating
echo validating setup
python object_detection/builders/model_builder_test.py

echo Done.. Moving to second part
sleep 3

# Second part

echo Setting up object detection research

cd /tensorflow/tensorflow/models/research
python setup.py build
python setup.py install

cd slim
pip install -e .

echo done.

echo "Remember to run the following command:"
echo "export PYTHONPATH=$PYTHONPATH:\`pwd\`:\`pwd\`/slim"
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
