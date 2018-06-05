# How to train a TensorFlow object classifier (Very much Work in progress!!)
This is a quick run down on how to train a TensorFlow model for object classifiaction with images.
This example focuses on training with a single classifier.

This project requrires docker to be installed.

These are the different topics that will be covered
- Setup
    1. Preparing training data
    2. Process annotations
    3. Prepare configs
    4. Download the TensorFlow research files
    5. Setup docker container
- Training with the dataset
    1. Run preparation script
    2. The actual training
    3. Follow along with the training process (Optional)
    4. Process trained result
    5. Copying the trained results
- Extras
    1. Training with GPU
    2. Evaluate training process with live testing

## Setup

### This tutorial uses two paths you need to adapt to your use case
The first is the folder that you attach to you docker container, for which everything that needs to be kept will reside within.  
Docker attached folder for me is: `$HOME/andet/training/docker-training-shared`  

And inside this folder i created the folder, that will be the working directory for this tutorial.  
The working directory here is called: `object-training`  

Inside the docker container, the `object-training` folder will reside at `$HOME/sharedfolder/object-training`

From now on, this guide will assume that you have created the working directory and have a terminal open at its path.  
Remember, the working directory will be accessible from inside the docker container.  


### 1. Preparing training data (The actual images)
This guide will assume that you already have prepared a dataset to train upon, as described in the guide "How to create and label a dataset to train on".  
If not, I'll quickly show you how to get a hold of one to test with in the following section..

#### If you want to download a dataset
Luckily for you and med, there's a freely available dataset that can bes to test this guide with. **Remeber that this part is only needed if you don't already have your own dataset to test with.**

 So lets use the use the freely available dataset `dataset name` at `dataset url`.

##### Download the dataset
Execute the following command, to download the data set.  

| `curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz`

##### Unpack the dataset
| `tar xzf flower_photos.tgz`  
This should give you a folder called `flower_photos`, with five different folders inside, each with a bunch of images of said flower.

##### final task for downloaded dataset
Something..  

With the dataset now downloaded and prepared so it fits the self created dataset, we can now continue with the guide.

#### If you trained the dataset yourself
If you trained it yourself in accordance with the related guide, it should consist of a folder called images and another called annotations.  
That's t, and we can move on to the next part.

### 2. Process annotations

### 3. Prepare configs

### 4. Download the TensorFlow research files

### 5. Setup docker container

#### Copy files to workdir

`scripts` directory.

#### Start docker container (CPU)
This will start the cotnainer and attach your terminal to the container, so every command will be executed inside of it.  

So execute the following command:  
`docker run --rm -it -name tsCPU -p 8000:8888 -p 6005:6006 -v $HOME/andet/training/docker-training-shared:/root/sharedfolder:Z tensorflow/tensorflow:latest-devel`  

The `--rm` part of the command means it will be deleted the moment you exit the container. Makes it easier for rapid testing, as it will just create a new container the next time you run the command.  

The `:Z` part of the command, is to stop selinux on the host machine from blocking every single interraction with the attached folder from inside the container.  

With the docker container running, and your terminal attached to it. It's time to train with the dataset.

#### Run setup script
From inside the docker container, run the following command to execute the setup script. 
The purpose of this script is to perform all the changes needed, so the tensorflow docker image can be used for training object detection with classification of images.   
Without this, the training will simply not work.  

So execute the following command to run the script:
`$HOME/sharedfolder/scripts/setup_docker_object_detection.sh`

Now also inside the container, execute the following command:  
``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``

## Training with the dataset

### 1. Run preparation script


### 2. The actual training
To train with the dataset you run the following command inside the container, which easily can take 30 minutes to complete.  
`python train.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --train_dir=training/`  

### 3. Follow along with the training process (Optional)
Using Tensorboard, this will give you a statistic representation of how the training process is going.

From another terminal instance, execute the following command to attach a second terminal to the docker container:  
`docker exec -it tsCPU bash`  

Then to start tensorboard, execute the following command, which will keep and eye on the training results and give an overview of how it improves over time:  
`tensorboard --logdir=/tensorflow/tensorflow/models/research/object_detection/training/`

It should start up within 3 seconds, and you can now with your webbrowser go to:  
`http://localhost:6005`

### 4. Process trained result
From the training training snapshots and more will have been generated. This is not the directly usefull result, as these files will first have to be processed.  
So lets go ahead and process them so we can get the final trained model, which is what we want and need for actually testing and using the trained model.

### 5. Copying the trained results
To keep the trained model saved.
 We will copy it from the container, into the attached sharedfolder so it will be kept and can be used to detect and classify objects in images.

## The final piece. Classifying an image
If everything until now has worked. 

## The end.
