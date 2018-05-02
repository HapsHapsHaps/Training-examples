# Quick and dirty train and Classify: Flower images
This is a quick run down on how to train a tensorflow trainingresult to classify images.
In this flower example there's five different flower types, that will be used for the detection.  

This project requrires docker to be installed, and is based on the following tutorial as of 02-05-2018:  
https://www.tensorflow.org/tutorials/image_retraining

## Setup

### This tutorial uses two paths you need two adapt to your use case
The first is the folder that you attach to you docker container, for which everything that needs to be kept will reside within.  
Docker attached folder for me is: `/home/jacob/andet/training/docker-training-shared`  

And inside this folder i created the folder, that will be the working directory for this tutorial.  
It's called: `flower`  

Inside the docker container, the `flower` folder will reside at `$HOME/sharedfolder/flower`

### Preparing training data (The actual images)

#### Step 1. Download data set
First make sure you've created and is inside the working folder, that will be attached to the docker container.  

Then execute the following command, to download the data set.  

| `curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz`

#### Step 2. Unpack trainng images
| `tar xzf flower_photos.tgz`  
This should give you a folder called `flower_photos`, with five different folders inside, each with a bunch of images of said flower.

### Start docker container
This will start the cotnainer and attach your terminal to the container, so every command will be executed inside  of it.  

| `sudo docker run --rm -it -p 8001:8888 -v /home/jacob/andet/training/docker-training-shared:/root/sharedfolder:Z tensorflow/tensorflow:latest-devel`  

The `--rm` part of the command means it will be deleted the moment you exit the container. Makes it easier for rapid testing, as it will just create a new container the next time you run the command.  

The `:Z` part of the command, is to stop selinux on the host machine from blocking every single interraction with the attached folder from inside the container.  

With the docker container running, ad your terminal attached to it. It's time to train with the dataset.

## Training with the dataset

### Step 1. The actual training
To train with the dataset you run the following command inside the container, which easily can take 30 minutes to complete.  
| `python /tensorflow/tensorflow/examples/image_retraining/retrain.py --image_dir ~/sharedfolder/flower/flower_photos`  

### Step 2. Copying the trained results
Now to make it easier for the rest of this tutorial, and to keep the trained data saved. We will copy them from the temp directory the reside in, into the attached folder.  
This way you can restart the container without having to do the training again.

| `cp /tmp/output_graph.pb ~/sharedfolder/flower/`  
| `cp /tmp/output_labels.txt ~/sharedfolder/flower/`  

## The final piece. Classifying an image
If everything until now has worked. It should be as simple as just running the following command..

| `python /tensorflow/tensorflow/examples/label_image/label_image.py --graph=$HOME/sharedfolder/flower/output_graph.pb --labels=$HOME/sharedfolder/flower/output_labels.txt --input_layer=Mul --output_layer=final_result --image="$HOME/sharedfolder/flower/flower_photos/daisy/21652746_cc379e0eea_m.jpg"`  

This will give you the classifications for each trained flower type. 0.99 means it's 99 percent certain, that the image contains that specific flower.

If you want to try with another image, just change `"$HOME/sharedfolder/flower/flower_photos/daisy/21652746_cc379e0eea_m.jpg` to point to a different image.

## The end.
