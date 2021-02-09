# How to train a custom object detection model with the Tensorflow Object Detection API (TensorFlow2)

[TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

[Object Detection API Installation TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

<h3>1. Setup de object detection API</h3> 

Follow the steps from the Object Detection API Installation for TensorFlow 2 documentation  

Install protobuf but first download it from [this link](https://github.com/protocolbuffers/protobuf/releases)

Extract the downloaded zip, go to the bin folder, copy the protoc.exe file and paste it into the models/research directory  
 
execute: protoc object_detection/protos/*.proto --python_out=.  

Note: The *.proto designating all files does not work protobuf version 3.5 and higher. If you are using version 3.5, you have to go through each file individually. To make this easier, I created a python script that loops through a directory and converts all proto files one at a time.

Use the convert_proto_to_py from this repository in order to do it and run it:  
python convert_proto_to_py.py <path to proto files directory> <path to protoc file>  

**WARNING: You may be required to install MS Build Tools 2015+ in order to make some required packages compatible in windows.**

<h3>2. Create training and testing data</h3> 

Once the setup is complete, it's time to create the training tf records (the data format that the tf object detection API uses)

1. copy the labeled images directory as well as xml_to_csv.py, generate_tfrecord.py files into the **object detection directory**

2. run: python xml_to_csv.py

3. open the generate_tfrecord.py file and replace the labelmap inside the class_text_to_int method with your own label map.

4. run:
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record  

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record  


<h3>3. Configure the training setup</h3> 

1. create a new 'train' directory inside **object detection directory**



