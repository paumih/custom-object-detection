# How to train a custom object detection model with the Tensorflow Object Detection API (TensorFlow2)





<h3>1. Setup de object detection API</h3> 

Follow the steps from the [Object Detection API Installation TensorFlow 2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) documentation  

Install protobuf, but first download it from [this link](https://github.com/protocolbuffers/protobuf/releases)

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

1. create a new 'training' directory inside **object detection directory**  
2. create in this training directory the labelmap.pbtxt file where the id-label configuration is made  
3. choose a model architecture [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) download it and add it to the object detection directory, or a custom directory of models architectures. I chose the **EfficientDet D0 512x512** model



4. The base config for the model can be found inside the configs/tf2 folder. It needs to be changed to point to the custom data and pretrained weights. Some training parameters also need to be changed. Copy this config file in the training directory  

5.


<h3>4.Train the model</h3> 
From the object detection folder execute:
python model_main_tf2.py \
    --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --model_dir=training/training_model_outputs \
    --checkpoint_every_n=20
    --alsologtostderr

checkpoint_every_n=20, means that every 20 iterations, a checkpoint will be saved
training/training_model_outputs <- DOES NOT have to be an existing directory
**TensorBoard**
navigating to the **object_detection directory** and typing
tensorboard --logdir=training/training_model_outputs/train
training/train/training_model_outputs <-is the path where the events file and the checkpoints are saved while training

<h3>5. Export inference graph </h3>

Once the training is finished, to make it easier to use and deploy your model, I recommend converting it to a frozen graph file. This can be done using the exporter_main_v2.py script.

python exporter_main_v2.py 
    --trained_checkpoint_dir=training/training_model_outputs 
    --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config 
    --output_directory inference_graph

The command will use the latest saved weights to create a inference graph
trained_checkpoint_dir=training/training_model_outputs <-has to be the directory where the model checkpoints were saved

output_directory inference_graph <- DOES NOT have to be an existing directory

in the inference_graph directory, an entire structure of files and directories for saving the model into an inference graph

<h3>6. Run the trained model at inference. </h3>  

Inference can be done multiple ways. The most useful one is the realtime inference. In order to perform the inference we relly on some predefined tensorflow objecte detection api utils that can be found in the **object_detection directory** therefore place the inference script inside the object_detection directory

<h3>6. Results</h3> 

![All the custom objects are successfully detected. Also, visually similar objects are not confusing for the model](/doc_images/image_1.png)

![Multiple instances of the same objects are individually detected but some false positives may occur](/doc_images/img3.png)