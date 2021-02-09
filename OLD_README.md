# Custom object detection

Training a custom multiple objects detection using TF Object Detection API

This readme describes every step required to get going with your own object detection framework:

<h3>1. Set up TensorFlow Directory structure</h3>

The TensorFlow Object Detection API requires:  
  + using the **specific directory structure** provided in its GitHub repository. 
  + It also requires several additional Python packages (e.g. tensoflow-gpu for windows), 
  + specific additions to the PATH and PYTHONPATH variables,
  + a few extra setup commands to get everything set up to run or train an object detection model.  

This setup process is fairly meticulous, but follow the instructions closely, because **improper setup can cause unwieldy errors down the road.**  


<h4>1a. Download TensorFlow Object Detection API repository from GitHub</h4>

Create a folder directly in C (or any other place) and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.  

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.  

**Note: The TensorFlow models repository's code (which contains the object detection API) is continuously updated by the developers. Sometimes they make changes that break functionality with old versions of TensorFlow. IT IS ALWAYS BEST TO USE THE LATEST VERSION OF TENSORFLOW AND DOWNLOAD THE LATEST MODELS REPOSITORY. If you are not using the latest version, clone or download the commit for the version you are using from https://github.com/tensorflow/models repository. For example, if a project was originally done using TensorFlow v1.5 and with the latest GitHub commit of the TensorFlow Object Detection API available at that time and if you're using the latest version of TensorFlow 2.4 and therefore parts of this project will not work anymore, it may be necessary to install TensorFlow v1.5 and use the exact commit from then rather than the most up-to-date version.**

<h4>1b. Download the centernet_resnet50_v1_fpn_512x512_coco17_tpu-8 model from TensorFlow's model</h4>

TensorFlow provides several object detection models (pre-trained on the COCO 2017 dataset with different neural network architectures) in its [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy.  

You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the Faster-RCNN models.

**Important note: If you cannot download a pre-trained model from tf model zoo by simply clicking on it, convert the markdown documentation file into its raw format, copy the download link and open it in a new tab. This hopefully will start the model downloading.**

This tutorial will use the CenterNet Resnet50 V1 FPN 512x512 model. Open the downloaded centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar file with a file archiver such as WinZip or 7-Zip and extract the centernet_resnet50_v1_fpn_512x512_coco17_tpu-8 entire folder (not its pieces) to the C:\tensorflow1\models\research\object_detection folder. (Note: The model date and version will likely change in the future, but it should still work with this tutorial.)

<h4>1c. Download this tutorial's repository from GitHub</h4>
-
- TBD
-



<h4>1d. Setup Anaconda Virtual Environment. Installing Anaconda, CUDA, cuDNN</h4>

**TensorFlow-GPU allows your PC to use the video card to provide extra processing power while TRAINING.**
Using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8 (3 hours to train instead of 24 hours). 
The CPU-only version of TensorFlow can also be used for this tutorial, but it will take longer. 
**If you use CPU-only TensorFlow, you do not need to install CUDA and cuDNN in Step 1.**

![The deep learning software and hardware stack](/doc_images/dl_sw_hw_stack.png)


**Conventional Approach** (not recommended). 
To install Tensorflow for GPU follow the steps (steps are for windows) :
1. First find if the GPU is compatible with Tensorflow GPU or not! 
2. Download and Install Cuda Toolkit from here: https://developer.nvidia.com/cuda-toolkit-archive
3. Download cuDNN by signing up on Nvidia Developer Website https://developer.nvidia.com/rdp/cudnn-archive
4. Install cuDNN by extracting the contents of cuDNN into the Toolkit path installed in Step 2. There will be files that you have to replace in CUDA Toolkit Directory.
5. Is that it? No, then you need to check your path variables if CUDA_HOME is present or not. If not, please add it manually.
6. Then check in the path variables if your toolkit paths are available or not.
7. Then finally install Anaconda or Miniconda
8. Creating an Environment with Python and Pip packages installed.
9. Then finally 'pip install tensorflow-gpu'.
10. Test your installation.

**There is a probability of 1% that this process will go right for you! Why? *Because of the version numbering***

Different Versions of Tensorflow support different cuDNN and CUDA Verisons. **Also cuDNN and conda were not a part of conda.**
Here is a table showing which version of TensorFlow/TensorFLow-gpu requires which versions of CUDA and cuDNN on Windows:
https://www.tensorflow.org/install/source_windows

**BEST Approach (steps are for windows)**
1. Install Miniconda or Anaconda or any other setup with Python and conda installed
2. Open a **the terminal (that has conda Miniconda/Anaconda in the env variables) or an Anaconda Prompt** and run this command: conda create --name tf_gpu tensorflow-gpu (or if you want to use a specific version of python for that environment use: conda create --name tf_gpu tensorflow-gpu python=3.6)
**This command will create an environment first named with 'tf_gpu' and will install all the packages required by tensorflow-gpu including the cuda and cuDNN compatible verisons.**

The previous command can be broken down into 3 individual commands:    
  Create an environment  
  * conda create --name tf_gpu (or conda create --name tf_gpu python=3.6)  
  Activate that environment once it has been setup
  * activate tf_gpu   
  Install tensorflow-gpu in this environment by issuing   
  * conda install tensorflow-gpu 

Since we're using Anaconda, installing tensorflow-gpu will also automatically download and install the correct versions of CUDA and cuDNN.  

(Note: You can also use the CPU-only version of TensorFow, but it will run much slower. If you want to use the CPU-only version, just use "tensorflow" instead of "tensorflow-gpu" in the previous command.)

3. To test your tensorflow installation follow these steps:  
  * Open Terminal and activate environment using 'activate tf_gpu'.
  * Go to python console using 'python' and then type:

    TF1.x hello world:  
      + import tensorflow as tf
      + msg = tf.constant('Hello, TensorFlow!')
      + sess = tf.Session()
      + print(sess.run(msg))
      
    TF2.x hello world:  
      + import tensorflow as tf
      + msg = tf.constant('Hello, TensorFlow!')
      + tf.print(msg)

As future versions of TensorFlow-gpu are released you are likely need to continue updating CUDA and cuDNN to the latest supported version.
**Conda will automatically install the correct version of CUDA and cuDNN for the version of TensorFlow you are using**, so you shouldn't have to worry about this.

Install the other necessary packages by issuing the following commands:
(tf-gpu) C:\> conda install -c anaconda protobuf    ->to install protobuf compliler. **It may have been installed already as the tensorflow-gpu dependency**
(tf-gpu) C:\> conda install pillow
(tf-gpu) C:\> conda install lxml
(tf-gpu) C:\> conda install Cython   -> not really necessary but good to have in certain contexts
(tf-gpu) C:\> conda install contextlib2
(tf-gpu) C:\> conda install jupyter
(tf-gpu) C:\> conda install matplotlib
(tf-gpu) C:\> conda install pandas
(tf-gpu) C:\> conda install opencv-python or conda install -c menpo opencv or pip install opencv-python  

You can also use 'pip' package manager to install any of these dependencies

(Note: The 'lxml',‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)


<h4>1e. Configure PYTHONPATH environment variable</h4>

We need to add some folders to the environmental variables

A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):  
 **set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim**  

(set PYTHONPATH=C:\Users\paulmi\Desktop\MachineLearning\tf_obj_det_api_workspace\tensorflow1\models;C:\Users\paulmi\Desktop\MachineLearning\tf_obj_det_api_workspace\tensorflow1\models\research;C:\Users\paulmi\Desktop\MachineLearning\tf_obj_det_api_workspace\tensorflow1\models\research\slim)  

Add the PYTHONPATH variable to the PATH variables the command:  
**set PATH=%PATH%;PYTHONPATH**  
To check if the PYTHONPATH variable was added to the PATH variables type:echo %PATH%  

<h2>WARNING!!!</h2> 

**(Note: Every time the "tf-gpu" virtual environment is exited or the (anaconda) command prompt that was used to set those variables is closed, the PYTHONPATH variable is reset and needs to be set up again. You can use echo %PYTHONPATH% to see if it has been set or not.)**

**ALTERNATIVE: either make the built binaries available to your path python path, or simply copy the directories slim and object_detection to your [Anaconda3]/Lib/site-packages directory !AFTER! THE 1f step was completed**

<h4>1f. Compile Protobufs and run setup.py</h4>

Next, compile the Protobuf files, which are used by TensorFlow to configure model and training hyperparameters.
Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. **Every .proto file** in the \object_detection\protos directory **must be called out individually** by the command.

In the Anaconda Command Prompt, change directories to the \models\research directory: 
 + (tf-gpu) C:\> cd C:\tensorflow1\models\research  

Then copy and paste the following command into the command line and press Enter:  

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto .\object_detection\protos\fpn.proto

This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

**(Note: TensorFlow occassionally adds new .proto files to the \protos folder. If you get an error saying ImportError: cannot import name 'something_something_pb2' , you may need to update the protoc command to include the new .proto files.)**  

Finally, you need to build and install the packages with:  
  from the C:\tensorflow1\models\research\slim directory:
  <!-- cd [TF-models]\research\slim -->
 + (tf-gpu) C:\tensorflow1\models\research\slim> python setup.py build
  <!-- cd [TF-models]\research -->
 + (tf-gpu) C:\tensorflow1\models\research> python setup.py install  

**If you get the exception error: could not create 'BUILD': Cannot create a file when that file already exists here, delete the BUILD file inside the slim directory first, it will be re-created automatically. Also if CANNOT install the packages running the command from C:\tensorflow1\models\research directory then run the command from C:\tensorflow1\models\research\slim> directory**

