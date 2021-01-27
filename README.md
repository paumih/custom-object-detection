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

**Note: The TensorFlow models repository's code (which contains the object detection API) is continuously updated by the developers. Sometimes they make changes that break functionality with old versions of TensorFlow. IT IS ALWAYS BEST TO USE THE LATEST VERSION OF TENSORFLOW AND DOWNLOAD THE LATEST MODELS REPOSITORY. If you are not using the latest version, clone or download the commit for the version you are using from https://github.com/tensorflow/models repository. For example, if a project was originally done using TensorFlow v1.5 and with the latest GitHub commit of the TensorFlow Object Detection API available at that time and if you're using the latest version of TensorFlow 2.4 and therefore parts of this project do not work anymore, it may be necessary to install TensorFlow v1.5 and use the exact commit from then rather than the most up-to-date version.**

<h4>1b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model</h4>

TensorFlow provides several object detection models (pre-trained on the COCO 2017 dataset with different neural network architectures) in its [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy.  

You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the Faster-RCNN models.

**Important note: If you cannot download a pre-trained model from tf model zoo by simply clicking on it, convert the markdown documentation file into its raw format, copy the download link and open it in a new tab. This hopefully will start the model downloading**

This tutorial will use the CenterNet Resnet50 V1 FPN 512x512 model. Open the downloaded centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar file with a file archiver such as WinZip or 7-Zip and extract the centernet_resnet50_v1_fpn_512x512_coco17_tpu-8 entire folder (not its pieces) to the C:\tensorflow1\models\research\object_detection folder. (Note: The model date and version will likely change in the future, but it should still work with this tutorial.)




<h3>1. Setup Anaconda Virtual Environment. Installing Anaconda, CUDA, cuDNN</h3>

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
2. Open a **the terminal (that has conda Miniconda/Anaconda in the env variables) or an Anaconda Prompt** and run this command: conda create --name tf_gpu tensorflow-gpu   
**This command will create an environment first named with 'tf_gpu' and will install all the packages required by tensorflow-gpu including the cuda and cuDNN compatible verisons.**

The previous command can be broken down into 3 individual commands:  
  * conda create --name tf_gpu  
  * activate tf_gpu 
  * conda install tensorflow-gpu 

3. To test your tensorflow installation follow these steps:  
  * Open Terminal and activate environment using 'activate tf_gpu'.
  * Go to python console using 'python' and then type:
    + import tensorflow as tf
    + sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

As future version of TensorFlow are released you are likely need to continue updating CUDA and cuDNN to the latest supported version.
**Conda will automatically install the correct version of CUDA and cuDNN for the version of TensorFlow you are using**, so you shouldn't have to worry about this.


