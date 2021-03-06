# Import packages
import numpy as np
import os
import cv2
import tensorflow as tf

# Import utilites
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

def load_model(model_dir):
    """
        Load the Tensorflow model into memory.
        Arguments:
            model_dir:string -> directory of the saved model
    """
    model = tf.saved_model.load(str(model_dir))
    return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = cv2.imread(PATH_TO_IMAGE)
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  cv2.imshow('rgb',image_np)
  cv2.waitKey(0)


CWD_PATH = os.getcwd()
IMAGE_NAME = 'IMG_7077.JPG'

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,'test_images',IMAGE_NAME)

# Path to the saved model dir
saved_model_dir= 'D:/Machine_Learning/TF2-OBJECT-DET-API/models/research/object_detection/inference_graph_2/saved_model'

# Load the Tensorflow model into memory.
detection_model = load_model(saved_model_dir)

# Run the detection model on the given image
show_inference(detection_model, PATH_TO_IMAGE)