import itertools
import os
from PIL import Image
import io
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import time, progressbar
from object_detection.inference import detection_inference
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.core import data_parser
from object_detection.core import standard_fields as fields

from object_detection.metrics.tf_example_parser import BoundingBoxParser, StringParser, Int64Parser, FloatParser

tf.compat.v1.flags.DEFINE_string('input_tfrecord_path', None,
                                 'Input tf record path')
tf.compat.v1.flags.DEFINE_string('output_path', None,
                                 'Path to the output CSV.')
tf.compat.v1.flags.DEFINE_string('inference_graph', None,
                                 'Path to the inference graph with embedded weights.')
tf.compat.v1.flags.DEFINE_string('class_labels', None,
                                 'Path to classes.pbtxt file.')
tf.compat.v1.flags.DEFINE_string('draw_option', "False",
                                 'Whether or not to save images with boxes drawn')
tf.compat.v1.flags.DEFINE_string('draw_save_path', None,
                                 'If using draw_option, where to save images')

FLAGS = tf.compat.v1.flags.FLAGS
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

class CustomParser(data_parser.DataToNumpyParser):
  """Tensorflow Example proto parser."""

  def __init__(self):
    self.items_to_handlers = {
        fields.InputDataFields.image: StringParser(
                    fields.TfExampleFields.image_encoded),
        fields.InputDataFields.groundtruth_boxes: BoundingBoxParser(
                    fields.TfExampleFields.object_bbox_xmin,
                    fields.TfExampleFields.object_bbox_ymin,
                    fields.TfExampleFields.object_bbox_xmax,
                    fields.TfExampleFields.object_bbox_ymax),
        fields.InputDataFields.groundtruth_classes: Int64Parser(
                    fields.TfExampleFields.object_class_label)
    }
    # Optional
    self.filename = StringParser(fields.TfExampleFields.filename)

  def parse(self, tf_example):
    """
    Parses tensorflow example and returns a tensor dictionary.
    Args:
        tf_example: a tf.Example object.
    Returns:
        A dictionary of the following numpy arrays:
        image               - string containing input image.
        filename            - string containing input filename (optional, None if not specified)
        groundtruth_boxes   - a numpy array containing groundtruth boxes.
        groundtruth_classes - a numpy array containing groundtruth classes.
    """
    results_dict = {}
    parsed = True
    for key, parser in self.items_to_handlers.items():
        results_dict[key] = parser.parse(tf_example)
        parsed &= (results_dict[key] is not None)

    # TODO: need to test
    filename = self.filename.parse(tf_example)
    results_dict['filename'] = filename # could be None

    return results_dict if parsed else None

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

# def process_detections(input_dataset, model, categories, draw_option, draw_save_path):
def process_detections(input_tfrecord_path, model, categories, draw_option, draw_save_path):
    """
    Creates input dataset from tfrecord, runs detection model, compares detection results with ground truth
    Args:
        input_tfrecord_path: path of input tfrecord file
        model: path of detection model .pb file
        categories: ordered array of class IDs
        draw_option: whether or not to visualize and save detections and ground truth boxes
        draw_save_path: where to save visualizations if draw_option is true
    """
    data_parser = CustomParser()
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
    
    # Create dataset from records
    input_dataset = tf.data.TFRecordDataset(input_tfrecord_path)

    with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
        for image_index, record in enumerate(input_dataset):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            decoded_dict = data_parser.parse(example)
            
            if decoded_dict:
                image               = decoded_dict[fields.InputDataFields.image]
                input_tensor        = np.expand_dims(Image.open(io.BytesIO(image)), axis=0)
                groundtruth_boxes   = decoded_dict[fields.InputDataFields.groundtruth_boxes]
                groundtruth_classes = decoded_dict[fields.InputDataFields.groundtruth_classes].astype('uint8')
                detections          = model(input_tensor) # Run model inference
                detection_scores    = detections['detection_scores'][0].numpy()
                detection_boxes     = detections['detection_boxes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD]
                detection_classes   = detections['detection_classes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD].astype('uint8')
                filename            = decoded_dict[fields.InputDataFields.filename]
                filename            = filename.decode('UTF-8') if filename is not None else f'image-{image_index}.png'

                matches = []
                image_index += 1
                
                for i, groundtruth_box in enumerate(groundtruth_boxes):
                    for j, detection_box in enumerate(detection_boxes):
                        iou = compute_iou(groundtruth_box, detection_box)
                        if iou > IOU_THRESHOLD:
                            matches.append([i, j, iou])

                matches = np.array(matches)
                if matches.shape[0] > 0:
                    # Sort list of matches by descending IOU so we can remove duplicate detections
                    # while keeping the highest IOU entry.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    
                    # Remove duplicate detections from the list.
                    matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                    
                    # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                    # our previous sort.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    
                    # Remove duplicate ground truths from the list.
                    matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                    
                for i in range(len(groundtruth_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                        confusion_matrix[groundtruth_classes[i] - 1][int(detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1)] += 1 
                    else:
                        confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                        
                for i in range(len(detection_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                        confusion_matrix[confusion_matrix.shape[0] - 1][int(detection_classes[i] - 1)] += 1
                
                if draw_option:
                    draw(filename, draw_save_path, input_tensor[0], categories,
                        groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, detection_scores)
            else:
                print(f'Skipping image {image_index}')

            bar.update(image_index)

    print(f'Processed {image_index + 1} images')

    return confusion_matrix
    
def display(confusion_matrix, categories, output_path):
    '''
    Displays confusion matrix as pandas df to terminal and saves as CSV
    Args:
      confusion_matrix: matrix to be displayed
      categories: ordered array of class IDs
      output_path: where to save CSV
    '''
    print('\nConfusion Matrix:')
    print(confusion_matrix, '\n')
    results = []

    for i in range(len(categories)):
        id = categories[i]['id'] - 1
        name = categories[i]['name']
        
        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])
        
        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)
        
        results.append({'category' : name, f'precision_@{IOU_THRESHOLD}IOU' : precision, f'recall_@{IOU_THRESHOLD}IOU' : recall})
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)

def draw(image_name, image_path, image, categories, groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, detection_scores):
    '''
    Draws ground truth and detection boxes and labels onto image, saves image
    Args:
      image_name: what to save the image as
      image_path: where to save the image
      image: image as np array
      categories: ordered array of class IDs for detections
      groundtruth_boxes: coordinates of ground truth boxes
      groundtruth_classes: class IDs corresponding to "categories"
      detection_boxes: coordinates of detection boxes from model
      detection_classes: class IDs correpsonding to "categories"
      detection_scores: scores for each detection box
    '''
    # Convert category array to category index
    cat_index = {}
    for i, item in enumerate(categories):
        cat_index[i+1] = item

    # Draw ground truth
    '''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    NOTE: as of 11.17.2020, viz_util's ground truth
    option won't print the class name with the box.
    To see the class name for ground truth boxes, add

    class_name = category_index[classes[i]]['name']
    box_to_display_str_map[box].append(class_name)

    under "if scores is None:" (line 1182)
    '''
    image_viz = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_viz,
            groundtruth_boxes,
            groundtruth_classes,
            None, # scores = None triggers ground truth mode
            cat_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=CONFIDENCE_THRESHOLD,
            agnostic_mode=False,
            line_thickness=3,
            groundtruth_box_visualization_color='blue') # opencv will make this red
    # Draw detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_viz,
            detection_boxes,
            detection_classes,
            detection_scores,
            cat_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=CONFIDENCE_THRESHOLD,
            agnostic_mode=False,
            line_thickness=1)

    cv2.imwrite(os.path.join(image_path, image_name), image_viz)

def main(_):
    required_flags = ['input_tfrecord_path', 'output_path',
                      'inference_graph', 'class_labels']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError(f'Flag --{flag_name} is required')
    
    input_tfrecord_path = FLAGS.input_tfrecord_path
    print(input_tfrecord_path)

    draw_option    = FLAGS.draw_option.lower() == 'true'
    draw_save_path = FLAGS.draw_save_path

    if draw_option:
        if draw_save_path is None:
            raise ValueError('If --draw_option is True, --draw_save_path is required')

    # Get class names
    label_map  = label_map_util.load_labelmap(FLAGS.class_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    # Load model
    print('Loading model...')
    model = tf.saved_model.load(FLAGS.inference_graph)

    # Run inference and compute confusion matrix
    print('Evaluating model...')
    confusion_matrix = process_detections(input_tfrecord_path, model, categories, draw_option, draw_save_path)

    # Save to CSV
    print('Saving confusion matrix...')
    display(confusion_matrix, categories, FLAGS.output_path)

    print('Done!')  

if __name__ == '__main__':
    tf.compat.v1.app.run()
