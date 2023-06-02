"""
The base code of this script is Matterport implementation of Mask R-CNN model Written by Waleed Abdulla (Copyright (c) 2017 Matterport, Inc.).

*** Order of function calls: ***

* Config the model:
  - call the spot.spotConfig() and add to that
  
* initialize the model class:
  - model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR) 
    - in this init network architecture is built: self.keras_model = self.build(mode=mode, config=config)
      self.build builds the whole multi-task learning framework of maskrcnn
      
      
* initialize the model weights:
  - model.load_weights()
  
* train the model:
  -  model.train(dataset_train, dataset_val, learning_rate=config.lr,epochs=200, layers=config.layers_to_tune); 
     - creates train_generator, val_generator       
     - call self.keras_model.fit_generator function


Author: 
Marzieh Haghighi
"""

import datetime
import math
import multiprocessing
import os
import random
import re
import sys
from collections import OrderedDict
from distutils.version import LooseVersion

import keras_cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.utils as KU
import tensorflow_addons as tfa
from focal_loss import SparseCategoricalFocalLoss
from keras_cv.layers import BaseImageAugmentationLayer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorflow.python.eager import context

import barcodefit.model.model_utils as utils
from barcodefit.model.data_generators import *
from barcodefit.model.networks import *

assert LooseVersion(tf.__version__) >= LooseVersion("2.0")
import json

tf.compat.v1.disable_eager_execution()
import pdb

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

############### Set seeds
seed = 42

# Set seed for NumPy
np.random.seed(seed)

# Set seed for TensorFlow
tf.random.set_seed(seed)

random.seed(seed)

############################################################
# Utility Functions
############################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += "shape: {:20}  ".format(str(array.shape))
        if array.size:
            text += "min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max())
        else:
            text += "min: {:10}  max: {:10}".format("", "")
        text += "  {}".format(array.dtype)
    print(text)


############################################################
#  Proposal Layer
############################################################


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config["config"] = self.config.to_dict()
        config["proposal_count"] = self.proposal_count
        config["nms_threshold"] = self.nms_threshold
        return config

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(
            self.config.PRE_NMS_LIMIT, tf.shape(input=anchors)[1]
        )
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice(
            [scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU
        )
        deltas = utils.batch_slice(
            [deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU
        )
        pre_nms_anchors = utils.batch_slice(
            [anchors, ix],
            lambda a, x: tf.gather(a, x),
            self.config.IMAGES_PER_GPU,
            names=["pre_nms_anchors"],
        )

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice(
            [pre_nms_anchors, deltas],
            lambda x, y: apply_box_deltas_graph(x, y),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors"],
        )

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(
            boxes,
            lambda x: clip_boxes_graph(x, window),
            self.config.IMAGES_PER_GPU,
            names=["refined_anchors_clipped"],
        )

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes,
                scores,
                self.proposal_count,
                self.nms_threshold,
                name="rpn_non_max_suppression",
            )
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(input=proposals)[0], 0)
            proposals = tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])
            return proposals

        #         if self.config.assign_label_mode == "clustering":
        if 0:
            # merge all boxes and scores and then apply non-max suppression
            print("scores.shape", boxes.shape, scores.shape)
            boxes_merged = tf.reshape(
                boxes,
                [
                    tf.shape(input=boxes)[0] * tf.shape(input=boxes)[1],
                    tf.shape(input=boxes)[2],
                ],
            )
            scores_merged = tf.reshape(
                scores, [tf.shape(input=scores)[0] * tf.shape(input=scores)[1]]
            )

            proposals_batches_merged = nms(boxes_merged, scores_merged)
            proposals = tf.keras.backend.repeat_elements(
                tf.expand_dims(proposals_batches_merged, axis=0),
                self.config.IMAGES_PER_GPU,
                axis=0,
            )

        else:
            proposals = utils.batch_slice(
                [boxes, scores], nms, self.config.IMAGES_PER_GPU
            )

        #         proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        proposals = tf.cast(proposals, "float32")

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(None)
            proposals.set_shape(out_shape)
        return proposals

    def compute_output_shape(self, input_shape):
        return None, self.proposal_count, 4


#         return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################


def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, stage5_enabled, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.stage5_enabled = stage5_enabled

    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        config["pool_shape"] = self.pool_shape
        return config

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        #         print('boxes',boxes.shape)#  boxes (4, ?, 4)
        #         print('boxes2',boxes)   #Tensor("proposal_targets/rois:0", shape=(4, ?, 4), dtype=float32)

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)["image_shape"][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))

        #         roi_level =tf.experimental.numpy.log2(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(
            5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32))
        )
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.

        if self.stage5_enabled:
            final_p_layer_k = 5
        else:
            final_p_layer_k = 4

        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, final_p_layer_k + 1)):
            ix = tf.compat.v1.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(
                tf.image.crop_and_resize(
                    feature_maps[i],
                    level_boxes,
                    box_indices,
                    self.pool_shape,
                    method="bilinear",
                )
            )

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(input=box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(input=box_to_level)[0]).indices[
            ::-1
        ]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat(
            [tf.shape(input=boxes)[:2], tf.shape(input=pooled)[1:]], axis=0
        )
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


############################################################
#  Detection Target Layer
############################################################


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(
        tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(input=boxes2)[0]]), [-1, 4]
    )
    b2 = tf.tile(boxes2, [tf.shape(input=boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(input=boxes1)[0], tf.shape(input=boxes2)[0]])
    return overlaps


def detection_targets_graphX(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(
            tf.greater(tf.shape(input=proposals)[0], 0),
            [proposals],
            name="roi_assertion",
        ),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(
        tensor=gt_class_ids, mask=non_zeros, name="trim_gt_class_ids"
    )
    #     gt_masks = tf.gather(gt_masks, tf.compat.v1.where(non_zeros)[:, 0], axis=2,
    #                          name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.compat.v1.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    #     gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(input_tensor=crowd_overlaps, axis=1)
    no_crowd_bool = crowd_iou_max < 0.001

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.compat.v1.where(
        tf.math.logical_and(roi_iou_max < 0.5, no_crowd_bool)
    )[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(input=positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = (
        tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    )
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64),
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    #     transposed_masks = tf.expand_dims(tf.transpose(a=gt_masks, perm=[2, 0, 1]), -1)
    # Pick the right mask for each ROI
    #     roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    #     box_ids = tf.range(0, tf.shape(input=roi_masks)[0])
    #     masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
    #                                      box_ids,
    #                                      config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    #     masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    #     masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(tensor=roi_gt_class_ids, paddings=[(0, N + P)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas


# def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
def detection_targets_graph(proposals0, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    # Assertions
    asserts = [
        tf.Assert(
            tf.math.greater(tf.shape(input=proposals0)[0], 0),
            [proposals0],
            name="roi_assertion",
        ),
    ]
    with tf.control_dependencies(asserts):
        proposals0 = tf.identity(proposals0)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals0, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    #     gt_masks = tf.gather(gt_masks, tf.compat.v1.where(non_zeros)[:, 0], axis=2,
    #                          name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]

    if config.rpn_clustering:
        non_crowd_ix = tf.compat.v1.where(gt_class_ids >= 0)[
            :, 0
        ]  # changed by marzi for rpn clustering - need to double check
    else:
        non_crowd_ix = tf.compat.v1.where(gt_class_ids > 0)[:, 0]

    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    #     gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = crowd_iou_max < 0.001

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= config.positive_roi_iou_min_thr
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.compat.v1.where(
        tf.math.logical_and(
            roi_iou_max < config.negative_roi_iou_max_thr, no_crowd_bool
        )
    )[:, 0]

    #     if 0:
    def set_pos_neg_inds(
        roi_iou_max,
        gt_boxes,
        gt_class_ids,
        negative_indices,
        proposals,
        overlaps,
        proposals0,
        negative_roi_iou_max_thr,
        positive_roi_iou_min_thr,
    ):
        TRAIN_ROIS_PER_IMAGE = config.TRAIN_ROIS_PER_IMAGE
        ROI_POSITIVE_RATIO = config.ROI_POSITIVE_RATIO
        print("roi_iou_max", roi_iou_max.shape, roi_iou_max.min(), roi_iou_max.max())
        print(
            "gt_boxes",
            gt_boxes.shape,
            "proposals",
            proposals.shape,
            "overlaps",
            overlaps.shape,
            "proposals0",
            proposals0.shape,
        )
        #         print('gt_boxes',gt_boxes[:2,:])
        print("gt_class_ids", np.unique(gt_class_ids))
        #         positive_roi_bool = (roi_iou_max >= 0.5)
        #         print('positive_roi_bool',np.sum(positive_roi_bool))
        #         print('negative_indices',negative_indices.shape)

        #         positive_roi_iou_min_thr=0.7
        #         negative_roi_iou_max_thr=0.1

        positive_indices = np.where(roi_iou_max >= positive_roi_iou_min_thr)[0]
        negative_indices = np.where(roi_iou_max < negative_roi_iou_max_thr)[0]
        print("pos neg ind", positive_indices.shape, negative_indices.shape)

        #         if len(positive_indices)==0:
        #             positive_indices=np.where(roi_iou_max >= (roi_iou_max.max()-0.1))[0]
        #         if roi_iou_max>8:
        #         if roi_iou_max < 8:
        assert (
            roi_iou_max.shape[0] > 8
        ), "Overlap of proposals and gt_box is less than 8!"  # denominator can't be 0

        pos_target_counts = int(TRAIN_ROIS_PER_IMAGE * ROI_POSITIVE_RATIO)
        neg_target_counts = int(TRAIN_ROIS_PER_IMAGE * (1 - ROI_POSITIVE_RATIO))

        print("counts", pos_target_counts, neg_target_counts)

        top = np.minimum(pos_target_counts, np.maximum(len(positive_indices), 4))
        bot = np.minimum(neg_target_counts, np.maximum(len(negative_indices), 4))

        positive_indices = np.argpartition(roi_iou_max, -top)[-top:]

        if bot != 0 and bot < len(negative_indices):
            negative_indices = np.random.choice(negative_indices, bot, replace=False)
        #             negative_indices = np.argpartition(roi_iou_max, bot)[:bot]

        #         print(negative_indices)
        #             np.random.shuffle(negative_indices)
        #         print("print(negative_indices)",negative_indices)
        #         positive_indices=np.where(roi_iou_max >= 0.5)[0]

        return [positive_indices.astype(np.int32), negative_indices.astype(np.int32)]

    def lambda_check_overlap(cluster_inp, config):
        indices_ls = tf.numpy_function(
            func=set_pos_neg_inds,
            inp=[
                cluster_inp[0],
                cluster_inp[1],
                cluster_inp[2],
                cluster_inp[3],
                cluster_inp[4],
                cluster_inp[5],
                cluster_inp[6],
                config.negative_roi_iou_max_thr,
                config.positive_roi_iou_min_thr,
            ],
            Tout=[tf.int32, tf.int32],
            name="check_dets",
        )

        indices_ls[0].set_shape([None])
        indices_ls[1].set_shape([None])
        #         clustering_labels_rpn_ls[2].set_shape([None,None])
        #         clustering_labels_rpn_ls[3].set_shape([None,None,4])
        return indices_ls

    if config.assign_label_mode == "clustering":
        indices = KL.Lambda(
            lambda x: lambda_check_overlap(x, config), name="check_dets"
        )(
            [
                roi_iou_max,
                gt_boxes,
                gt_class_ids,
                negative_indices,
                proposals,
                overlaps,
                proposals0,
            ]
        )

        positive_indices = indices[0]
        negative_indices = indices[1]

    else:
        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        #         positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_indices = positive_indices[:positive_count]
    #         positive_count = tf.shape(positive_indices)[0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = tf.shape(input=positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = (
        tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    )
    negative_indices = negative_indices[:negative_count]
    #     negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.math.greater(tf.shape(input=positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64),
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(tensor=roi_gt_class_ids, paddings=[(0, N + P)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])
    return rois, roi_gt_class_ids, deltas


class DetectionTargetLayer(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(DetectionTargetLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        if self.config.USE_RPN_ROIS:
            #         you need to shuffle proposals2 TODO MARZI

            proposals_batches_merged = KL.Lambda(
                lambda t: tf.reshape(
                    t,
                    [
                        tf.shape(proposals)[0] * tf.shape(proposals)[1],
                        tf.shape(proposals)[2],
                    ],
                )
            )(proposals)

            indices = tf.range(
                start=0, limit=tf.shape(proposals_batches_merged)[0], dtype=tf.int32
            )
            shuffled_indices = tf.expand_dims(tf.random.shuffle(indices), axis=1)

            #                 indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
            #                 class_scores = tf.gather_nd(probs, indices)    # Class-specific bounding box deltas
            #                 deltas_specific = tf.gather_nd(deltas, indices)

            merged_shuffled = tf.gather_nd(proposals_batches_merged, shuffled_indices)

            proposals2 = KL.Lambda(
                lambda t: keras.backend.repeat_elements(
                    tf.expand_dims(t, axis=0), proposals.shape[0], axis=0
                )
            )(merged_shuffled)

        else:
            proposals2 = inputs[0]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = utils.batch_slice(
            [proposals2, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets_graph(w, x, y, self.config),
            self.config.IMAGES_PER_GPU,
            names=names,
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]


class DetectionTargetLayer_barcode(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer_barcode, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals0 = inputs[0]
        gt_class_ids_batches = inputs[1]
        gt_boxes_batches = inputs[2]

        #         if self.config.USE_RPN_ROIS:
        proposals_batches_merged = KL.Lambda(
            lambda t: tf.reshape(
                t,
                [
                    tf.shape(proposals0)[0] * tf.shape(proposals0)[1],
                    tf.shape(proposals0)[2],
                ],
            )
        )(proposals0)

        #         gt_boxes=KL.Lambda(lambda t: tf.squeeze(t[0,:,:]))(gt_boxes_batches)

        gt_boxes = gt_boxes_batches[0, :, :]

        print(
            proposals0.shape,
            proposals_batches_merged.shape,
            gt_boxes_batches.shape,
            gt_boxes.shape,
        )
        
        #       (9, 3000, 4) (27000, 4) (None, 150, 4) (150, 4)
        #             proposals2 = KL.Lambda(lambda t: keras.backend.repeat_elements(tf.expand_dims(t, axis=0), proposals.shape[0], axis=0))(proposals_batches_merged)

        #         else:
        #             proposals2=inputs[0]

        #         you need to shuffle proposals2 TODO MARZI

        #         names = ["rois", "target_class_ids", "target_bbox"]
        #         outputs = utils.batch_slice(
        #             [proposals2, gt_class_ids, gt_boxes],
        #             lambda w, x, y: detection_targets_graph(
        #                 w, x, y, self.config),
        #             self.config.IMAGES_PER_GPU, names=names)
        #         print("gt_class_ids",gt_class_ids)
        #         print("target_class_ids",outputs[1])

        # Assertions
        asserts = [
            tf.Assert(
                tf.math.greater(tf.shape(input=proposals_batches_merged)[0], 0),
                [proposals_batches_merged],
                name="roi_assertion",
            ),
        ]
        with tf.control_dependencies(asserts):
            proposals_batches_merged = tf.identity(proposals_batches_merged)

        # Remove zero padding
        proposals, _ = trim_zeros_graph(proposals_batches_merged, name="trim_proposals")
        gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(
            gt_class_ids_batches, non_zeros, axis=1, name="trim_gt_class_ids"
        )

        print(proposals.shape, gt_boxes.shape, gt_class_ids.shape)

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        #         crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]

        gt_class_ids_min = tf.math.reduce_min(gt_class_ids, axis=0)

        #         if self.config.rpn_clustering:
        non_crowd_ix = tf.compat.v1.where(gt_class_ids_min >= 0)[:, 0]
        # changed by marzi for rpn clustering - need to double check
        #         else:
        #             non_crowd_ix = tf.compat.v1.where(gt_class_ids_min > 0)[:, 0]

        #         crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix, axis=1)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        #         crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        #         crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        #         no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = roi_iou_max >= self.config.positive_roi_iou_min_thr
        positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]

        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.compat.v1.where(
            roi_iou_max < self.config.negative_roi_iou_max_thr
        )[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(
            self.config.TRAIN_ROIS_PER_IMAGE * self.config.ROI_POSITIVE_RATIO
        )
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        #             positive_indices = positive_indices[:positive_count]
        #         positive_count = tf.shape(positive_indices)[0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = tf.shape(input=positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.config.ROI_POSITIVE_RATIO
        negative_count = (
            tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        )
        #         negative_indices = negative_indices[:negative_count]
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            pred=tf.math.greater(tf.shape(input=positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64),
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment, axis=1)

        print("roi_gt_class_ids", roi_gt_class_ids.shape)

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= self.config.BBOX_STD_DEV

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(input=negative_rois)[0]
        P = tf.maximum(self.config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
        rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
        #         roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(
            tensor=roi_gt_class_ids,
            paddings=[(0, 0), (0, N + P)],
            name="target_class_ids",
        )
        deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])

        rois_batch = KL.Lambda(
            lambda t: keras.backend.repeat_elements(
                tf.expand_dims(t, axis=0), proposals0.shape[0], axis=0
            ),
            name="rois",
        )(rois)

        deltas_batch = KL.Lambda(
            lambda t: keras.backend.repeat_elements(
                tf.expand_dims(t, axis=0), proposals0.shape[0], axis=0
            ),
            name="target_bbox",
        )(deltas)

        #         names = ["rois", "target_class_ids", "target_bbox"]

        return rois_batch, roi_gt_class_ids, deltas_batch

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]


############################################################
#  Detection Layer
############################################################


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.compat.v1.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[
            :, 0
        ]
        keep = tf.sets.intersection(
            tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0)
        )
        #         keep = tf.sparse_tensor_to_dense(keep)[0]
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD,
        )
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=class_keep)[0]
        class_keep = tf.pad(
            tensor=class_keep, paddings=[(0, gap)], mode="CONSTANT", constant_values=-1
        )
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        #         print("config.DETECTION_MAX_INSTANCES",config.DETECTION_MAX_INSTANCES)
        return class_keep

    # 2. Map over class IDs

    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)

    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    #     keep = tf.sparse_tensor_to_dense(keep)[0]
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(input=class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat(
        [
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis],
            tf.gather(probs, keep),
        ],
        axis=1,
    )

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=detections)[0]
    detections = tf.pad(tensor=detections, paddings=[(0, gap), (0, 0)], mode="CONSTANT")
    return detections


class DetectionLayer(KL.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        config.update({"config": self.config.to_dict()})
        return config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m["image_shape"][0]
        window = norm_boxes_graph(m["window"], image_shape[:2])

        #         rois_batches_merged=tf.reshape(rois, [tf.shape(rois)[0]*tf.shape(rois)[1],\
        #                                               tf.shape(rois)[2]])
        #         rois2=tf.keras.backend.repeat_elements(tf.expand_dims(rois_batches_merged,\
        #                                                               axis=0), rois.shape[0], axis=0)

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph_barcode(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU,
        )

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        print(detections_batch.shape)
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6 + 5],
        )

    #             [self.config.BATCH_SIZE, self.config.POST_NMS_ROIS_INFERENCE*self.config.BATCH_SIZE, 6+5])

    def compute_output_shape(self, input_shape):
        #         return (None, self.config.POST_NMS_ROIS_INFERENCE*self.config.BATCH_SIZE, 6+5)
        return (None, self.config.DETECTION_MAX_INSTANCES, 6 + 5)


############################################################
#  Region Proposal Network (RPN)
############################################################
def rpn_graph(feature_map, anchors_per_location, anchor_stride, name_prefix):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(
        512,
        (3, 3),
        padding="same",
        activation="relu",
        strides=anchor_stride,
        name=name_prefix + "rpn_conv_shared",
    )(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(
        2 * anchors_per_location,
        (1, 1),
        padding="valid",
        activation="linear",
        name=name_prefix + "rpn_class_raw",
    )(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]),
        name=name_prefix + "lambda_resh1",
    )(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name=name_prefix + "rpn_class_xxx")(
        rpn_class_logits
    )

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(
        anchors_per_location * 4,
        (1, 1),
        padding="valid",
        activation="linear",
        name=name_prefix + "rpn_bbox_pred",
    )(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]),
        name=name_prefix + "lambda_resh2",
    )(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def rpn_label_reassignment2(
    input_image,
    rpn_class_probs0,
    rpn_probs0,
    rpn_bbox0,
    rpn_match,
    rpn_match_class,
    anchors,
    gt_boxes,
    input_image_meta,
    RPN_TRAIN_ANCHORS_PER_IMAGE,
    MAX_GT_INSTANCES,
    img_aug,
):
    """
    Inputs:
    input_image:  (9, 256, 256, 4)
    rpn_class_probs0: logits output of the network [batch, 5456, 2]
    rpn_probs: probs output of the network [batch, 5456, 2]
    rpn_bbox: predicted box of the network [batch, 5456, 4]
    rpn_match: gt generated by datagenerator labeling each anchor
    rpn_match_class: (batch, 5120, 1)
    anchors:  [batch, 5456, 4]
    gt_boxes: [batch, MAX_GT_INSTANCES, 4]
    RPN_TRAIN_ANCHORS_PER_IMAGE

    Outputs:
    rpn_psudo_labels: reassigned labels for rpn_match
    input_rpn_bbox_match: deltas for each forground anchor indicating shifts to gt bbox
    input_gt_class_ids_byRPN: input to DetectionTargetLayer which is a model input generated by dataGen in
    load_image_gt function (class_ids) generated for each bbox (gt_boxes)
    gt_boxes_byRPN: detected as forground anchors in normalized cordinats are assumed to be gt bbox as inputs to
    """
    print("rpn_match_class.shape", rpn_match_class.shape, np.unique(rpn_match_class))
    print(
        "rpn_class_logits.shape", rpn_class_probs0.shape, rpn_match.shape, anchors.shape
    )
    #     print('gt_boxes',gt_boxes.shape,utils.denorm_boxes(gt_boxes[0,:,:], (256,256)))

    #     df2 = pd.DataFrame(rpn_probs,columns=['rpn_probs_'+i for i in range()])

    #     pdb.set_trace()
    #     gt_masks, class_ids_batch = create_mask_batch(images)
    epoch = input_image_meta[0, -1]
    image_id = input_image_meta[:, 0]

    batch_size, anchors_len, _ = anchors.shape
    df_rpn_probs = pd.DataFrame(
        np.squeeze(rpn_class_probs0[:, :, 1]).T,
        columns=["rpn_probs_" + str(i) for i in range(batch_size)],
    )

    df_rpn_match_class = pd.DataFrame(
        np.squeeze(rpn_match_class).T,
        columns=["rpn_match_class_" + str(i) for i in range(batch_size)],
    )
    df_rpn_match = pd.DataFrame(
        np.squeeze(rpn_match).T,
        columns=["rpn_match_" + str(i) for i in range(batch_size)],
    )
    df_anchors = pd.DataFrame(
        np.squeeze(anchors[0, :, :]), columns=["y1", "x1", "y2", "x2"]
    )

    gt_bbox_x = np.zeros((anchors.shape[1], 4))
    #     print(gt_boxes.shape,np.where(np.median(rpn_match,axis=0)==1))
    #     gt_bbox_x[np.where(rpn_match[0,:]==1)[0],:]=np.squeeze(gt_boxes[0,:,:])
    gt_bbox_x[: gt_boxes.shape[1], :] = np.squeeze(gt_boxes[0, :, :])

    df_gt_boxes = pd.DataFrame(
        gt_bbox_x, columns=["gt_boxes_" + i for i in ["y1", "x1", "y2", "x2"]]
    )

    #     anchors.shape for RPN_ANCHOR_SCALES = (8,16) is (5120, 4)
    #     anchors.shape for RPN_ANCHOR_SCALES = (8,16,32,64,128) is (5456, 4)
    #     anchors_len=5120

    rpn_class_probs = rpn_class_probs0[:, :anchors_len, :]
    rpn_probs = rpn_probs0[:, :anchors_len, :]
    rpn_bbox = rpn_bbox0[:, :anchors_len, :]
    # maxproject across batches:
    #     print('shared.shape',shared.shape)
    #     print('rpn_class_logits.shape',rpn_class_logits.shape,rpn_probs.shape, rpn_bbox.shape)
    #     print('rpn_class_logits.shape',rpn_class_probs.shape, anchors.shape) #(9, 5456, 2) (9, 5456, 4)
    #     print('rpn_class_probs.shape',rpn_class_probs.min(),rpn_class_probs.max(),rpn_class_probs[0:2,0:2,:])
    #     print('rpn_bbox.shape',rpn_bbox.min(),rpn_bbox.max(),rpn_class_probs[0:2,0:2,:])
    #     print('rpn_match.shape',rpn_match.min(),rpn_match.max(),rpn_match[0:2,0:2,:])

    n_samples = rpn_class_probs.shape[0] * rpn_class_probs.shape[1]
    #     print('n_samples',n_samples,rpn_class_probs.shape,rpn_class_logits.shape)

    fg_median = np.median(rpn_class_probs[:, :, 1], axis=0)
    bg_median = np.median(rpn_class_probs[:, :, 0], axis=0)

    fg_median_ps = np.median(rpn_probs[:, :, 1], axis=0)
    bg_median_ps = np.median(rpn_probs[:, :, 0], axis=0)

    fg_by_ps = np.sum(fg_median_ps > 0.99965000)
    print("fgs: ", np.sum(rpn_match[0, :] == 1), fg_by_ps)

    fg_var_p = np.var(rpn_probs[:, :, 1], axis=0)
    bg_var_p = np.var(rpn_probs[:, :, 0], axis=0)

    fg_var = np.var(rpn_class_probs[:, :, 1], axis=0)
    fg_rpn_bbox_median = np.median(rpn_bbox, axis=0)  # shape: [5456,4]
    fg_rpn_bbox_var = np.var(rpn_bbox, axis=0)
    print("fg_median", fg_median.shape, "fg_rpn_bbox_var", fg_rpn_bbox_var.shape)
    fg_rpn_bbox_var_mean = np.mean(fg_rpn_bbox_var, 1)
    print(
        "fg_rpn_bbox_var2",
        fg_rpn_bbox_var_mean.shape,
        fg_rpn_bbox_var_mean.min(),
        fg_rpn_bbox_var_mean.max(),
    )

    df_rpn_bbox = pd.DataFrame(
        np.squeeze(fg_rpn_bbox_median),
        columns=["rpn_bbox_median_" + i for i in ["y1", "x1", "y2", "x2"]],
    )
    #     pdb.set_trace()
    #     anchors_all=anchors.reshape(n_samples,4)
    #     rpn_match_class_all=rpn_match_class.reshape(n_samples,)
    #     print("fg_median",fg_median.shape,fg_median.min(),fg_median.max(),fg_rpn_bbox_median.shape)
    #     fg_median_all=np.tile(fg_median,(rpn_class_probs.shape[0],1)).reshape(n_samples)
    #     fg_rpn_bbox_median_all=np.tile(fg_rpn_bbox_median,(rpn_class_probs.shape[0],1)).reshape(n_samples,4)

    #     print("fg_rpn_bbox_median_all",fg_rpn_bbox_median_all.shape,fg_rpn_bbox_median_all[:2,:])
    #     print("fg_median_all",fg_median_all.shape,fg_median.min(),fg_median.max())

    prob_arr = rpn_class_probs.reshape(n_samples, 2)
    #     pred_class_arr=np.argmax(prob_arr,axis=1)
    #     print('pred_class_arr',pred_class_arr.shape)

    #     pred_rpn_match=np.zeros((n_samples))
    #     perc90=np.percentile(fg_median,99)
    # #     perc90=4
    #     perc10=np.percentile(fg_median,1)
    #     print("perc90,perc10",perc90,perc10)
    #     pred_rpn_match[fg_median_all>=perc90]=1
    #     pred_rpn_match[fg_median_all<perc10]=-1

    #     top=int((RPN_TRAIN_ANCHORS_PER_IMAGE*9)/2)#48*9
    #     idx_top = np.argpartition(fg_median_all, -top)[-top:]
    #     idx_bot = np.argpartition(fg_median_all, top)[:top]

    fg_by_ps = np.sum(fg_median_ps > 0.99965000)
    idx_top_passed = np.where(fg_median_ps >= 0.99965000)[0]

    half_rpn_tr_ancs = int(RPN_TRAIN_ANCHORS_PER_IMAGE / 2)

    #     if len(idx_top_passed)<half_rpn_tr_ancs and len(idx_top_passed)>20:
    #         top=len(idx_top_passed)

    #     elif len(idx_top_passed)<20:
    #         top=20
    #     else:
    #         top=half_rpn_tr_ancs

    top = half_rpn_tr_ancs
    htop = int(top / 2)
    #     htop=int(top)

    idx_top0 = np.argpartition(fg_median_ps, -htop)[-htop:]

    idx_top = np.argpartition(fg_median_ps, -top)[-top:]
    idx_top0 = np.sort(idx_top[np.argsort(fg_var_p[idx_top])[:htop]])

    #     idx_top0 = np.argpartition(fg_median, -htop)[-htop:]
    #     idx_top = np.argpartition(fg_median, -top)[-top:]

    #     idx_top2 = np.argpartition(fg_median, -top*2)[-top*2:]
    #     idx_bot = np.argpartition(fg_median, top*50)[:top*50]
    idx_bot = np.argpartition(fg_median_ps, top * 20)[: top * 20]
    idx_bot0 = idx_bot[np.argsort(bg_var_p[idx_bot])[: top * 10]]
    #     idx_bot = np.argpartition(fg_median, top*50)[:top*50]
    #     idx_bot = np.argpartition(bg_median, -top*50)[-top*50:]

    #     idx_top_varrpn = np.argpartition(fg_rpn_bbox_var_mean[idx_top2], top)[:top]
    #     idx_top=idx_top2[idx_top_varrpn]

    idx_bot_rand = np.random.choice(idx_bot0, top, replace=False)
    idx_bot_rand0 = np.random.choice(idx_bot0, htop, replace=False)
    #     idx_bot = np.argpartition(fg_median, top)[:top]

    pred_rpn_match = np.zeros((batch_size, anchors_len, 1))

    print("rpn_match==1 len ", sum(rpn_match[0, :, :] == 1)[0])
    print("fg_median_top", fg_median[idx_top].min(), fg_median[idx_top].max())
    print("fg_median_bot", fg_median[idx_bot].min(), fg_median[idx_bot].max())

    print("fg_median_ps_top", fg_median_ps[idx_top].min(), fg_median_ps[idx_top].max())
    print("fg_median_ps_bot", fg_median_ps[idx_bot].min(), fg_median_ps[idx_bot].max())

    #     if (fg_median[idx_top].min()<3) and (fg_median[idx_top].max()>3):

    #         idx_top_passed=np.where(fg_median>=3)[0]
    #         pred_rpn_match[:,idx_top_passed,:]=1
    #         pos_count=len(idx_top_passed)
    #         if pos_count==1:
    #             pos_count=2
    #             idx_top_passed = np.argpartition(fg_median, -pos_count)[-pos_count:]
    #             pred_rpn_match[:,idx_top_passed,:]=1
    # #             idx_bot = np.argpartition(fg_median, top*50)[:top*50]

    # #         pred_rpn_match[:,idx_top,:]=1
    # #         pos_count=pred_rpn_match[pred_rpn_match==1].shape[0]
    #         print('pos_count',pos_count,top)
    #         idx_bot2 = np.argpartition(fg_median, 5*pos_count)[:5*pos_count]
    #         idx_bot_rand = np.random.choice(idx_bot2, pos_count, replace=False)
    #         pred_rpn_match[:,idx_bot_rand,:]=-1

    #         idx_top=np.copy(idx_top_passed)
    #         top=np.copy(pos_count)

    # #         print("cor rpn_match==1 len ",sum(rpn_match.reshape(n_samples,)==1))
    #         print("cor fg_median_top",fg_median[fg_median>=3].min(),fg_median[fg_median>=3].max())
    #         print("cor fg_median_bot",fg_median[idx_bot2].min(),fg_median[idx_bot2].max())

    #     elif fg_median[idx_top].max()<3:
    #         top=5
    #         idx_top = np.argpartition(fg_median, -top)[-top:]
    #         idx_bot = np.argpartition(fg_median, top*50)[:top*50]
    #         idx_bot_rand = np.random.choice(idx_bot, top, replace=False)
    #         pred_rpn_match[:,idx_top,:]=1
    #         pred_rpn_match[:,idx_bot_rand,:]=-1

    #     else:
    #         pred_rpn_match[:,idx_top,:]=1
    #         pred_rpn_match[:,idx_bot_rand,:]=-1

    pred_rpn_match[:, idx_top0, :] = 1
    pred_rpn_match[:, idx_bot_rand0, :] = -1

    #     pred_rpn_match[:,idx_top,:]=1
    #     pred_rpn_match[:,idx_bot_rand,:]=-1

    #     print(fg_rpn_bbox_median_all[np.where(pred_rpn_match ==1)[0],:])

    #     gt_class_id_gt_by_rpn=rpn_match_class_all[pred_rpn_match==1]
    #     gt_class_id_gt_by_rpn[gt_class_id_gt_by_rpn==0]=5

    gt_class_id_gt_by_rpn = np.squeeze(rpn_match_class[:, idx_top, :])
    gt_class_id_gt_by_rpn[gt_class_id_gt_by_rpn == 0] = 5

    #     print("gt_class_id_gt_by_rpn 5 ",)
    #     gt_fg_len=int(gt_class_id_gt_by_rpn.shape[0]/rpn_class_probs.shape[0])
    #     gt_fg_len=int(gt_class_id_gt_by_rpn.shape[0]/batch_size)

    #     gt_fg_len=int(sum(pred_rpn_match==1)/batch_size)
    print(
        "gt_class_id_gt_by_rpn 5 ",
        sum(np.median(gt_class_id_gt_by_rpn, axis=0) == 5),
        gt_class_id_gt_by_rpn.shape,
        top,
    )
    #     gt_class_id_gt_by_rpn=gt_class_id_gt_by_rpn.reshape(rpn_class_probs.shape[0],gt_fg_len)
    #     gt_bbox_anchors=anchors_all[pred_rpn_match==1].reshape(rpn_class_probs.shape[0],gt_fg_len,4)

    #     xx2=utils.denorm_boxes(gt_bbox_anchors, (256,256))
    #     print('derived by anchor',xx2)
    #     print("gt_bbox",gt_bbox.shape)
    #     print("gt_class_id_gt_by_rpn",gt_class_id_gt_by_rpn.shape)
    fg_rpn_bbox_median_all = np.tile(fg_rpn_bbox_median, (batch_size, 1)).reshape(
        batch_size, anchors_len, 4
    )
    #     print("fg_rpn_bbox_median_all.shape",fg_rpn_bbox_median_all.shape,fg_rpn_bbox_median_all[:,0,:])
    # .reshape(n_samples,4)

    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    gt_bbox = utils.apply_box_deltas(
        anchors[:, idx_top, :].reshape(top * batch_size, 4),
        (fg_rpn_bbox_median_all[:, idx_top, :].reshape(top * batch_size, 4))
        * RPN_BBOX_STD_DEV,
    ).reshape(batch_size, top, 4)

    #     pdb.set_trace()
    #     idx_topp=np.where(rpn_match[0,:,:] ==1)[0]
    #     gt_bbox_orig=utils.apply_box_deltas(anchors[0,idx_topp,:],rpn_bbox0[0,idx_topp,:]*RPN_BBOX_STD_DEV)
    #     gt_boxes[
    #     gt_bbox=np.clip(utils.apply_box_deltas(anchors[:,idx_top,:].reshape(top*batch_size,4),
    #     (fg_rpn_bbox_median_all[:,idx_top,:].reshape(top*batch_size,4))*\
    #                 RPN_BBOX_STD_DEV).reshape(batch_size,top,4),0,1)

    #     gt_bbox=utils.apply_box_deltas(anchors[:,idx_top,:].reshape(top*batch_size,4),
    #     (fg_rpn_bbox_median_all[:,idx_top,:].reshape(top*batch_size,4))).reshape(batch_size,top,4)
    #     print('gt_bbox',gt_bbox.min(),gt_bbox.max())

    #     gt_bbox=np.clip(gt_bbox,0,1)
    #     gt_bbox=anchors[:,idx_top,:]

    picked = utils.non_max_suppression(
        np.squeeze(gt_bbox[0, :, :]), fg_median[idx_top], 0.5
    )
    gt_bbox_picked = gt_bbox[:, picked, :]
    gt_class_id_gt_by_rpn2 = gt_class_id_gt_by_rpn[:, picked]
    #     gt_bbox_picked=np.tile(gt_bbox_picked0,(batch_size,1))

    #     print('after nms:',gt_bbox.shape,picked.shape,gt_bbox_picked.shape)
    #     gt_bbox

    gt_bbox_xx = np.zeros((anchors.shape[1], 4))
    gt_bbox_xx[idx_top, :] = gt_bbox[0, :, :]

    df_gt_bbox = pd.DataFrame(
        gt_bbox_xx, columns=["gt_bbox_est_" + i for i in ["y1", "x1", "y2", "x2"]]
    )

    #     print("pred_rpn_match",pred_rpn_match.min(),pred_rpn_match.max())
    #     print("rpn_match_class",rpn_match_class.min(),rpn_match_class.max())
    #     print('rpn acc: ',balanced_accuracy_score(rpn_match.reshape(n_samples,), pred_rpn_match))

    #     from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(
        rpn_match.reshape(
            n_samples,
        ),
        pred_rpn_match.reshape(
            n_samples,
        ),
        labels=[-1, 0, 1],
    )
    #     print(matrix.diagonal()/matrix.sum(axis=1))
    print("rpn match confusion mat:", matrix)
    #     pdb.set_trace()
    #     print("anchors_all[pred_rpn_match==1,:]",anchors_all[pred_rpn_match==1,:])
    #     print('rpn_match.shape',rpn_match.shape)
    #     y = np.bincount((pred_rpn_match+1).astype('int64'))
    #     ii = np.nonzero(y)[0]
    #     print("rpn labels: ",np.vstack((ii,y[ii])).T)

    #     y = np.bincount((rpn_match.reshape(n_samples,)+1).astype('int64'))
    #     ii = np.nonzero(y)[0]
    #     print("rpn gt labels: ",np.vstack((ii,y[ii])).T)

    #     pred_class_arr_by_batch_medianproj=pred_rpn_match.reshape(rpn_class_probs.shape[0],rpn_class_probs.shape[1])[:,:,np.newaxis]
    #     pred_class_arr_by_batch_medianproj=rpn_match
    #     input_rpn_bbox_match=np.zeros((rpn_class_probs.shape[0],rpn_class_probs.shape[1],4))
    #     input_rpn_bbox_match=(fg_rpn_bbox_median_all[np.where(pred_rpn_match ==1)[0],\
    #                                                 :]*RPN_BBOX_STD_DEV).reshape(rpn_class_probs.shape[0],gt_fg_len,4)
    #     input_rpn_bbox_match=(fg_rpn_bbox_median_all[np.where(pred_rpn_match ==1)[0],\
    #                                                 :]).reshape(rpn_class_probs.shape[0],gt_fg_len,4)

    #     input_rpn_bbox_match=fg_rpn_bbox_median_all[:,np.concatenate(idx_top,idx_bot),:]
    input_rpn_bbox_match = fg_rpn_bbox_median_all[:, idx_top0, :]

    fg_rpn_bbox_var_match = np.var(input_rpn_bbox_match, axis=0)
    print("fg_rpn_bbox_var_match", fg_rpn_bbox_var.shape, idx_top.shape, idx_bot.shape)
    fg_rpn_bbox_var_mean_match = np.mean(fg_rpn_bbox_var_match, 1)
    print(
        "fg_rpn_bbox_var_match2",
        fg_rpn_bbox_var_mean_match.shape,
        fg_rpn_bbox_var_mean_match.min(),
        fg_rpn_bbox_var_mean_match.max(),
    )

    #     input_rpn_bbox_match2=np.concatenate((input_rpn_bbox_match,np.zeros((rpn_class_probs.shape[0],gt_fg_len,4))),axis=1)

    #     df_pred_rpn_match_class = pd.DataFrame(np.squeeze(pred_rpn_match).T,columns=['pred_rpn_match_class_'+str(i) for i in range(batch_size)])
    #     print(fg_rpn_bbox_median.shape,fg_var.shape,fg_rpn_bbox_var.shape,bg_median.shape)
    #     d = {'fg_median': np.squeeze(fg_median),'fg_var':np.squeeze(fg_var), 'fg_rpn_bbox_var': np.squeeze(fg_rpn_bbox_var_mean),'bg_median':np.squeeze(bg_median)}
    #     df = pd.DataFrame(data=d)

    #     results_df = pd.concat([df,df_rpn_probs,df_rpn_bbox,df_rpn_match_class,df_pred_rpn_match_class,df_rpn_match,\
    #                            df_anchors,df_gt_bbox,df_gt_boxes], axis=1)
    #     results_df.to_csv('results_df_'+str(epoch)+'_'+str(image_id)+'.csv')

    #     img_aug=False
    #     if img_aug:
    if 0:
        pred_rpn_match2 = np.concatenate((pred_rpn_match, pred_rpn_match), axis=0)
        input_rpn_bbox_match2 = np.concatenate(
            (input_rpn_bbox_match, input_rpn_bbox_match), axis=0
        )
    else:
        pred_rpn_match2 = np.copy(pred_rpn_match)
        input_rpn_bbox_match2 = np.copy(input_rpn_bbox_match)

    print(
        "check",
        pred_rpn_match.shape,
        pred_rpn_match2.shape,
        input_rpn_bbox_match.shape,
        input_rpn_bbox_match2.shape,
        gt_class_id_gt_by_rpn.shape,
        gt_bbox.shape,
    )  # ,gt_bbox_picked.shape)

    return (
        pred_rpn_match2.astype(np.int32),
        input_rpn_bbox_match2.astype(np.float32),
        gt_class_id_gt_by_rpn2.astype(np.int32),
        gt_bbox_picked.astype(np.float32),
    )


def build_rpn_model(anchor_stride, anchors_per_location, depth, name_prefix=""):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = keras.Input(
        shape=[None, None, depth], name=name_prefix + "input_rpn_feature_map"
    )
    outputs = rpn_graph(
        input_feature_map, anchors_per_location, anchor_stride, name_prefix
    )
    #     return KM.Model([input_feature_map], outputs, name="rpn_model")
    return keras.Model(
        inputs=input_feature_map, outputs=outputs, name=name_prefix + "rpn_model"
    )


#


def build_rpn_model_clustering(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = keras.Input(
        shape=[None, None, depth], name="input_rpn_feature_map"
    )
    outputs = rpn_graph_clustering(
        input_feature_map, anchors_per_location, anchor_stride
    )
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################


def fpn_classifier_graph(rois, feature_maps, image_meta, config):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    bc_ref: barcode refrence library array ()

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
        clustering_psudo_labels: [batch, num_rois]
    """
    pool_size = config.POOL_SIZE
    num_classes = config.NUM_CLASSES
    train_bn = config.TRAIN_BN
    fc_layers_size = config.FPN_CLASSIF_FC_LAYERS_SIZE
    stage5_enabled = config.stage5_enabled

    name_prefix = "teach_"

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign(
        [pool_size, pool_size],
        stage5_enabled,
        name=name_prefix + "roi_align_classifier",
    )([rois, image_meta] + feature_maps)

    #     x = KL.RandomRotation(factor=0.5,fill_mode='nearest', interpolation='nearest')(x)
    #     tf.keras.layers.RandomFlip(
    #     mode=HORIZONTAL_AND_VERTICAL, seed=None, **kwargs
    # )

    #     x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.5)(x)

    # Two 1024 FC layers (implemented with Conv2D for consistency)

    #     x_a = KL.TimeDistributed(KL.RandomRotation(factor=0.5,fill_mode='nearest', interpolation='nearest'),
    #                            name="mrcnn_class_randomrotation")(x)
    #     x = KL.TimeDistributed(KL.RandomFlip("horizontal_and_vertical"),
    #                            name="mrcnn_class_randomflip")(x)

    #     x = KL.Concatenate(axis=1, name='mrcnn_class_aug_cat')([x,x_a])

    x = KL.TimeDistributed(
        KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
        name=name_prefix + "mrcnn_class_conv1",
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name=name_prefix + "mrcnn_class_bn1")(
        x, training=train_bn
    )
    x = KL.Activation("relu")(x)
    x = KL.TimeDistributed(
        KL.Conv2D(fc_layers_size, (1, 1)), name=name_prefix + "mrcnn_class_conv2"
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name=name_prefix + "mrcnn_class_bn2")(
        x, training=train_bn
    )
    x = KL.Activation("relu")(x)

    shared = KL.Lambda(
        lambda x: K.squeeze(K.squeeze(x, 3), 2), name=name_prefix + "pool_squeeze"
    )(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes), name=name_prefix + "mrcnn_class_logits"
    )(shared)

    mrcnn_probs = KL.TimeDistributed(
        KL.Activation("softmax"), name=name_prefix + "mrcnn_class"
    )(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(
        KL.Dense(num_classes * 4, activation="linear"),
        name=name_prefix + "mrcnn_bbox_fc",
    )(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name=name_prefix + "mrcnn_bbox")(
            x
        )
    else:
        mrcnn_bbox = KL.Reshape(
            (s[1], num_classes, 4), name=name_prefix + "mrcnn_bbox"
        )(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def fpn_clustering_graph_t(rois, feature_maps, image_meta, gt_ids, config):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    bc_ref: barcode refrence library array ()

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
        clustering_psudo_labels: [batch, num_rois]
    """
    pool_size = config.POOL_SIZE
    num_classes = config.NUM_CLASSES
    bc_ref = config.barcode_ref_array
    bc_match = config.BC_MATCH
    # pre_match_clustering = config.pre_match_clustering
    train_bn = config.TRAIN_BN
    fc_layers_size = config.FPN_CLASSIF_FC_LAYERS_SIZE
    stage5_enabled = config.stage5_enabled
    tau_s = config.tau_s
    tau_g = config.tau_g

    name_prefix = "teach_"

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign(
        [pool_size, pool_size],
        stage5_enabled,
        name=name_prefix + "roi_align_classifier",
    )([rois, image_meta] + feature_maps)

    conv1_layer = KL.Conv2D(
        fc_layers_size,
        (pool_size, pool_size),
        padding="valid",
        name=name_prefix + "conv2d_1",
    )

    x = KL.TimeDistributed(conv1_layer, name=name_prefix + "mrcnn_class_conv1")(x)

    x = KL.TimeDistributed(
        BatchNorm(name=name_prefix + "bn_1"), name=name_prefix + "mrcnn_class_bn1"
    )(x, training=train_bn)
    x = KL.Activation("relu")(x)
    conv2_layer = KL.Conv2D(
        fc_layers_size, (1, 1), padding="valid", name=name_prefix + "conv2d_2"
    )
    x = KL.TimeDistributed(conv2_layer, name=name_prefix + "mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(
        BatchNorm(name=name_prefix + "bn_2"), name=name_prefix + "mrcnn_class_bn2"
    )(x, training=train_bn)
    x = KL.Activation("relu")(x)

    shared = KL.Lambda(
        lambda x: K.squeeze(K.squeeze(x, 3), 2), name=name_prefix + "pool_squeeze"
    )(x)
    #     print("shared",shared)
    #     nClust=5;

    def my_lambda_func(
        x, pred_class_ids, mrcnn_class_logits, gt_ids, num_classes, bc_ref, bc_match
    ):
        clustering_labels_ls = tf.numpy_function(
            func=roi_pseudo_labeling,
            inp=[
                x,
                pred_class_ids,
                mrcnn_class_logits,
                gt_ids,
                num_classes,
                bc_ref,
                bc_match,
                tau_s,
                tau_g,
            ],
            Tout=[tf.int32, tf.int32],
            name="kmeansclus",
        )
        sh = K.int_shape(mrcnn_class_logits)
        clustering_labels_ls[0].set_shape([None, sh[1]])
        clustering_labels_ls[1].set_shape([None, sh[1]])
        #         clustering_labels.set_shape(x.get_shape())
        return clustering_labels_ls

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes, name=name_prefix + "_dense_mrlog"),
        name=name_prefix + "mrcnn_class_logits",
    )(shared)
    #     if config.img_aug:
    if config.assign_label_mode == "clustering":
        mrcnn_class_logits2 = tf.slice(mrcnn_class_logits, [0, 0, 1], [-1, -1, 4])
        #     print("mrcnn_class_logits2",mrcnn_class_logits2)
        pred_class_ids = tf.argmax(
            mrcnn_class_logits2, name="pred_4_lambda_layer", axis=2
        )
    #     print("pred_class_ids",pred_class_ids)

    mrcnn_probs = KL.TimeDistributed(
        KL.Activation("softmax", name=name_prefix + "_act1"),
        name=name_prefix + "mrcnn_class",
    )(mrcnn_class_logits)

    sh = K.int_shape(mrcnn_probs)

    #     if config.img_aug:
    if config.assign_label_mode == "clustering":
        clustering_psudo_labels_ls = KL.Lambda(
            lambda x: my_lambda_func(
                x, pred_class_ids, mrcnn_probs, gt_ids, num_classes, bc_ref, bc_match
            ),
            name="lambda_layer",
        )(shared)

    else:
        clustering_psudo_labels_ls = [gt_ids, gt_ids]

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(
        KL.Dense(
            num_classes * 4, activation="linear", name=name_prefix + "_dense_mrfc"
        ),
        name=name_prefix + "mrcnn_bbox_fc",
    )(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name=name_prefix + "mrcnn_bbox")(
            x
        )
    else:
        mrcnn_bbox = KL.Reshape(
            (s[1], num_classes, 4), name=name_prefix + "mrcnn_bbox"
        )(x)

    return (
        clustering_psudo_labels_ls[0],
        clustering_psudo_labels_ls[1],
        mrcnn_class_logits,
        mrcnn_probs,
        mrcnn_bbox,
    )


def fpn_clustering_graph_s(rois, feature_maps, image_meta, gt_ids, config):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    bc_ref: barcode refrence library array ()

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
        clustering_psudo_labels: [batch, num_rois]
    """
    pool_size = config.POOL_SIZE
    num_classes = config.NUM_CLASSES
    bc_ref = config.barcode_ref_array
    bc_match = config.BC_MATCH
    train_bn = config.TRAIN_BN
    fc_layers_size = config.FPN_CLASSIF_FC_LAYERS_SIZE
    stage5_enabled = config.stage5_enabled
    tau_s = config.tau_s
    tau_g = config.tau_g

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign(
        [pool_size, pool_size], stage5_enabled, name="roi_align_classifier"
    )([rois, image_meta] + feature_maps)

    #     x = KL.RandomRotation(factor=0.5,fill_mode='nearest', interpolation='nearest')(x)
    #     tf.keras.layers.RandomFlip(
    #     mode=HORIZONTAL_AND_VERTICAL, seed=None, **kwargs
    # )

    #     x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.5)(x)

    # Two 1024 FC layers (implemented with Conv2D for consistency)

    #     x_a = KL.TimeDistributed(KL.RandomRotation(factor=0.5,fill_mode='nearest', interpolation='nearest'),
    #                            name="mrcnn_class_randomrotation")(x)
    x = KL.TimeDistributed(
        KL.RandomFlip("horizontal_and_vertical"), name="mrcnn_class_randomflip"
    )(x)

    #     x = KL.Concatenate(axis=1, name='mrcnn_class_aug_cat')([x,x_a])

    x = KL.TimeDistributed(
        KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
        name="mrcnn_class_conv1",
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn1")(x, training=train_bn)
    x = KL.Activation("relu")(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(
        x
    )
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_class_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(
        KL.Dense(num_classes), name="mrcnn_class_logits"
    )(shared)

    #     mrcnn_class_logits2 = tf.slice(mrcnn_class_logits, [0, 0,1], [-1, -1,4])
    #     print("mrcnn_class_logits2",mrcnn_class_logits2)
    #     pred_class_ids = tf.argmax(mrcnn_class_logits2, name="pred_4_lambda_layer", axis=2)
    #     print("pred_class_ids",pred_class_ids)

    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(
        mrcnn_class_logits
    )

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(
        KL.Dense(num_classes * 4, activation="linear"), name="mrcnn_bbox_fc"
    )(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def sklearn_kmeans_foreground(
    X,
    pred_class_ids,
    mrcnn_class_logits,
    gt_ids,
    nClust,
    bc_ref,
    bc_match,
    tau_s,
    tau_g,
):
    """the function called by a lambda layer for clustering and label reassignment

    Inputs:
        X:   CNN extracted features     (n_batches, n_samples_perbacth, n_features)
                         feat_arr       (n_samples, n_features)

        pred_class_ids:  predictions of the CURRENT model (n_batches, n_samples_perbacth)     0,1,2,3

        mrcnn_class_logits:   logits before softmax normalizations  (n_batches, n_samples_perbacth, n_clusters)

        gt_ids:   which is basically target_class_ids      (n_batches, n_samples_perbacth)

        nClust: number of clusters + background (eg. 5)

        bc_ref: barcode refrence library array

        bc_match:  if True projection to the reference library is performed


    Returns:
        psudo_labels_based_on_clustering_reshaped: is psudo_labels_based_on_clustering
        reshaped to


    # psudo_labels_based_on_clustering  (n_batches, n_samples_perbacth,9)
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
        clustering_psudo_labels: [batch, num_rois]
    """

    #     nClust=5
    #     print('shared.shape',X.shape)#,(9, 96, 512)
    #     print('mrcnn_class_logits.shape',mrcnn_class_logits[0:2,0:4,:])#,(9, 96, 5)
    #     print("np.argmax(logit_arr,1)",np.argmax(mrcnn_class_logits,2).shape)

    predicted_class_labels = np.unique(pred_class_ids)
    nonzero_class_columns = np.where(np.max(gt_ids, axis=0) != 0)[0]

    #     seq_probs=np.argmax(mrcnn_class_logits,2)
    #     print("seq_probs",seq_probs[:,0:10].T)

    n_samples = X.shape[0] * X.shape[1]
    n_feats = X.shape[2]
    #     print("gt_ids",gt_ids)
    gt_ids_vec = gt_ids.flatten()
    feat_arr = X.reshape(n_samples, n_feats)
    #     print(mrcnn_class_logits.shape)
    logit_arr = mrcnn_class_logits.reshape(n_samples, mrcnn_class_logits.shape[2])
    #     pred_labels_nonZero=np.argmax(logit_arr,1).flatten()
    #     pred_class_ids=pred_class_ids+1
    #     print(mrcnn_class_logits)
    pred_labels_arr = pred_class_ids.flatten() + 1
    print("pred_labels_arr", pred_labels_arr.shape, pred_labels_arr)
    #     print("pred_labels_nonZero",pred_labels_nonZero)
    #     print("pred_labels_arr",pred_labels_arr[0:10])
    #     print("gt_ids_vec",gt_ids_vec[0:10])
    if 0:
        pca = PCA(n_components=64, whiten=True)
        x_pca = pca.fit_transform(feat_arr)
        norm = np.linalg.norm(x_pca, axis=1)
        feat_arr = x_pca / norm[:, np.newaxis]

    psudo_labels_based_on_clustering = np.zeros(n_samples)
    #     forg_index=np.where(pred_labels_arr!=0)[0]
    forg_index = np.where(gt_ids_vec != 0)[0]
    feat_arr_forg = feat_arr[forg_index, :]
    #     print("forg_index",forg_index,feat_arr_forg.shape)

    #     pred_labels_arr_cents=utils.calculate_cluster_centroids(feat_arr_forg,pred_labels_arr[forg_index],nClust)

    ###     y = np.bincount(pred_labels_arr[forg_index])
    y = np.bincount(gt_ids_vec[forg_index])
    ii = np.nonzero(y)[0]
    print("True Labels: \n", np.vstack((ii, y[ii])).T)

    if len(forg_index) > nClust:

        if 0:
            if len(forg_index) > 70:
                pca = PCA(n_components=64, whiten=True)
                x_pca = pca.fit_transform(feat_arr_forg)
                norm = np.linalg.norm(x_pca, axis=1)
                feat_arr_forg = x_pca / norm[:, np.newaxis]

        #         oversample = RandomOverSampler()
        #         if len(np.unique(gt_ids_vec[forg_index]))>1:
        #             feat_arr_forg2, y = oversample.fit_resample(feat_arr_forg, gt_ids_vec[forg_index]);
        #         else:
        feat_arr_forg2 = feat_arr_forg

        scaler = StandardScaler()
        feat_arr_forg2 = scaler.fit_transform(feat_arr_forg)

        pred_labels_arr_cents = utils.calculate_cluster_centroids(
            feat_arr_forg2, pred_labels_arr[forg_index] - 1, nClust - 1
        )
        #         init_centroids=np.zeros((nClust,pred_labels_arr()))

        from sklearn.cluster import AgglomerativeClustering

        kmlabels = (
            AgglomerativeClustering(n_clusters=nClust - 1).fit(feat_arr_forg2).labels_
        )

        #         from k_means_constrained import KMeansConstrained
        #         size_max_val=int(feat_arr_forg2.shape[0]/3)
        #         size_min_val=int(feat_arr_forg2.shape[0]/5)
        #         clf = KMeansConstrained(n_clusters=nClust-1, size_min=size_min_val, size_max=size_max_val,random_state=0)
        #         clf.fit_predict(feat_arr_forg2)
        #         kmlabels=clf.labels_

        #         kmeans = KMeans(n_clusters=nClust-1, init=pred_labels_arr_cents).fit(feat_arr_forg2)
        #         kmlabels=kmeans.predict(feat_arr_forg2)
        psudo_labels_based_on_clustering[forg_index] = kmlabels + 1
    #         kmeans = KMeans(n_clusters=nClust-1, random_state=0).fit(feat_arr_forg)

    #         kmeans = KMeans(n_clusters=nClust-1,init=np.nan_to_num(pred_labels_arr_cents[1:,:])).fit(feat_arr_forg)
    #         psudo_labels_based_on_clustering[forg_index]=kmeans.labels_+1
    else:
        psudo_labels_based_on_clustering[forg_index] = pred_labels_arr[forg_index]

    if len(forg_index) > nClust:
        #         reassigned_labels=reas_labels2(psudo_labels_based_on_clustering[forg_index],\
        #                                        pred_labels_arr[forg_index])

        # #         reassigned_labels=reas_labels3(kmeans.labels_,\
        # #                                        pred_labels_arr[forg_index],kmeans.cluster_centers_,feat_arr_forg,nClust-1)
        #         reassigned_labels=utils.reas_labels3(kmeans.labels_,\
        #                                        pred_labels_arr_cents,kmeans.cluster_centers_,feat_arr_forg,nClust-1)

        reassigned_labels = utils.reas_labels4(
            kmlabels, pred_labels_arr[forg_index] - 1, nClust - 1
        )

        #         reassigned_labels=kmeans.labels_+1
        # (clustering_labels,kmeans_centers,X)

        logit_arr2 = logit_arr[forg_index, 1:]
        #         print('reassigned_labels',reassigned_labels)
        #         reassigned_labels_probs=logit_arr2[np.arange(len(logit_arr2)),reassigned_labels]

        cross_ent_loss_val = np.round(
            log_loss(reassigned_labels - 1, logit_arr2, labels=[0, 1, 2, 3]), 2
        )
        print("log_loss:", cross_ent_loss_val)

        psudo_labels_based_on_clustering[forg_index] = reassigned_labels

        if (cross_ent_loss_val > 0.7) and (len(predicted_class_labels) > 3):
            print("NOT UPDATING THE LABELS!")
            psudo_labels_based_on_clustering[forg_index] = pred_labels_arr[forg_index]

        else:
            psudo_labels_based_on_clustering[forg_index] = reassigned_labels

    #         if cross_ent_loss_val<0.2:
    #             import sys
    #             sys.exit()
    #         print('correlation:',np.round(np.corrcoef(reassigned_labels, pred_labels_arr[forg_index]-1)[0,1],2))
    #         print('nmi:',np.round(normalized_mutual_info_score(reassigned_labels, pred_labels_arr[forg_index]-1),2))

    y = np.bincount(psudo_labels_based_on_clustering[forg_index].astype("int64"))
    ii = np.nonzero(y)[0]
    print("psudo labels \n: ", np.vstack((ii, y[ii])).T)
    #     psudo_labels_based_on_clustering[np.where(gt_ids_vec!=0)[0]]=0

    pred_labels_arr_reshaped = pred_labels_arr.reshape(X.shape[0], X.shape[1]).astype(
        "int32"
    )
    print("pred_labels_arr_reshaped", pred_labels_arr_reshaped)

    psudo_labels_based_on_clustering_reshaped_prematched = (
        psudo_labels_based_on_clustering.reshape(X.shape[0], X.shape[1]).astype("int32")
    )

    psudo_labels_based_on_clustering_reshaped = np.copy(
        psudo_labels_based_on_clustering_reshaped_prematched
    )

    #     psudo_labels_based_on_clustering_reshaped2=np.copy(psudo_labels_based_on_clustering_reshaped)
    #     gt_labels_reshaped=
    #     print("psudo_labels_based_on_clustering_reshaped",psudo_labels_based_on_clustering_reshaped.shape)  # (9, 96)

    #     psudo_labels_based_on_clustering_reshaped_matched=np.zeros(psudo_labels_based_on_clustering_reshaped.shape);

    #     bc_match=True
    #     print("bc_match",bc_match)
    if bc_match:

        #         for s in range(psudo_labels_based_on_clustering_reshaped.shape[1]):
        for s in nonzero_class_columns:
            spot_seq = psudo_labels_based_on_clustering_reshaped[:, s]
            spot_class_probs = np.squeeze(mrcnn_class_logits[:, s, 1:])  # (9,4)

            print("pre matched        bc", pred_labels_arr_reshaped[:, s])
            print("pre matched clustered", spot_seq)
            if np.any(np.all(bc_ref == spot_seq, axis=1)):

                #                 print(spot_seq)
                potential_bcs_probs = spot_class_probs[
                    np.arange(len(spot_class_probs)), spot_seq.astype(int) - 1
                ]
                selected_bc_prob = np.prod(potential_bcs_probs)
                psudo_labels_based_on_clustering_reshaped[:, s] = spot_seq
                #                 if selected_bc_prob>0.001:
                #                 if selected_bc_prob>=1e-8:
                #                     psudo_labels_based_on_clustering_reshaped[:,s]=spot_seq
                #                 else:
                # #                     psudo_labels_based_on_clustering_reshaped[:,s]=0
                #                     psudo_labels_based_on_clustering_reshaped[:,s]=pred_labels_arr_reshaped[:,s]
                print("orig ", selected_bc_prob, potential_bcs_probs)
            else:

                matched_bc, selected_bc_prob = utils.map_to_closest_barcode(
                    spot_seq, bc_ref, spot_class_probs
                )
                #             psudo_labels_based_on_clustering_reshaped_matched[:,s]=matched_bc
                #                 if selected_bc_prob>0.001:
                if selected_bc_prob >= 1e-8:
                    psudo_labels_based_on_clustering_reshaped[:, s] = matched_bc
                else:
                    #                     psudo_labels_based_on_clustering_reshaped[:,s]=0
                    psudo_labels_based_on_clustering_reshaped[
                        :, s
                    ] = pred_labels_arr_reshaped[:, s]

                print("matched ", selected_bc_prob)
            print("matched_bc", psudo_labels_based_on_clustering_reshaped[:, s])
            print("gt________", gt_ids[:, s])
    #     y = np.bincount(psudo_labels_based_on_clustering_reshaped.flatten()[forg_index].astype('int64'))
    #     ii = np.nonzero(y)[0]
    #     print("matched: \n: ",np.vstack((ii,y[ii])).T)

    #     for s in range(np.min([len(nonzero_class_columns),5])):
    # #         print('pr  ',psudo_labels_based_on_clustering_reshaped2[:,nonzero_class_columns[s]])
    #         print('prm ',psudo_labels_based_on_clustering_reshaped[:,nonzero_class_columns[s]])
    #         print('gt  ',gt_ids[:,s])

    #     print("nmi: ",utils.NMI_clus_class(psudo_labels_based_on_clustering[forg_index],gt_ids_vec[forg_index]))

    #     psudo_labels_based_on_clustering[np.where(gt_ids_vec!=0)[0]]=0
    #     print(psudo_labels_based_on_clustering_reshaped.shape)
    return [psudo_labels_based_on_clustering_reshaped, pred_labels_arr_reshaped]


def roi_pseudo_labeling(
    X,
    pred_class_ids,
    mrcnn_class_logits,
    gt_ids,
    nClust,
    bc_ref,
    bc_match,
    tau_s,
    tau_g,
):
    """the function called by a lambda layer for clustering and label reassignment

    Inputs:
        X:   CNN extracted features     (n_batches, n_samples_perbacth, n_features)
                         feat_arr       (n_samples, n_features)

        pred_class_ids:  predictions of the CURRENT model (n_batches, n_samples_perbacth)     0,1,2,3

        mrcnn_class_logits:   logits before softmax normalizations  (n_batches, n_samples_perbacth, n_clusters)

        gt_ids:   which is basically target_class_ids      (n_batches, n_samples_perbacth)

        nClust: number of clusters + background (eg. 5)

        bc_ref: barcode refrence library array

        bc_match:  if True projection to the reference library is performed


    Returns:
        psudo_labels_based_on_clustering_reshaped: is psudo_labels_based_on_clustering
        reshaped to


    # psudo_labels_based_on_clustering  (n_batches, n_samples_perbacth,9)
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
        clustering_psudo_labels: [batch, num_rois]
    """

    predicted_class_labels = np.unique(pred_class_ids)
    nonzero_class_columns = np.where(np.max(gt_ids, axis=0) != 0)[0]
    print("nonzero_class_columns", nonzero_class_columns.shape, predicted_class_labels)
    #     print("nonzero_class_columns",np.where(np.max(gt_ids,axis=0)!=0),nonzero_class_columns.shape)

    # the below line is needed for rpn_clustering case since the above line remove the zero detected classes by rpn
    # and since target_class_ids are ordered as positive and negative anchors we just need to find the index of maximum non-zero element (need to check again)
    #     nonzero_class_columns=np.arange(0, np.where(np.max(gt_ids,axis=0)!=0)[0][-1])

    #     seq_probs=np.argmax(mrcnn_class_logits,2)
    #     print("seq_probs",seq_probs[:,0:10].T)

    print(
        "gt_ids", gt_ids.shape, X.shape, pred_class_ids.shape, mrcnn_class_logits.shape
    )
    n_samples_perbatch_before_aug = gt_ids.shape[1]
    X = X[:, :n_samples_perbatch_before_aug, :]
    pred_class_ids = pred_class_ids[:, :n_samples_perbatch_before_aug]
    mrcnn_class_logits = mrcnn_class_logits[:, :n_samples_perbatch_before_aug, :]

    n_samples = X.shape[0] * X.shape[1]
    n_feats = X.shape[2]
    #     print("gt_ids",gt_ids)
    gt_ids_vec = gt_ids.flatten()
    feat_arr = X.reshape(n_samples, n_feats)
    #     print(mrcnn_class_logits.shape)
    logit_arr = mrcnn_class_logits.reshape(n_samples, mrcnn_class_logits.shape[2])
    pred_labels_nonZero = np.argmax(logit_arr, 1).flatten()
    #     pred_class_ids=pred_class_ids+1
    print("diff", np.sum(pred_labels_nonZero - pred_class_ids.flatten()))
    pred_labels_arr = pred_class_ids.flatten() + 1
    print(
        "unqs: ",
        np.unique(gt_ids_vec),
        np.unique(pred_labels_nonZero),
        np.unique(pred_class_ids),
    )
    #     print("pred_labels_nonZero",pred_labels_nonZero)
    #     print("pred_labels_arr",pred_labels_arr[0:10])
    #     print("gt_ids_vec",gt_ids_vec[0:10])

    psudo_labels_based_on_clustering = np.zeros(n_samples)
    #     forg_index=np.where(pred_labels_arr!=0)[0]
    forg_index = np.where(gt_ids_vec != 0)[0]
    feat_arr_forg = feat_arr[forg_index, :]

    y = np.bincount(gt_ids_vec[forg_index])
    ii = np.nonzero(y)[0]
    print("True Labels: \n", np.vstack((ii, y[ii])).T)

    psudo_labels_based_on_clustering[forg_index] = pred_labels_arr[forg_index]
    logit_arr2 = logit_arr[forg_index, 1:]

    cross_ent_loss_val = np.round(
        log_loss(pred_labels_arr[forg_index] - 1, logit_arr2, labels=[0, 1, 2, 3]), 2
    )
    print("log_loss:", cross_ent_loss_val)

    y = np.bincount(psudo_labels_based_on_clustering[forg_index].astype("int64"))
    ii = np.nonzero(y)[0]
    print("psudo labels \n: ", np.vstack((ii, y[ii])).T)
    #     psudo_labels_based_on_clustering[np.where(gt_ids_vec!=0)[0]]=0

    psudo_labels_based_on_clustering_reshaped_prematched = (
        psudo_labels_based_on_clustering.reshape(X.shape[0], X.shape[1]).astype("int32")
    )

    psudo_labels_based_on_clustering_reshaped = np.copy(
        psudo_labels_based_on_clustering_reshaped_prematched
    )

    for s in nonzero_class_columns:
        spot_seq = psudo_labels_based_on_clustering_reshaped[:, s]
        print("pre matched clustered", spot_seq)
        spot_class_probs = np.squeeze(mrcnn_class_logits[:, s, 1:])  # (9,4)

        potential_bcs_probs = spot_class_probs[
            np.arange(len(spot_class_probs)), spot_seq.astype(int) - 1
        ]
        selected_bc_prob = np.prod(potential_bcs_probs)

        if selected_bc_prob >= tau_g:
            if bc_match:
                if np.any(np.all(bc_ref == spot_seq, axis=1)):

                    #                     psudo_labels_based_on_clustering_reshaped[:,s]=spot_seq

                    #                     if selected_bc_prob>=tau_g:
                    #                         psudo_labels_based_on_clustering_reshaped[:,s]=spot_seq
                    #                     else:
                    #                         psudo_labels_based_on_clustering_reshaped[:,s]=0
                    #                     psudo_labels_based_on_clustering_reshaped[:,s]=pred_labels_arr_reshaped[:,s]
                    #                     print("orig ",selected_bc_prob,potential_bcs_probs)
                    print("orig ", selected_bc_prob)
                else:

                    matched_bc, selected_bc_prob = utils.map_to_closest_barcode(
                        spot_seq, bc_ref, spot_class_probs
                    )
                    if selected_bc_prob >= tau_g:
                        psudo_labels_based_on_clustering_reshaped[:, s] = matched_bc
                        print("matched ", selected_bc_prob)
                    else:
                        #                         psudo_labels_based_on_clustering_reshaped[:,s]=0
                        psudo_labels_based_on_clustering_reshaped[:, s] = -10

        #                 if selected_bc_prob>0.001:
        #                 if 1:
        #                 if selected_bc_prob>=tau_g:
        #                     psudo_labels_based_on_clustering_reshaped[:,s]=matched_bc
        else:
            #             psudo_labels_based_on_clustering_reshaped[:,s]=0
            psudo_labels_based_on_clustering_reshaped[:, s] = -10
        #             spot_seq[potential_bcs_probs<tau_s]=-10
        #             psudo_labels_based_on_clustering_reshaped[:,s]=spot_seq

        print("matched_bc", psudo_labels_based_on_clustering_reshaped[:, s])
        print("gt________", gt_ids[:, s])
    #         pdb.set_trace()

    return [
        psudo_labels_based_on_clustering_reshaped,
        psudo_labels_based_on_clustering_reshaped_prematched,
    ]


def build_fpn_mask_graph(
    rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True
):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    # (1, 500, 14, 14, 256)
    # (1, 500, 14, 14, 256)
    # (1, 500, 14, 14, 256)
    # (1, 500, 14, 14, 256)

    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")(
        [rois, image_meta] + feature_maps
    )

    # Conv layers
    #     print(x.shape)
    x = KL.TimeDistributed(
        KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1"
    )(x)

    #     print(x.shape)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn1")(x, training=train_bn)
    #     print(x.shape)
    x = KL.Activation("relu")(x)
    #     print(x.shape)
    x = KL.TimeDistributed(
        KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2"
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn2")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(
        KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3"
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn3")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(
        KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4"
    )(x)
    x = KL.TimeDistributed(BatchNorm(), name="mrcnn_mask_bn4")(x, training=train_bn)
    x = KL.Activation("relu")(x)

    x = KL.TimeDistributed(
        KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
        name="mrcnn_mask_deconv",
    )(x)
    x = KL.TimeDistributed(
        KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
        name="mrcnn_mask",
    )(x)
    return x


############################################################
#  Loss Functions
############################################################
# rpn_class_loss_graph(rpn_match, rpn_class_logits)  -> RPN anchor classifier loss
#
# rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox) -> RPN bounding box loss graph
#
# mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids) -> loss for the classifier head of Mask R-CNN
#
# mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox) -> loss for Mask R-CNN bounding box refinement
#
#


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.math.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def smooth_l1_loss2(y_true, y_pred):
    abs_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    res = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
    return tf.reduce_mean(res, axis=-1)


def _huber_loss(y_true, y_pred, delta):

    num_non_zeros = tf.math.count_nonzero(y_true, dtype=tf.float32)

    huber_keras_loss = tf.keras.losses.Huber(
        delta=delta, reduction=tf.keras.losses.Reduction.SUM, name="huber_loss"
    )

    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    huber_loss = huber_keras_loss(y_true, y_pred)

    assert huber_loss.dtype == tf.float32

    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name="huber_loss")

    assert huber_loss.dtype == tf.float32
    return huber_loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.compat.v1.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss

    loss = K.sparse_categorical_crossentropy(
        target=anchor_class, output=rpn_class_logits, from_logits=True
    )
    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, input_rpn_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    input_rpn_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.

    print("input_rpn_bboxoxxx", input_rpn_bbox.shape)
    print("rpn_matchxx", rpn_match.shape)
    print("rpn_bboxxx", rpn_bbox.shape)

    rpn_match = K.squeeze(rpn_match, -1)

    #     mask = tf.equal(rpn_match, 1)
    #     mask = tf.cast(mask, tf.float32)

    indices = tf.compat.v1.where(K.equal(rpn_match, 1))
    #     indices = tf.compat.v1.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)

    #     if config.img_aug:
    if 0:
        n_b = config.IMAGES_PER_GPU * 2
    else:
        n_b = config.IMAGES_PER_GPU * 1
    input_rpn_bbox = batch_pack_graph(input_rpn_bbox, batch_counts, n_b)

    loss = smooth_l1_loss(input_rpn_bbox, rpn_bbox)
    #     mask = tf.not_equal(indices, -1)
    #     mask = tf.cast(mask, tf.float32)

    #     box_loss = _huber_loss(y_true=input_rpn_bbox, y_pred=rpn_bbox, delta=1)
    #     assert box_loss.dtype == tf.float32

    loss = K.switch(
        tf.size(input=loss) > 0,
        tf.math.reduce_mean(loss),
        tf.constant(0.0, dtype=tf.float32),
    )
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, "int64")
    target_class_ids = tf.reshape(target_class_ids, (-1,))

    #     sample_weights=K.not_equal(target_class_ids, -10)
    #     sample_weights= tf.ones(shape=(9,10),dtype=tf.dtypes.float32)
    #     sample_weights=tf.ones_like(target_class_ids)
    #     mask=K.not_equal(target_class_ids, -10)

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(input=pred_class_logits, axis=2)

    #     sample_weights = sample_weights * tf.cast(mask, sample_weights.dtype)
    #     target_class_ids = target_class_ids * tf.cast(mask, target_class_ids.dtype)
    pred_class_ids = tf.reshape(pred_class_ids, (-1,))

    pred_class_logits = tf.reshape(
        pred_class_logits, (-1, K.int_shape(pred_class_logits)[2])
    )

    indices = tf.compat.v1.where(K.not_equal(target_class_ids, -10))
    #     indices = tf.compat.v1.where(tf.math.greater(target_class_ids, 0))
    #     positive_roi_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
    #     # Pick rows that contribute to the loss and filter out the rest.
    pred_class_ids = tf.reshape(tf.gather(pred_class_ids, indices), (1, -1))
    target_class_ids = tf.reshape(tf.gather(target_class_ids, indices), (1, -1))
    #     pred_class_ids =tf.gather(pred_class_ids, indices)
    #     target_class_ids =tf.gather(target_class_ids, indices)
    pred_class_logits = tf.gather_nd(pred_class_logits, indices)
    pred_class_logits = tf.reshape(
        pred_class_logits, (1, -1, K.int_shape(pred_class_logits)[1])
    )

    #     print("pred_class_logits",pred_class_logits)
    #     print("pred_class_ids",pred_class_ids.shape)
    #     print("target_class_ids",target_class_ids.shape)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    #     pred_active=tf.Print(pred_active,[pred_active],"pred_active")
    #     loss_focal = SparseCategoricalFocalLoss(gamma=2,from_logits=True)
    #     loss = loss_focal(y_true=target_class_ids, y_pred=pred_class_logits)#,sample_weight=sample_weights)

    #     loss = K.sparse_categorical_crossentropy(target=target_class_ids,
    #                                              output=pred_class_logits,
    #                                              from_logits=True)
    #     ,ignore_class=-10
    #     # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits
    )

    #     # Erase losses of predictions of classes that are not in the active
    #     # classes of the image.
    loss = loss * pred_active

    #     # Computer loss mean. Use only predictions that contribute
    #     # to the loss to get a correct mean.
    loss = tf.reduce_sum(input_tensor=loss) / tf.reduce_sum(input_tensor=pred_active)
    return loss


def mrcnn_class_loss_graph3(target_class_ids, pred_class_logits, active_class_ids):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, "int64")

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(input=pred_class_logits, axis=2)
    #     print("pred_class_logits",pred_class_logits)
    #     print("pred_class_ids",pred_class_ids.shape)
    #     print("target_class_ids",target_class_ids.shape)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    tf.print(pred_active, "pred_active_print", output_stream=sys.stdout)
    #     pred_active=tf.print(pred_active,[pred_active],"pred_active_print",output_stream=sys.stdout)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits
    )

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(input_tensor=loss) / tf.reduce_sum(input_tensor=pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64
    )
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(
        tf.size(input=target_bbox) > 0,
        smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
        tf.constant(0.0),
    )
    loss = K.mean(loss)
    return loss


############################################################
#  MaskRCNN Class
############################################################
# EMAUpdateCallback(frozen_layers_names=ls_pr, target_model=self.keras_model, ema_decay=0.9)
class EMAUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, frozen_layers_names, target_model, ema_decay=0.99, n_batch=50):
        super(EMAUpdateCallback, self).__init__()

        frozen_layers = []
        trainable_layers = []
        for n in frozen_layers_names:
            frozen_layers.append(target_model.get_layer(n))
            trainable_layers.append(target_model.get_layer(n[len("teach_") :]))

        self.frozen_layers = frozen_layers
        self.trainable_layers = trainable_layers
        self.ema_decay = ema_decay
        self.n_batch = n_batch
        self.batch_counter = 0

    #     def on_epoch_end(self, epoch, logs=None):
    #         for frozen_layer, trainable_layer in zip(self.frozen_layers, self.trainable_layers):
    #             frozen_layer_weights = frozen_layer.get_weights()
    #             trainable_layer_weights = trainable_layer.get_weights()
    #             updated_weights = [self.ema_decay * w1 + (1 - self.ema_decay) * w2 for w1, w2 in zip(frozen_layer_weights, trainable_layer_weights)]
    #             frozen_layer.set_weights(updated_weights)

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.batch_counter % self.n_batch == 0:
            for frozen_layer, trainable_layer in zip(
                self.frozen_layers, self.trainable_layers
            ):
                #                 if
                frozen_layer_weights = frozen_layer.get_weights()
                trainable_layer_weights = trainable_layer.get_weights()
                updated_weights = [
                    self.ema_decay * w1 + (1 - self.ema_decay) * w2
                    for w1, w2 in zip(frozen_layer_weights, trainable_layer_weights)
                ]
                frozen_layer.set_weights(updated_weights)


import copy


class MaskRCNN(object):
    """Encapsulates the Mask RCNN model functionality.
    rpn_class_loss : How well the Region Proposal Network separates background with objetcs
    rpn_bbox_loss : How well the RPN localize objects
    mrcnn_bbox_loss : How well the Mask RCNN localize objects
    mrcnn_class_loss : How well the Mask RCNN recognize each class of object
    mrcnn_mask_loss : How well the Mask RCNN segment objects
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ["training", "inference", "inference2"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        if mode == "training":
            self.set_log_dir()
            #         print(self.log_dir)

            self.save_config(config)

        self.keras_model = self.build(mode=mode, config=config)

        #         if config.init_with=="fixed":
        if config.pretrained_model_type == "class":
            config2 = copy.deepcopy(config)
            #         config2 = config.copy()

            config2.assign_label_mode = "classification"
            config2.img_aug = False
            config2.BC_MATCH = False
            config2.rpn_clustering = False
            config2.init_with = "scratch"

            self.keras_model2 = self.build(mode=mode, config=config2)

        else:

            config2 = copy.deepcopy(config)
            #         config2 = config.copy()

            config2.assign_label_mode = "clustering"
            config2.img_aug = True
            config2.BC_MATCH = False
            config2.rpn_clustering = True
            config2.init_with = "scratch"

            self.keras_model2 = self.build(mode=mode, config=config2)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
        input_shape: The shape of the input image.
        mode: Either "training" or "inference". The inputs and
            outputs of the model differ accordingly.
        """
        assert mode in ["training", "inference", "inference2"]

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )

        # Inputs

        #         if config.rpn_clustering:
        if config.img_aug:
            input_image = keras.Input(
                shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image_t"
            )
            #             input_image_aug = keras_cv.layers.RandomColorDegeneration(factor=0.1, name='input_image_s')(input_image)
            input_image_aug = RandomChannelTint(factor=0.5, name="input_image_s")(
                input_image
            )

            #             input_image_aug = keras.layers.GaussianNoise(stddev=1,name="input_aug")(input_image)
            #             input_image_aug = keras.layers.DepthwiseConv2D(kernel_size=3, use_bias=False,\
            #                                                   trainable=False,padding='same')(input_image)
            #             input_image_aug = keras.layers.RandomContrast(factor=0.5,seed=42, name="input_aug")(input_image)
            #             input_image_aug = keras.layers.RandomContrast(factor=0.5, seed=None,name="input_aug")(input_image)
            #             input_image_aug = tf.keras.layers.RandomFlip(mode="horizontal",name="input_aug")(input_image)
            #             input_image_aug = keras.Input(
            #                 shape=[None, None, config.IMAGE_SHAPE[2]], name="input_aug")

            #             input_image_aug = keras_cv.layers.ChannelShuffle(groups=4, name='input_aug')(input_image)

            #         input_image_aug = KL.Lambda(lambda x: keras_cv.layers.ChannelShuffle(groups=4),
            #                        name="input_aug")(input_image0)

            #         input_image_aug = KL.Lambda(lambda x: tf.image.random_brightness(x,0.2),
            #                        name="input_aug")(input_image0)
            #         input_image_aug=tf.image.random_brightness(input_image0, 0.2, name='input_aug')
            #             input_image = KL.Concatenate(axis=0, name='input_aug_cat')([input_image0,input_image_aug])
            #             input_image = KL.Concatenate(axis=0, name='input_aug_cat')([input_image0,input_image_aug])
            # print("YO", input_image.shape, input_image_aug.shape)
        else:
            input_image = keras.Input(
                shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image"
            )

        input_image_meta = keras.Input(
            shape=[config.IMAGE_META_SIZE], name="input_image_meta"
        )

        if mode == "training":
            # RPN GT
            input_rpn_match = keras.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32
            )
            input_rpn_match_class = keras.Input(
                shape=[None, 1], name="input_rpn_match_class", dtype=tf.int32
            )
            input_rpn_bbox = keras.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32
            )

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = keras.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32
            )
            #             [None,config.MAX_GT_INSTANCES]
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            #             [None,config.MAX_GT_INSTANCES, 4]
            input_gt_boxes = keras.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32
            )
            # Normalize coordinates
            #             print(tf.shape(input_image)[1:3])
            gt_boxes = KL.Lambda(
                lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3])
            )(input_gt_boxes)


        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, stage5=True, train_bn=config.TRAIN_BN
            )
        else:
            if config.BACKBONE in ["resnet50", "resnet101"]:
                if config.img_aug:
                    _, C2, C3, C4, C5 = resnet_graph(
                        input_image_aug,
                        config.BACKBONE,
                        stage5=config.stage5_enabled,
                        train_bn=config.TRAIN_BN,
                    )
                _, C2_t, C3_t, C4_t, C5_t = resnet_graph(
                    input_image,
                    config.BACKBONE,
                    stage5=config.stage5_enabled,
                    train_bn=config.TRAIN_BN,
                    name_prefix="teach_",
                )
            else:  # config.BACKBONE= "resnet18"
                if config.img_aug:
                    _, C2, C3, C4, C5 = resnet_graph_light(
                        input_image_aug,
                        config.BACKBONE,
                        stage5=config.stage5_enabled,
                        train_bn=config.TRAIN_BN,
                    )
                _, C2_t, C3_t, C4_t, C5_t = resnet_graph_light(
                    input_image,
                    config.BACKBONE,
                    stage5=config.stage5_enabled,
                    train_bn=config.TRAIN_BN,
                    name_prefix="teach_",
                )
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config

        if config.stage5_enabled:
            if config.img_aug:
                P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c5p5")(
                    C5
                )
                P4 = KL.Add(name="fpn_p4add")(
                    [
                        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                        KL.Conv2D(
                            config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4"
                        )(C4),
                    ]
                )

            P5_t = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="teach_fpn_c5p5"
            )(C5_t)
            P4_t = KL.Add(name="teach_fpn_p4add")(
                [
                    KL.UpSampling2D(size=(2, 2), name="teach_fpn_p5upsampled")(P5_t),
                    KL.Conv2D(
                        config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="teach_fpn_c4p4"
                    )(C4_t),
                ]
            )

        else:
            if config.img_aug:
                P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4")(
                    C4
                )
            P4_t = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="teach_fpn_c4p4"
            )(C4_t)

        if config.img_aug:
            P3 = KL.Add(name="fpn_p3add")(
                [
                    KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(
                        C3
                    ),
                ]
            )
            P2 = KL.Add(name="fpn_p2add")(
                [
                    KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                    KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(
                        C2
                    ),
                ]
            )
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2"
            )(P2)
            P3 = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3"
            )(P3)
            P4 = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4"
            )(P4)

        P3_t = KL.Add(name="teach_fpn_p3add")(
            [
                KL.UpSampling2D(size=(2, 2), name="teach_fpn_p4upsampled")(P4_t),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="teach_fpn_c3p3")(
                    C3_t
                ),
            ]
        )
        P2_t = KL.Add(name="teach_fpn_p2add")(
            [
                KL.UpSampling2D(size=(2, 2), name="teach_fpn_p3upsampled")(P3_t),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="teach_fpn_c2p2")(
                    C2_t
                ),
            ]
        )
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2_t = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="teach_fpn_p2"
        )(P2_t)
        P3_t = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="teach_fpn_p3"
        )(P3_t)
        P4_t = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="teach_fpn_p4"
        )(P4_t)

        if config.stage5_enabled:
            if config.img_aug:
                P5 = KL.Conv2D(
                    config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5"
                )(P5)
                P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

                rpn_feature_maps = [P2, P3, P4, P5, P6]
                mrcnn_feature_maps = [P2, P3, P4, P5]

            P5_t = KL.Conv2D(
                config.TOP_DOWN_PYRAMID_SIZE,
                (3, 3),
                padding="SAME",
                name="teach_fpn_p5",
            )(P5_t)
            P6_t = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="teach_fpn_p6")(
                P5_t
            )

            rpn_feature_maps_t = [P2_t, P3_t, P4_t, P5_t, P6_t]
            mrcnn_feature_maps_t = [P2_t, P3_t, P4_t, P5_t]

        else:
            if config.img_aug:
                P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P4)

                rpn_feature_maps = [P2, P3, P4, P6]
                mrcnn_feature_maps = [P2, P3, P4]

            P6_t = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="teach_fpn_p6")(
                P4_t
            )

            rpn_feature_maps_t = [P2_t, P3_t, P4_t, P6_t]
            mrcnn_feature_maps_t = [P2_t, P3_t, P4_t]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

            anchors = KL.Lambda(
                lambda x: tf.convert_to_tensor(anchors), name="anchors", trainable=False
            )(input_image)

        #             if config.img_aug:
        #                 anchors_aug = KL.Lambda(lambda x: tf.convert_to_tensor(anchors), name="anchors",\
        #                                         trainable=False)(input_image_aug)
        else:
            anchors = input_anchors

        #         # RPN Model
        #         input_feature_map = tf.keras.Input(shape=[None, None, config.TOP_DOWN_PYRAMID_SIZE], name="input_rpn_feature_map")

        if config.img_aug:

            rpn = build_rpn_model(
                config.RPN_ANCHOR_STRIDE,
                len(config.RPN_ANCHOR_RATIOS),
                config.TOP_DOWN_PYRAMID_SIZE,
                name_prefix="",
            )

        rpn_t = build_rpn_model(
            config.RPN_ANCHOR_STRIDE,
            len(config.RPN_ANCHOR_RATIOS),
            config.TOP_DOWN_PYRAMID_SIZE,
            name_prefix="teach_",
        )


        if config.img_aug:
            # Loop through pyramid layers
            layer_outputs = []  # list of lists
            for p in rpn_feature_maps:
                layer_outputs.append(rpn([p]))
            # Concatenate layer outputs
            # Convert from list of lists of level outputs to list of lists
            # of outputs across levels.
            # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
            outputs = list(zip(*layer_outputs))
            outputs = [
                KL.Concatenate(axis=1, name=n)(list(o))
                for o, n in zip(outputs, output_names)
            ]

            rpn_class_logits, rpn_class, rpn_bbox = outputs

        layer_outputs_t = []  # list of lists
        for p in rpn_feature_maps_t:
            layer_outputs_t.append(rpn_t([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits_t", "rpn_class_t", "rpn_bbox_t"]
        outputs_t = list(zip(*layer_outputs_t))
        outputs_t = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs_t, output_names)
        ]

        rpn_class_logits_t, rpn_class_t, rpn_bbox_t = outputs_t

        if config.rpn_clustering:

            #             print(input_gt_class_ids)
            def lambda_func_rpn(
                cluster_inp,
                RPN_TRAIN_ANCHORS_PER_IMAGE,
                MAX_GT_INSTANCES,
                img_aug,
                layer_name,
            ):
                clustering_labels_rpn_ls = tf.numpy_function(
                    func=rpn_label_reassignment2,
                    inp=[
                        cluster_inp[0],
                        cluster_inp[1],
                        cluster_inp[2],
                        cluster_inp[3],
                        cluster_inp[4],
                        cluster_inp[5],
                        cluster_inp[6],
                        cluster_inp[7],
                        cluster_inp[8],
                        RPN_TRAIN_ANCHORS_PER_IMAGE,
                        MAX_GT_INSTANCES,
                        img_aug,
                    ],
                    Tout=[tf.int32, tf.float32, tf.int32, tf.float32],
                    name=layer_name,
                )

                clustering_labels_rpn_ls[0].set_shape([None, None, 1])
                clustering_labels_rpn_ls[1].set_shape([None, None, 4])
                #                 clustering_labels_rpn_ls[2].set_shape([None,config.MAX_GT_INSTANCES])
                #                 clustering_labels_rpn_ls[3].set_shape([None,config.MAX_GT_INSTANCES,4])

                clustering_labels_rpn_ls[2].set_shape([None, None])
                clustering_labels_rpn_ls[3].set_shape([None, None, 4])
                return clustering_labels_rpn_ls

            clustering_labels_rpn_ls = KL.Lambda(
                lambda x: lambda_func_rpn(
                    x,
                    config.RPN_TRAIN_ANCHORS_PER_IMAGE,
                    config.MAX_GT_INSTANCES,
                    config.img_aug,
                    layer_name="rpnclusters",
                ),
                name="lambda_layer_rpn",
            )(
                [
                    input_image,
                    rpn_class_logits_t,
                    rpn_class_t,
                    rpn_bbox_t,
                    input_rpn_match,
                    input_rpn_match_class,
                    anchors,
                    gt_boxes,
                    input_image_meta,
                ]
            )

            rpn_psudo_labels = clustering_labels_rpn_ls[0]
            input_rpn_bbox_match = clustering_labels_rpn_ls[1]

            #             input_gt_class_ids_byRPN=tf.cast(clustering_labels_rpn_ls[2],tf.int32)
            input_gt_class_ids_byRPN = clustering_labels_rpn_ls[2]
            #             gt_boxes_byRPN=tf.cast(clustering_labels_rpn_ls[3],tf.float32)
            gt_boxes_byRPN = clustering_labels_rpn_ls[3]
        #             rpn_psudo_labels.set_shape([rpn_class.shape[0],rpn_class.shape[1]])

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            config.POST_NMS_ROIS_TRAINING
            if mode == "training"
            else config.POST_NMS_ROIS_INFERENCE
        )



        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config,
        )([rpn_class_t, rpn_bbox_t, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = keras.Input(
                    shape=[config.POST_NMS_ROIS_TRAINING, 4],
                    name="input_roi",
                    dtype=np.int32,
                )

                # Normalize coordinates
                target_rois = KL.Lambda(
                    lambda x: norm_boxes_graph(x, tf.shape(input_image)[1:3])
                )(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            if config.rpn_clustering:

                rois, target_class_ids, target_bbox = DetectionTargetLayer(
                    config, name="proposal_targets"
                )([target_rois, input_gt_class_ids_byRPN, gt_boxes_byRPN])


            else:

                rois, target_class_ids, target_bbox = DetectionTargetLayer(
                    config, name="proposal_targets"
                )([target_rois, input_gt_class_ids, gt_boxes])


            def lambda_layer_ap(target_classes, layer_name):
                ap = tf.numpy_function(
                    func=utils.compute_ap_bbox_metric,
                    inp=[
                        target_classes[0],
                        target_classes[1],
                        target_classes[2],
                        target_classes[3],
                        target_classes[4],
                    ],
                    Tout=[tf.float32],
                    name=layer_name,
                )
                return ap

            def lambda_nmi(
                target_classes, save_seqs, log_dir, barcodeFolderName, layer_name
            ):
                NMI = tf.numpy_function(
                    func=utils.NMI_clus_class,
                    inp=[
                        target_classes[0],
                        target_classes[1],
                        target_classes[2],
                        target_classes[3],
                        target_classes[4],
                        save_seqs,
                        log_dir,
                        barcodeFolderName,
                    ],
                    Tout=[tf.float32],
                    name=layer_name,
                )
                return NMI

            if config.assign_label_mode == "classification":
                #                 mrcnn_class_logits_t, mrcnn_class_t, mrcnn_bbox_t =\
                #                     fpn_classifier_graph(rois, mrcnn_feature_maps_t,input_image_meta,config)

                (
                    target_class_ids2,
                    target_class_ids2_prematched,
                    mrcnn_class_logits_t,
                    mrcnn_class_t,
                    mrcnn_bbox_t,
                ) = fpn_clustering_graph_t(
                    rois,
                    mrcnn_feature_maps_t,
                    input_image_meta,
                    target_class_ids,
                    config,
                )


                NMI = KL.Lambda(
                    lambda t: lambda_nmi(
                        t,
                        save_seqs=0,
                        log_dir=self.log_dir,
                        barcodeFolderName="seqs",
                        layer_name="nmi",
                    ),
                    output_shape=[1],
                    name="lambda_nmi",
                )(
                    [
                        target_class_ids,
                        target_class_ids,
                        rois,
                        mrcnn_class_t,
                        input_image_meta,
                    ]
                )

                AP_50 = KL.Lambda(
                    lambda t: lambda_layer_ap(t, layer_name="ap50"),
                    output_shape=[1],
                    name="lambda_ap",
                )(
                    [
                        input_gt_boxes,
                        input_gt_class_ids,
                        rois,
                        target_class_ids,
                        mrcnn_class_t,
                    ]
                )

            elif config.assign_label_mode == "clustering":

                bc_match = config.BC_MATCH

                (
                    target_class_ids2,
                    target_class_ids2_prematched,
                    mrcnn_class_logits_t,
                    mrcnn_class_t,
                    mrcnn_bbox_t,
                ) = fpn_clustering_graph_t(
                    rois,
                    mrcnn_feature_maps_t,
                    input_image_meta,
                    target_class_ids,
                    config,
                )

                #                 pdb.set_trace()
                if config.img_aug:
                    (
                        mrcnn_class_logits,
                        mrcnn_class,
                        mrcnn_bbox,
                    ) = fpn_clustering_graph_s(
                        rois,
                        mrcnn_feature_maps,
                        input_image_meta,
                        target_class_ids,
                        config,
                    )

                NMI = KL.Lambda(
                    lambda t: lambda_nmi(
                        t,
                        save_seqs=config.save_seqs,
                        log_dir=self.log_dir,
                        barcodeFolderName="seqs",
                        layer_name="nmi",
                    ),
                    output_shape=[1],
                    name="lambda_nmi",
                )(
                    [
                        target_class_ids,
                        target_class_ids2,
                        rois,
                        mrcnn_class_t,
                        input_image_meta,
                    ]
                )

                NMI2 = KL.Lambda(
                    lambda t: lambda_nmi(
                        t,
                        save_seqs=config.save_seqs2,
                        log_dir=self.log_dir,
                        barcodeFolderName="seqs2",
                        layer_name="nmi2",
                    ),
                    output_shape=[1],
                    name="lambda_nmi_2",
                )(
                    [
                        target_class_ids,
                        target_class_ids2_prematched,
                        rois,
                        mrcnn_class_t,
                        input_image_meta,
                    ]
                )
                #

                AP_50 = KL.Lambda(
                    lambda t: lambda_layer_ap(t, layer_name="ap50"),
                    output_shape=[1],
                    name="lambda_ap",
                )(
                    [
                        input_gt_boxes,
                        input_gt_class_ids,
                        rois,
                        target_class_ids2,
                        mrcnn_class_t,
                    ]
                )

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            #             if config.rpn_clustering:
            if config.rpn_clustering:
                if config.img_aug:

                    rpn_class_loss = KL.Lambda(
                        lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
                    )([rpn_psudo_labels, rpn_class_logits])

                    rpn_bbox_loss = KL.Lambda(
                        lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
                    )([input_rpn_bbox_match, rpn_psudo_labels, rpn_bbox])
                else:
                    rpn_class_loss = KL.Lambda(
                        lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
                    )([rpn_psudo_labels, rpn_class_logits_t])

                    rpn_bbox_loss = KL.Lambda(
                        lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
                    )([input_rpn_bbox_match, rpn_psudo_labels, rpn_bbox_t])

            else:
                if config.img_aug:
                    rpn_class_loss = KL.Lambda(
                        lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
                    )([input_rpn_match, rpn_class_logits])

                    rpn_bbox_loss = KL.Lambda(
                        lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
                    )([input_rpn_bbox, input_rpn_match, rpn_bbox])

                else:
                    rpn_class_loss = KL.Lambda(
                        lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
                    )([input_rpn_match, rpn_class_logits_t])

                    rpn_bbox_loss = KL.Lambda(
                        lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
                    )([input_rpn_bbox, input_rpn_match, rpn_bbox_t])

            if config.assign_label_mode == "clustering":
                if config.img_aug:
                    class_loss = KL.Lambda(
                        lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss"
                    )([target_class_ids2, mrcnn_class_logits, active_class_ids])

                    bbox_loss = KL.Lambda(
                        lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss"
                    )([target_bbox, target_class_ids2, mrcnn_bbox])
                else:
                    class_loss = KL.Lambda(
                        lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss"
                    )([target_class_ids2, mrcnn_class_logits_t, active_class_ids])

                    bbox_loss = KL.Lambda(
                        lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss"
                    )([target_bbox, target_class_ids2, mrcnn_bbox_t])

            else:
                class_loss = KL.Lambda(
                    lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss"
                )([target_class_ids, mrcnn_class_logits_t, active_class_ids])
                bbox_loss = KL.Lambda(
                    lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss"
                )([target_bbox, target_class_ids, mrcnn_bbox_t])

            inputs = [
                input_image,
                input_image_meta,
                input_rpn_match,
                input_rpn_bbox,
                input_gt_class_ids,
                input_gt_boxes,
                input_rpn_match_class,
            ]

            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)

            #             if config.img_aug:
            if config.assign_label_mode == "clustering":
                outputs_to_add = [NMI, NMI2]
            else:
                outputs_to_add = [NMI]

            if config.img_aug:
                outputs = [
                    rpn_class_logits,
                    rpn_class,
                    rpn_bbox,
                    mrcnn_class_logits,
                    mrcnn_class,
                    mrcnn_bbox,
                    rpn_rois,
                    output_rois,
                    rpn_class_loss,
                    rpn_bbox_loss,
                    class_loss,
                    bbox_loss,
                    AP_50,
                ] + outputs_to_add  # ,NMI,NMI2]#,NMI2]
            else:
                outputs = [
                    rpn_class_logits_t,
                    rpn_class_t,
                    rpn_bbox_t,
                    mrcnn_class_logits_t,
                    mrcnn_class_t,
                    mrcnn_bbox_t,
                    rpn_rois,
                    output_rois,
                    rpn_class_loss,
                    rpn_bbox_loss,
                    class_loss,
                    bbox_loss,
                    AP_50,
                ] + outputs_to_add  # ,NMI,NMI2]#,NMI2]

            #             model_teacher = KM.Model(inputs_t, outputs_t, name='mask_rcnn_teacher')

            model_student = KM.Model(inputs, outputs, name="mask_rcnn")

        #             if config.img_aug:
        #                 self.update_teacher_weights(model_student)

        return model_student

    def update_teacher_weights(self, model, alpha=0.99):
        for layer in model.layers:
            #
            if layer.name.startswith("teach_"):

                # Extract the corresponding student layer's name
                student_layer_name = layer.name[len("teach_") :]
                print(layer.name, student_layer_name)
                # Find the student layer in the model
                student_layer = model.get_layer(student_layer_name)

                # Update the teacher layer's weights with the EMA of the student layer's weights
                new_weights = []
                for teach_weight, student_weight in zip(
                    layer.get_weights(), student_layer.get_weights()
                ):
                    new_weight = alpha * teach_weight + (1 - alpha) * student_weight
                    new_weights.append(new_weight)

                # Set the updated weights to the teacher layer
                layer.set_weights(new_weights)

    def update_ema_variables(model_teacher, model_student, alpha):
        for layer_t, layer_s in zip(model_teacher.layers, model_student.layers):
            weights_t = layer_t.get_weights()
            weights_s = layer_s.get_weights()

            if len(weights_t) > 0:
                new_weights = [
                    (1 - alpha) * w_t + alpha * w_s
                    for w_t, w_s in zip(weights_t, weights_s)
                ]
                layer_t.set_weights(new_weights)

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno

            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir),
            )
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno

            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name)
            )
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        self.keras_model.load_weights(filepath)



    def compile(self, learning_rate, momentum, clear_loss=True):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        #         super(GAN, self).compile()

        #         self.keras_model.metrics_tensors = []
        # Optimizer object

        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM,
        )

        #         sgd = keras.optimizers.SGD(
        #             learning_rate=learning_rate, momentum=momentum)

        #         optimizer=keras.optimizers.Adam(learning_rate=learning_rate/100,clipnorm=self.config.GRADIENT_CLIP_NORM)

        #         opt.assign_average_vars(model.variables)
        #         optimizer = tfa.optimizers.MovingAverage(sgd,clipnorm=self.config.GRADIENT_CLIP_NORM)
        #         optimizer = tfa.optimizers.SWA(sgd,clipnorm=self.config.GRADIENT_CLIP_NORM)
        #         moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
        #         optimizer = tfa.optimizers.SWA(sgd)
        self.optimizer = optimizer
        # Add Losses
        # First, clear previously set losses to avoid duplication
        #         self.keras_model._losses = []
        #         self.keras_model._per_input_losses = {}

        if clear_loss:
            assert self.keras_model.losses == []

        # Train all
        if self.config.USE_RPN_ROIS:
            loss_names = [
                "rpn_class_loss",
                "rpn_bbox_loss",
                "mrcnn_class_loss",
                "mrcnn_bbox_loss",
            ]
        # Train all but RPN
        elif not self.config.USE_RPN_ROIS:
            loss_names = ["mrcnn_class_loss", "mrcnn_bbox_loss"]

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            #             if layer.output in self.keras_model.losses:
            #                 continue
            self.keras_model.add_loss(
                lambda x=layer: tf.reduce_mean(input_tensor=x.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.0)
            )
        #             loss = (
        #                 tf.reduce_mean(input_tensor=layer.output, keepdims=True)
        #                 * self.config.LOSS_WEIGHTS.get(name, 1.))
        #             self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
            / tf.cast(tf.size(input=w), tf.float32)
            for w in self.keras_model.trainable_weights
            if "gamma" not in w.name and "beta" not in w.name
        ]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        #         self.keras_model.compile(optimizer=optimizer)
        self.keras_model.compile(
            optimizer=optimizer, loss=[None] * len(self.keras_model.outputs)
        )

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = tf.reduce_mean(
                input_tensor=layer.output, keepdims=True
            ) * self.config.LOSS_WEIGHTS.get(name, 1.0)

            self.keras_model.add_loss(
                lambda x=layer: tf.reduce_mean(input_tensor=x.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.0)
            )

            #             self.keras_model.metrics_tensors.extend(loss)
            #             self.keras_model.metrics_tensors.append(loss)
            self.keras_model.add_metric(loss, name=name, aggregation="mean")

        if self.config.assign_label_mode == "clustering":
            vars_to_track = ["lambda_ap", "lambda_nmi", "lambda_nmi_2"]
        else:
            vars_to_track = ["lambda_ap", "lambda_nmi"]

        for name in vars_to_track:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            #             self.keras_model.metrics_names.append("nmi2")

            var_val = tf.reduce_mean(input_tensor=layer.output, keepdims=True)
            #             self.keras_model.metrics_tensors.extend(var_val)

            #             self.keras_model.add_metric(lambda x=layer: tf.reduce_mean(input_tensor=x.output, keepdims=True))
            self.keras_model.add_metric(var_val, name=name, aggregation="mean")

    #             self.keras_model.metrics_tensors.append(var_val)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        for layer in layers:
            #             print(layer.__class__.__name__)
            #             pdb.set_trace()
            # Is the layer a model?
            #             if layer.__class__.__name__ == 'Model':
            if (
                layer.__class__.__name__ == "Functional"
                or layer.__class__.__name__ == "Model"
                or layer.__class__.__name__ == "RPNLayer"
            ):
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if self.config.img_aug:
                if "teach_" in layer.name:
                    trainable = False
            print(layer.name, trainable)
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == "TimeDistributed":
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log(
                    "{}{:20}   ({})".format(
                        " " * indent, layer.name, layer.__class__.__name__
                    )
                )

    def save_config(self, config):
        import pickle

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        output = open(self.log_dir + "/config.pkl", "wb")
        pickle.dump(config.to_dict(), output, -1)
        output.close()
        return

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.ckpt"
            m = re.match(regex, str(model_path))
            if m:
                now = datetime.datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                )
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print("Re-starting from epoch %d" % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(
            self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)
        )

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(
            self.log_dir, "mask_rcnn_{}_*epoch*.ckpt".format(self.config.NAME.lower())
        )
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(
        self,
        learning_rate,
        epochs,
        layers,
        augmentation=None,
        custom_callbacks=None,
        no_augmentation_sources=None,
        just_compile=False,
    ):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(.*mrcnn\_.*)|(.*fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        if self.config.assign_label_mode == "classification":
            self.config.LOSS_WEIGHTS = self.config.CLASS_LOSS_WEIGHTS
        else:
            self.config.LOSS_WEIGHTS = self.config.CLUS_LOSS_WEIGHTS

        train_generator = DataGenerator(
            self.config.list_of_sites,
            self.config,
            shuffle=False,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources,
        )
        val_generator = DataGenerator(
            self.config.val_list_of_sites,
            self.config,
            shuffle=False,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources,
        )
        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir + "/seqs/"):
            #             os.makedirs(self.log_dir)
            os.makedirs(self.log_dir + "/seqs/")
            os.makedirs(self.log_dir + "/seqs2/")

        #         # Callbacks
        callbacks = [
            keras.callbacks.CSVLogger(
                filename=self.log_dir + "/csvlog.log", separator=",", append=False
            )
            #             keras.callbacks.LambdaCallback(on_batch_end=lambdaCallbackFunc)
        ]

        if os.name == "nt":
            workers = 0
        else:
            workers = self.config.n_workers_to_use

        if self.config.init_with == "fixed":
            #             # Create a dictionary to map layer names to layers
            layer_dict = {
                layer.name: layer
                for layer in self.keras_model.layers
                if "teach" in layer.name
            }
            ls_pr = list(layer_dict.keys())
            #             print(ls_pr)
            latest = tf.train.latest_checkpoint(self.config.pretrained_model_path)
            print("latest", latest)
            self.keras_model2.load_weights(latest)

            self.transfer_weights_partial(
                self.keras_model2, self.keras_model, to_st_as_well=True
            )

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))

        if self.config.init_with == "fixed":
            matched_strings = []
            print(ls_pr)
            for string in ls_pr:
                if re.match(layers, string):
                    matched_strings.append(string)

            print("matched_strings", matched_strings)
            #             pdb.set_trace()
            ema_update_callback = EMAUpdateCallback(
                matched_strings,
                self.keras_model,
                ema_decay=0.99,
                n_batch=self.config.update_teacher_n_batch,
            )
            callbacks.append(ema_update_callback)

        print("layers", layers)
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, clear_loss=True)

        if not just_compile:
            # Start the training loop with the initialized session

            self.keras_model.fit(
                train_generator,
                initial_epoch=self.epoch,
                epochs=epochs,
                steps_per_epoch=self.config.STEPS_PER_EPOCH,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_steps=self.config.VALIDATION_STEPS,
                max_queue_size=self.config.max_q_size_generator,
                workers=workers,
                use_multiprocessing=True,  # changed by Marzi to False
                shuffle=False,
            )

            self.keras_model.save_weights(self.log_dir + "/final_model/final_model")
            self.epoch = max(self.epoch, epochs)

    def transfer_weights(
        self, source_model, target_model, layer_names, to_st_as_well=True
    ):
        for layer_name in layer_names:
            source_layer = source_model.get_layer(layer_name)
            target_layer_t = target_model.get_layer(layer_name)

            #                 target_layer.set_weights(source_layer.get_weights())
            # Update the teacher layer's weights with the EMA of the student layer's weights
            new_weights = []
            for w in source_layer.get_weights():
                new_weights.append(w)

            # Set the updated weights to the teacher layer
            target_layer_t.set_weights(new_weights)
            if to_st_as_well:
                st_layer_name = layer_name[len("teach_") :]
                if st_layer_name in target_model.layers:
                    target_layer_s = target_model.get_layer(st_layer_name)
                    target_layer_s.set_weights(new_weights)

    def transfer_weights(
        self, source_model, target_model, layer_names, to_st_as_well=True
    ):
        for layer_name in layer_names:
            source_layer = source_model.get_layer(layer_name)
            target_layer_t = target_model.get_layer(layer_name)

            #                 target_layer.set_weights(source_layer.get_weights())
            # Update the teacher layer's weights with the EMA of the student layer's weights
            new_weights = []
            for w in source_layer.get_weights():
                new_weights.append(w)

            # Set the updated weights to the teacher layer
            target_layer_t.set_weights(new_weights)
            if to_st_as_well:
                st_layer_name = layer_name[len("teach_") :]
                if st_layer_name in target_model.layers:
                    target_layer_s = target_model.get_layer(st_layer_name)
                    target_layer_s.set_weights(new_weights)

    def transfer_weights_partial(self, model_source, model_target, to_st_as_well=True):
        """
        Transfers the weights from the source model to the corresponding layers in the target model.

        Args:
            model_source (tf.keras.Model): The source model from which the weights will be transferred.
            model_target (tf.keras.Model): The target model to which the weights will be transferred.
        """

        def transfer_weights_recursive(source_layers, target_layers):
            for layer_name, target_layer in target_layers.items():
                if layer_name in source_layers:
                    source_layer = source_layers[layer_name]

                    if source_layer.__class__.__name__ == "TimeDistributed":
                        source_layer = source_layer.layer
                        target_layer = target_layer.layer

                    if hasattr(source_layer, "layers") and hasattr(
                        target_layer, "layers"
                    ):
                        source_sublayers = {
                            sublayer.name: sublayer for sublayer in source_layer.layers
                        }
                        target_sublayers = {
                            sublayer.name: sublayer for sublayer in target_layer.layers
                        }
                        transfer_weights_recursive(source_sublayers, target_sublayers)
                    elif hasattr(source_layer, "layer") and hasattr(
                        target_layer, "layer"
                    ):
                        transfer_weights_recursive(
                            source_layer.layer, target_layer.layer
                        )
                    else:
                        if 0:
                            #                         if source_layer.get_config() != target_layer.get_config():
                            print(
                                f"Source layer '{layer_name}' and target layer '\
                            {layer_name}' must have the same configuration."
                            )
                            print(source_layer.get_config())
                            print(target_layer.get_config())

                        print(source_layer.name, target_layer.name)

                        target_layer.set_weights(source_layer.get_weights())

        source_layers = {layer.name: layer for layer in model_source.layers}
        target_layers = {layer.name: layer for layer in model_target.layers}

        transfer_weights_recursive(source_layers, target_layers)

        if to_st_as_well:
            source_layers_2 = {
                layer.name[len("teach_") :]: layer
                for layer in model_source.layers
                if "teach_" in layer.name
            }
            target_layers_st = {
                layer.name[len("teach_") :]: model_target.get_layer(
                    layer.name[len("teach_") :]
                )
                for layer in model_source.layers
                if "teach_" in layer.name
            }
            #             pdb.set_trace()
            transfer_weights_recursive(source_layers_2, target_layers_st)

    def evaluate_saved_model(
        self,
        learning_rate,
        layers,
        pretrained_model_path=None,
        augmentation=None,
        no_augmentation_sources=None,
    ):
        """evaluate the model"""

        train_generator = DataGenerator(
            self.config.list_of_sites,
            self.config,
            shuffle=False,
            augmentation=augmentation,
            batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources,
        )

        if not os.path.exists(self.log_dir + "/seqs/"):
            #             os.makedirs(self.log_dir)
            os.makedirs(self.log_dir + "/seqs/")
            os.makedirs(self.log_dir + "/seqs2/")

        callbacks = [
            keras.callbacks.CSVLogger(
                filename=self.log_dir + "/csvlog.log", separator=",", append=False
            )
        ]

        if pretrained_model_path:

            if 0:
                self.transfer_weights_partial(
                    pretrained_model_path.keras_model, self.keras_model
                )

            else:

                if "ckpt" in pretrained_model_path:
                    self.keras_model.load_weights(pretrained_model_path)
                else:

                    #             # Create a dictionary to map layer names to layers
                    layer_dict = {
                        layer.name: layer
                        for layer in self.keras_model.layers
                        if "teach" in layer.name
                    }
                    ls_pr = list(layer_dict.keys())
                    #             print(ls_pr)
                    latest = tf.train.latest_checkpoint(
                        self.config.pretrained_model_path
                    )
                    self.keras_model2.load_weights(latest)
                    print("latest", latest)

                self.transfer_weights_partial(
                    self.keras_model2, self.keras_model, to_st_as_well=False
                )
            #         pdb.set_trace()
            self.compile(learning_rate, self.config.LEARNING_MOMENTUM, clear_loss=False)

        if os.name == "nt":
            workers = 0
        else:
            workers = self.config.n_workers_to_use

        e = self.keras_model.evaluate(
            train_generator,
            verbose="auto",
            max_queue_size=self.config.max_q_size_generator,
            callbacks=callbacks,
            workers=workers,
            use_multiprocessing=True,
        )
        print(self.keras_model.metrics_names)
        #         e = {out: e[i] for i, out in enumerate(self.keras_model.metrics_names)}

        return e

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE,
            )
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0,
                image.shape,
                molded_image.shape,
                window,
                scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32),
                [0, 0],
            )
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        #         print("image_meta",image_meta.shape,image_meta)

        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        #         print(image_metas.shape)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE,
            )
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(
                a, image_shape[:2]
            )
        return self._anchor_cache[tuple(image_shape)]

    def get_anchors_test(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        # print("backbone_shapes", backbone_shapes.shape)
        # Generate Anchors
        a = utils.generate_pyramid_anchors(
            self.config.RPN_ANCHOR_SCALES,
            [0.5, 1, 2],
            backbone_shapes,
            self.config.BACKBONE_STRIDES,
            self.config.RPN_ANCHOR_STRIDE,
        )
        # print("a", a.shape)
        # Keep a copy of the latest anchors in pixel coordinates because
        # it's used in inspect_model notebooks.
        # TODO: Remove this after the notebook are refactored to not use it
        self.anchors = utils.norm_boxes(a, image_shape[:2])

        return self.anchors

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        #         print(layer.__class__.__name__)
        if layer.__class__.__name__ == "TimeDistributed":
            #             print(layer.__class__.__name__,layer.layer.name)
            return self.find_trainable_layer(layer.layer)

        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        #         i=0
        for l in self.keras_model.layers:
            #             print(i)
            #             i+=1
            # If layer is a wrapper, find inner trainable layer
            if l.__class__.__name__ == "Functional":
                for l2 in l.layers:
                    l2 = self.find_trainable_layer(l2)
                    if l2.get_weights():
                        layers.append(l2)
            else:
                l = self.find_trainable_layer(l)
                # Include layer if it has weights
                if l.get_weights():
                    layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """

        if not self.config.USE_RPN_ROIS:
            assert (
                len(images) == 1
            ), "Without RPN only one image can be input since proposals are for one image only."

        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        #         if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #             inputs += [K.learning_phase()]

        #         kf = K.function(model.inputs, list(outputs.values()))
        kf = tf.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # if RPN is disabled, generate rois for stage 2 externally (manually)
        if not self.config.USE_RPN_ROIS:

            # Use GT boxes only
            rpn_rois = generate_gt_rois(
                image_shape,
                self.config.POST_NMS_ROIS_INFERENCE,
                gt_class_ids,
                gt_bboxes,
            )

            # Convert to input format of the classifer head (RoIAlign) -> [batch, num_rois, (y1, x1, y2, x2)]
            # Since the rpn_rois are the GT, it will make no filtering, basically converting the data format.
            #
            (
                input_rois,
                mrcnn_class_ids,
                mrcnn_bbox,
                mrcnn_mask,
            ) = build_detection_targets(
                rpn_rois,
                gt_class_ids,
                gt_bboxes,
                gt_masks,
                self.config,
                self.config.POST_NMS_ROIS_INFERENCE,
            )

            # Create single batch with proposals
            batch_rois = np.zeros(
                (self.config.BATCH_SIZE,) + input_rois.shape, dtype=input_rois.dtype
            )
            batch_rois[self.config.BATCH_SIZE - 1] = input_rois

            # Data is passed in the format as stage 2 the classifer header would expect it
            model_in.append(batch_rois)

        # Run inference
        #         if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #             model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Miscellenous Graph Functions
############################################################


def trim_zeros_graph(boxes, name="trim_zeros"):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    #     non_zeros = tf.cast(tf.reduce_sum(input_tensor=tf.math.abs(boxes), axis=1), tf.bool)
    non_zeros = tf.cast(
        tf.reduce_sum(
            input_tensor=tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=1),
            axis=1,
        ),
        tf.bool,
    )

    boxes = tf.boolean_mask(tensor=boxes, mask=non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, : counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0.0, 0.0, 1.0, 1.0])

    boxes = tf.cast(boxes, tf.float32)  # by marzi
    return tf.math.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0.0, 0.0, 1.0, 1.0])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


class RandomChannelTint(keras_cv.layers.BaseImageAugmentationLayer):
    """RandomChannelTint randomly applies a blue tint to images.

    Args:
      value_range: value_range: a tuple or a list of two elements. The first value
        represents the lower bound for values in passed images, the second represents
        the upper bound. Images passed to the layer should have values within
        `value_range`.
      factor: A tuple of two floats, a single float or a
        `keras_cv.FactorSampler`. `factor` controls the extent to which the
        image is blue shifted. `factor=0.0` makes this layer perform a no-op
        operation, while a value of 1.0 uses the degenerated result entirely.
        Values between 0 and 1 result in linear interpolation between the original
        image and a fully blue image.
        Values should be between `0.0` and `1.0`.  If a tuple is used, a `factor` is
        sampled between the two values for every image augmented.  If a single float
        is used, a value between `0.0` and the passed float is sampled.  In order to
        ensure the value is always the same, please pass a tuple with two identical
        floats: `(0.5, 0.5)`.
    """

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        #         self.value_range = value_range
        self.factor = keras_cv.utils.parse_factor(factor)

    def get_random_transformation(self, **kwargs):
        # kwargs holds {"images": image, "labels": label, etc...}
        return self.factor() * 60

    def augment_image(self, image, transformation=None, **kwargs):
        value_range = (tf.reduce_min(image), tf.reduce_max(image))
        image = keras_cv.utils.transform_value_range(image, value_range, (0, 255))
        [a, t, c, g] = tf.unstack(image, axis=-1)
        rand_channel_ind = np.random.randint(4)
        new_ls = [a, t, c, g]
        new_ls[rand_channel_ind] = tf.clip_by_value(
            new_ls[rand_channel_ind] + transformation, 0.0, 255.0
        )
        result = tf.stack(new_ls, axis=-1)
        result = keras_cv.utils.transform_value_range(result, (0, 255), value_range)
        return result

    def augment_label(self, label, transformation=None, **kwargs):
        # you can use transformation somehow if you want

        if transformation > 100:
            # i.e. maybe class 2 corresponds to blue images
            return 2.0

        return label

    def augment_bounding_boxes(self, bounding_boxes, transformation=None, **kwargs):
        # you can also perform no-op augmentations on label types to support them in
        # your pipeline.
        return bounding_boxes
