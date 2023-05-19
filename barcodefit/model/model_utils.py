"""
[The base of this script is Matterport implementation of Mask R-CNN model Written by Waleed Abdulla (Copyright (c) 2017 Matterport, Inc.).]
"""

import logging
import math
import os
import random
import shutil
import sys
import urllib.request
import warnings
from distutils.version import LooseVersion

import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform
import sklearn
import tensorflow as tf
from scipy.spatial import distance

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = (
    "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
)


############################################################
#  Bounding Boxes
############################################################


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > 0.5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > 0.5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        #         print()
        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self):
        return self._image_ids  # commented by Marzi

    #          return  self.image_ids    #added by Marzi

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        # Load image
        image = skimage.io.imread(self.image_info[image_id]["path"])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning(
            "You are using the default load_mask(), maybe you need to define your own one."
        )
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(
            image, (round(h * scale), round(w * scale)), order=0, preserve_range=True
        )
    # (image, output_shape, order=1, mode='constant', cval=0, clip=True,
    #            preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None)
    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y : y + min_dim, x : x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y : y + h, x : x + w]
    else:
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape, order=0)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128] #RPN_ANCHOR_SCALES = (2,4)
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2] #RPN_ANCHOR_RATIOS = [1]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels. #BACKBONE_STRIDES = [4,8]
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel. #RPN_ANCHOR_STRIDE = 1
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    #     print("box_widths",box_widths)
    #     print("box_heights",box_heights)
    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1
    )
    return boxes


def generate_pyramid_anchors(
    scales, ratios, feature_shapes, feature_strides, anchor_stride
):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    feature_shapes= [[64 64]
                     [32 32]
                     [16 16]
                     [ 8  8]
                     [ 4  4]]


    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    #     print("len(scales)",len(scales),scales) # 2, (2,4)
    for i in range(len(scales)):
        #         print("feature_shapes[i]",feature_shapes[i])
        #         print(scales[i],feature_shapes[i],feature_strides[i])
        #                 8 [64 64] 4
        #                 16 [32 32] 8
        #                 32 [16 16] 16
        #                 64 [8 8] 32
        #                 128 [4 4] 64
        anc = generate_anchors(
            scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride
        )
        #         print(anc.shape,anc[0:2,:],anc[-1,:])

        anchors.append(anc)

    # 8 [64 64] 4
    # [[-4. -4.  4.  4.]
    #  [-4.  0.  4.  8.]] [248. 248. 256. 256.]
    # 16 [32 32] 8
    # [[-8. -8.  8.  8.]
    #  [-8.  0.  8. 16.]] [240. 240. 256. 256.]
    # 32 [16 16] 16
    # [[-16. -16.  16.  16.]
    #  [-16.   0.  16.  32.]] [224. 224. 256. 256.]
    # 64 [8 8] 32
    # [[-32. -32.  32.  32.]
    #  [-32.   0.  32.  64.]] [192. 192. 256. 256.]
    # 128 [4 4] 64
    # [[-64. -64.  64.  64.]
    #  [-64.   0.  64. 128.]] [128. 128. 256. 256.]
    #     print('anchors.shape',np.concatenate(anchors, axis=0).shape) #anchors.shape (5120, 4)
    #     print(np.concatenate(anchors, axis=0))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., : gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[: pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    #     overlaps = compute_overlaps(pred_boxes, gt_boxes)
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[: low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_matches_bbox(
    gt_boxes,
    gt_class_ids,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    #     gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[: pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    #     pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    print("x", pred_boxes.shape, gt_boxes.shape)
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    #     overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[: low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(
    gt_boxes,
    gt_class_ids,
    gt_masks,
    pred_boxes,
    pred_class_ids,
    pred_scores,
    pred_masks,
    iou_threshold=0.5,
):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        pred_boxes,
        pred_class_ids,
        pred_scores,
        pred_masks,
        iou_threshold,
    )

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_bbox(
    gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold=0.5
):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches_bbox(
        gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold
    )

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return mAP


#     return mAP, precisions, recalls, overlaps


def compute_ap_range(
    gt_box,
    gt_class_id,
    gt_mask,
    pred_box,
    pred_class_id,
    pred_score,
    pred_mask,
    iou_thresholds=None,
    verbose=1,
):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = compute_ap(
            gt_box,
            gt_class_id,
            gt_mask,
            pred_box,
            pred_class_id,
            pred_score,
            pred_mask,
            iou_threshold=iou_threshold,
        )
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print(
            "AP @{:.2f}-{:.2f}:\t {:.3f}".format(
                iou_thresholds[0], iou_thresholds[-1], AP
            )
        )
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(
        coco_model_path, "wb"
    ) as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(
    image,
    output_shape,
    order=1,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=False,
    anti_aliasing_sigma=None,
):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
        )
    else:
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
        )


def calculate_cluster_centroids(feat, labels, nClust):
    # feat (n_samples,n_features)
    # labels (n_samples,)
    #     nClust=np.max(labels)+1
    #     cents=np.empty((nClust,feat.shape[1]))
    cents = np.zeros((nClust, feat.shape[1]))
    #     cents[:] = np.nan
    for c in range(nClust):
        class_samples = feat[labels == c, :]
        if class_samples.shape[0] != 0:
            cents[c, :] = np.mean(class_samples, axis=0)
    return cents


def reas_labels4(clustering_labels, pred_labels_forg, n_cl):

    """
    clustering_labels: kmeans output labels 0,1,2,3
    pred_labels_forg: current model prediction labels 0,1,2,3
    n_cl: 4
    """
    y_pred, _ = get_y_preds(clustering_labels, pred_labels_forg, n_cl)
    #     print("clustering_labels",clustering_labels)
    #     print("pred_labels_forg",pred_labels_forg)
    #     print("y_pred",y_pred+1)

    #     unique, counts = np.unique(clustering_labels, return_counts=True)
    #     result = np.column_stack((unique, counts))
    #     print('clustering_labels',result)

    #     unique, counts = np.unique(pred_labels_forg, return_counts=True)
    #     result = np.column_stack((unique, counts))
    #     print('pred_labels_forg',result)

    #     unique, counts = np.unique(y_pred, return_counts=True)
    #     result = np.column_stack((unique, counts))
    #     print('y_pred',result)

    return y_pred.astype(int) + 1


from munkres import Munkres


def get_y_preds(cluster_assignments, y_true, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    #     print("conf",cluster_assignments,y_true)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


# def reas_labels3(clustering_labels,pred_labels_arr,kmeans_centers,X,n_cl):
def reas_labels3(clustering_labels, pred_labels_arr_cents, kmeans_centers, X, n_cl):
    from sklearn.neighbors import DistanceMetric

    dist = DistanceMetric.get_metric("euclidean")
    #     # a[nan_index_a,:]=np.nan
    #     b[nan_index_b,:]=np.nan
    #     clustering_labels_cents=calculate_cluster_centroids(X,clustering_labels)
    #     clustering_labels_cents=calculate_cluster_centroids(X,clustering_labels)
    #     pred_labels_arr_cents=calculate_cluster_centroids(X,pred_labels_arr,n_cl+1)
    distMat = dist.pairwise(kmeans_centers, pred_labels_arr_cents[1:, :])
    distMat = distMat / np.nansum(distMat, axis=0)
    map_dict = {}
    ordered_ass = np.argsort(distMat, axis=1)
    min_dist_clusters = np.nanargmin(distMat, axis=1)
    min_dist = np.nanmin(distMat, axis=1)
    confid_ordered_labels = np.argsort(min_dist)
    # non_nan_indexes=np.delete(range(5), 1)
    non_nan_indexes = list(range(n_cl))
    for ci in range(n_cl):
        #     min_dist_clusters[ci]
        ind_cluster_to_ass = confid_ordered_labels[ci]
        if min_dist_clusters[ind_cluster_to_ass] in non_nan_indexes:
            map_dict[ind_cluster_to_ass] = min_dist_clusters[ind_cluster_to_ass]
            non_nan_indexes.remove(min_dist_clusters[ind_cluster_to_ass])
        else:
            i = 1
            while ordered_ass[ind_cluster_to_ass, i] not in non_nan_indexes:
                i += 1
            #         for i in range(1,5):
            to_cl = ordered_ass[ind_cluster_to_ass, i]
            map_dict[ind_cluster_to_ass] = to_cl
            non_nan_indexes.remove(to_cl)

    #     print(map_dict,clustering_labels)
    reassigned_labels = np.vectorize(map_dict.get)(clustering_labels)
    #     print(reassigned_labels)
    return reassigned_labels + 1


def reas_labels2(clustering_labels, pred_labels_arr):
    from sklearn.neighbors import DistanceMetric

    dist = DistanceMetric.get_metric("hamming")
    #     print(utils.NMI_clus_class(clustering_labels,pred_labels_arr))
    #     print("shape",clustering_labels,pred_labels_arr)
    map_dict = {}
    clus_uniq_labels = np.unique(clustering_labels)
    clus_uniq_labels_list = list(clus_uniq_labels)
    n_clus = len(clus_uniq_labels)
    haming_based_clus = []
    for ci in range(n_clus):
        clustering_labels_binary = 100 * np.ones(clustering_labels.shape)
        c = clus_uniq_labels[ci]
        clustering_labels_binary[clustering_labels == c] = 1
        haming_bin_clus = []
        for ci2 in range(len(clus_uniq_labels_list)):
            pred_labels_arr_binary = 100 * np.ones(clustering_labels.shape)
            c2 = clus_uniq_labels_list[ci2]
            pred_labels_arr_binary[pred_labels_arr == c2] = 1
            haming_bin_clus.append(
                dist.pairwise([pred_labels_arr_binary, clustering_labels_binary])[0, 1]
            )

        best_cluster_based_on_haming = clus_uniq_labels_list[np.argmin(haming_bin_clus)]
        clus_uniq_labels_list.remove(best_cluster_based_on_haming)

        #         print("haming_bin_clus",haming_bin_clus)
        haming_based_clus.append(best_cluster_based_on_haming)
        map_dict[c] = best_cluster_based_on_haming

    #     print("map_dict",map_dict)
    #     print("clustering_labels",clustering_labels)
    reassigned_labels = np.vectorize(map_dict.get)(clustering_labels)
    return reassigned_labels


def reas_labels(clustering_labels, pred_labels_arr, bkg_c_i):
    from sklearn.neighbors import DistanceMetric

    dist = DistanceMetric.get_metric("hamming")
    #     print("shape",clustering_labels,pred_labels_arr)
    map_dict = {}
    map_dict[bkg_c_i] = 0
    clus_uniq_labels = np.unique(clustering_labels)
    clus_uniq_labels_list = list(clus_uniq_labels)
    clus_uniq_labels_list.remove(0)
    #     print(clus_uniq_labels_list)
    #     print(clus_uniq_labels,bkg_c_i)
    clus_uniq_labels = clus_uniq_labels[clus_uniq_labels != bkg_c_i]
    #     print(clus_uniq_labels,bkg_c_i)
    n_clus = len(clus_uniq_labels)
    haming_based_clus = []
    for ci in range(n_clus):
        clustering_labels_binary = 100 * np.ones(clustering_labels.shape)
        c = clus_uniq_labels[ci]
        clustering_labels_binary[clustering_labels == c] = 1
        haming_bin_clus = []
        for ci2 in range(len(clus_uniq_labels_list)):
            pred_labels_arr_binary = 100 * np.ones(clustering_labels.shape)
            c2 = clus_uniq_labels_list[ci2]
            pred_labels_arr_binary[pred_labels_arr == c2] = 1
            haming_bin_clus.append(
                dist.pairwise([pred_labels_arr_binary, clustering_labels_binary])[0, 1]
            )

        best_cluster_based_on_haming = clus_uniq_labels_list[np.argmin(haming_bin_clus)]
        clus_uniq_labels_list.remove(best_cluster_based_on_haming)

        #         print("haming_bin_clus",haming_bin_clus)
        haming_based_clus.append(best_cluster_based_on_haming)
        map_dict[c] = best_cluster_based_on_haming

    print("map_dict", map_dict)
    reassigned_labels = np.vectorize(map_dict.get)(clustering_labels)
    #     print("haming_based_clus",haming_based_clus)
    #     print("clustering_labels",clustering_labels)

    #     print("reassigned_labels",reassigned_labels)
    return reassigned_labels


def reas_labels_cent(clustering_labels, pred_labels_arr):
    from sklearn.neighbors import DistanceMetric

    dist = DistanceMetric.get_metric("hamming")
    #     print("shape",clustering_labels,pred_labels_arr)
    map_dict = {}
    clus_uniq_labels = np.unique(clustering_labels)
    clus_uniq_labels_list = list(clus_uniq_labels)
    n_clus = len(clus_uniq_labels)
    haming_based_clus = []
    for ci in range(n_clus):
        clustering_labels_binary = 100 * np.ones(clustering_labels.shape)
        c = clus_uniq_labels[ci]
        clustering_labels_binary[clustering_labels == c] = 1
        haming_bin_clus = []
        for ci2 in range(len(clus_uniq_labels_list)):
            pred_labels_arr_binary = 100 * np.ones(clustering_labels.shape)
            c2 = clus_uniq_labels_list[ci2]
            pred_labels_arr_binary[pred_labels_arr == c2] = 1
            haming_bin_clus.append(
                dist.pairwise([pred_labels_arr_binary, clustering_labels_binary])[0, 1]
            )

        best_cluster_based_on_haming = clus_uniq_labels_list[np.argmin(haming_bin_clus)]
        clus_uniq_labels_list.remove(best_cluster_based_on_haming)

        #         print("haming_bin_clus",haming_bin_clus)

        haming_based_clus.append(best_cluster_based_on_haming)
        map_dict[ci] = best_cluster_based_on_haming
    reassigned_labels = np.vectorize(map_dict.get)(clustering_labels)
    #     print("haming_based_clus",haming_based_clus)
    #     print("map_dict",map_dict)
    #     print("reassigned_labels",reassigned_labels)
    return reassigned_labels


def compute_ap_bbox_metric(gt_boxes, gt_class_ids, rois, target_class_ids, mrcnn_class):

    print(
        "compute_ap_bbox_metric:",
        gt_boxes.shape,
        gt_class_ids.shape,
        rois.shape,
        target_class_ids.shape,
        mrcnn_class.shape,
        np.unique(gt_class_ids),
    )

    print(rois.min(), rois.max(), gt_boxes.min(), gt_boxes.max())

    gt_class_min = np.min(gt_class_ids, axis=0)
    forgG_gt_index = np.where(gt_class_min > 0)[0]
    gt_class_ids = gt_class_ids[:, forgG_gt_index]
    gt_boxes = gt_boxes[:, forgG_gt_index, :]

    print(
        "compute_ap_bbox_metric:",
        gt_boxes.shape,
        gt_class_ids.shape,
        rois.shape,
        target_class_ids.shape,
        mrcnn_class.shape,
        np.unique(gt_class_ids),
    )

    target_class_min = np.min(target_class_ids, axis=0)
    forgG_index = np.where(target_class_min > 0)[0]

    if (len(target_class_ids.shape) > 1) and (len(forgG_index) > 1):

        target_class_ids = target_class_ids[:, forgG_index]
        rois = rois[:, forgG_index, :]
        mrcnn_class = mrcnn_class[:, forgG_index, :]

        pred_class_probs = mrcnn_class.reshape(
            mrcnn_class.shape[0] * mrcnn_class.shape[1], mrcnn_class.shape[2]
        )
        pred_scores = np.argmax(pred_class_probs, axis=1)

        pred_class_ids = target_class_ids.reshape(
            target_class_ids.shape[0] * target_class_ids.shape[1],
        )
        pred_boxes_normed = rois.reshape(rois.shape[0] * rois.shape[1], 4)

        pred_boxes = denorm_boxes(pred_boxes_normed, (256, 256))

        gt_boxes_f = gt_boxes.reshape(gt_boxes.shape[0] * gt_boxes.shape[1], 4)
        gt_class_ids_f = gt_class_ids.reshape(
            gt_class_ids.shape[0] * gt_class_ids.shape[1],
        )

        print(gt_boxes_f.min(), gt_boxes_f.max(), pred_boxes.min(), pred_boxes.max())

        if target_class_ids.shape[1] > 1:
            print(
                gt_boxes_f.shape,
                gt_class_ids_f.shape,
                pred_boxes.shape,
                pred_class_ids.shape,
                pred_scores.shape,
            )
            AP50 = np.float32(
                compute_ap_bbox(
                    gt_boxes_f,
                    gt_class_ids_f,
                    pred_boxes,
                    pred_class_ids,
                    pred_scores,
                    iou_threshold=0.5,
                )
            )

        if len(forgG_index) < 2:
            AP50 = np.float32(0.9666)
        else:
            print("AP50:", AP50)
            if AP50 == 1:
                AP50 = np.float32(0.96666)
    else:
        AP50 = np.float32(0.1)

    return AP50


from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score


def NMI_clus_class(
    target_class_ids,
    target_class_ids2,
    target_bbox,
    mrcnn_class_logits,
    input_image_meta,
    save_seqs,
    log_dir,
    barcodeFolderName,
):

    """This function calculates NMI at each iteration and saves the barcodes for targets on a batch

    Inputs:
        target_class_ids (9, 32):
        target_class_ids2 (9, 32):self
        target_bbox (9, 32, 4):
        mrcnn_class_logits (9, 32, 5):
        input_image_meta (9,):
        log_dir:               root model_save dir to save the detected barcodes inside /seqs/ folder

    """

    #     print('target_class_ids2',target_class_ids2.shape)
    #     print(target_class_ids2)
    print(
        "target_class_ids",
        target_class_ids.shape,
        target_class_ids2.shape,
        np.unique(target_class_ids),
        np.unique(target_class_ids2),
    )
    #     print(target_class_ids)
    #     mi 0.3560548676568811
    # bc_match True
    # (288,) (288,)
    #     epoch=0
    #     print("input_image_meta",input_image_meta)

    #     AP50=compute_ap_bbox(target_bbox, target_class_ids,pred_boxes, target_class_ids2, mrcnn_class_logits,\
    #                iou_threshold=0.5);

    #     AP50=compute_ap_bbox(gt_boxes, gt_class_ids,pred_boxes, pred_class_ids, pred_scores,\
    #                iou_threshold=0.5):

    input_im_id = input_image_meta[:, 0]
    input_im_site = input_image_meta[:, -2]
    #     epoch=np.unique(input_image_meta[:,-1])[0]
    epoch = input_image_meta[0, -1]
    print("input_im_site", input_im_site, epoch)
    #     print('active_class_ids', input_image_meta[:, 12:-2])
    if np.min(input_image_meta[:, 12:-2]) == 0:
        print("check this example")
        asdfafdf

    #     print(epoch,input_im_id) # example [2304. 2305. 2306. 2307. 2308. 2309. 2310. 2311. 2312.]

    #     print(target_class_ids.shape,target_class_ids2.shape,target_bbox.shape,mrcnn_class_logits.shape,input_im_id.shape) #(9, 32) (9, 32) (9, 32, 4) (9, 32, 5) (9,)
    #     print(target_bbox)
    #     print("target_class_ids",target_class_ids)
    #     print("target_class_ids2",target_class_ids2)

    #     target_class_max=np.max(target_class_ids,axis=0)
    #     target_class_min=np.min(target_class_ids,axis=0)
    target_class_min = np.min(target_class_ids2, axis=0)

    #     print('target_class_min',target_class_min.shape)
    #     print('np.where(~target_class_sum==0)',np.where(~target_class_sum==0))
    forgG_index = np.where(target_class_min > 0)[0]

    target_class_ids2[target_class_ids2 == -10] = 0
    target_class_ids[target_class_ids == 5] = 0

    #     print('target_class_ids',target_class_ids)
    #     print('target_class_ids2',target_class_ids2)
    #     print("forgG_index",forgG_index)

    #     print("len",len(target_class_ids.shape),target_class_ids.shape,forgG_index)
    if (len(target_class_ids.shape) > 1) and (len(forgG_index) > 1):

        target_class_ids = target_class_ids[:, forgG_index]
        target_class_ids2 = target_class_ids2[:, forgG_index]
        target_bbox = target_bbox[:, forgG_index, :]
        mrcnn_class_logits = mrcnn_class_logits[:, forgG_index, :]

        logit_arr = mrcnn_class_logits.reshape(
            mrcnn_class_logits.shape[0] * mrcnn_class_logits.shape[1],
            mrcnn_class_logits.shape[2],
        )
        target_class_ids2_flat = target_class_ids2.reshape(
            target_class_ids2.shape[0] * target_class_ids2.shape[1],
        )
        target_class_ids2_flat2 = target_class_ids2.reshape(
            target_class_ids2.shape[0] * target_class_ids2.shape[1], 1
        )
        target_class_ids_flat = target_class_ids.reshape(
            target_class_ids.shape[0] * target_class_ids.shape[1],
        )
        target_probs = np.take_along_axis(
            logit_arr, target_class_ids2_flat2, 1
        ).reshape(target_class_ids2.shape[0], target_class_ids2.shape[1])

        #         print(log_dir.decode('UTF-8'))
        #         print(int(input_im_id[0]))
        #         print(log_dir.decode('UTF-8')+'/seqs/'+str(int(input_im_id[0])))
        #         if epoch>1:
        if target_class_ids.shape[1] > 1:
            #             print(len(target_class_ids.shape),target_class_ids.shape)
            #             print(epoch.shape,np.squeeze(target_bbox[0,:,:]).shape,target_bbox.shape)
            logit_arr2 = logit_arr[:, 1:]
            cross_ent_loss_val = np.round(
                log_loss(target_class_ids2_flat - 1, logit_arr2, labels=[0, 1, 2, 3]), 2
            )
            print("log_loss2:", cross_ent_loss_val)
            acc = accuracy_score(target_class_ids_flat, target_class_ids2_flat)
            nmi = np.float32(
                normalized_mutual_info_score(
                    target_class_ids_flat, target_class_ids2_flat
                )
            )

            if save_seqs:
                print("epoch", epoch)
                a = np.concatenate(
                    (
                        epoch * np.ones((target_class_ids.shape[1], 1), dtype=np.int8),
                        np.squeeze(target_bbox[0, :, :]),
                        target_class_ids2.T,
                        target_probs.T,
                        cross_ent_loss_val * np.ones((target_class_ids.shape[1], 1)),
                        acc * np.ones((target_class_ids.shape[1], 1)),
                        nmi * np.ones((target_class_ids.shape[1], 1)),
                    ),
                    axis=1,
                )  # ,dtype=np.float16)
                #             print('a.shape',a.shape)
                f = open(
                    log_dir.decode("UTF-8")
                    + "/"
                    + barcodeFolderName.decode("UTF-8")
                    + "/im_id_"
                    + str(int(input_im_site[0]))
                    + "_"
                    + str(int(input_im_id[0]))
                    + ".txt",
                    "a",
                )
                np.savetxt(
                    f, a, delimiter=","
                )  # ,header="epoch,y1,x1,y2,x2,0,1,2,3,4,5,6,7,8,p0,p1,p2,p3,p4,p5,p6,p7,p8,ce,acc",comments='') #delimiter=','
                #             delimiter=",", header="ID,AMOUNT",
                #            fmt="%i", comments=''
                f.close()
            #                         with open(log_dir+'/seqs/'+str(image), 'a') as f:
            #                             df.to_csv(f, header=False)

            #             sh=target_class_ids.shape
            #             target_class_ids_flatten=target_class_ids.reshape(sh[0]*sh[1],)
            #             target_class_ids2_flatten=target_class_ids2.reshape(sh[0]*sh[1],)

            #             print(target_class_ids.shape,target_class_ids2.shape) # (288,) (288,)

            #         target_class_ids2=target_class_ids2[target_class_ids>0]        NMI based on forground defind by ground truth
            #         target_class_ids=target_class_ids[target_class_ids>0]

            target_class_ids2 = target_class_ids2_flat[target_class_ids2_flat > 0]
            target_class_ids = target_class_ids_flat[target_class_ids2_flat > 0]

        #         print(target_class_ids,target_class_ids.shape) #(90,)
        #         print(target_class_ids2,target_class_ids2.shape) #(90,)
        #     print(target_class_ids)
        #     print(normalized_mutual_info_score(target_class_ids,target_class_ids2))

        #     if list(target_class_ids)==list(target_class_ids2):
        if len(forgG_index) < 2:
            nmi = np.float32(0.9666)
        else:
            #             nmi=normalized_mutual_info_score(target_class_ids,target_class_ids2)#.astype('float32')
            #             acc=accuracy_score(target_class_ids, target_class_ids2)
            print("nmi:", nmi, "acc:", acc)
            #             if nmi!=1:

            #                 nmi=nmi.astype('float32')
            #             else:
            if nmi == 1:
                acc = np.float32(0.96666)
                nmi = np.float32(0.96666)
    else:
        acc = np.float32(0.1)
        nmi = np.float32(0.1)
    return nmi


def NMI_clus_class2(
    target_class_ids,
    target_class_ids2,
    target_bbox,
    mrcnn_class_logits,
    input_image_meta,
    save_seqs,
    log_dir,
    barcodeFolderName,
):

    """This function calculates NMI at each iteration and saves the barcodes for targets on a batch

    Inputs:
        target_class_ids (9, 32):
        target_class_ids2 (9, 32):self
        target_bbox (9, 32, 4):
        mrcnn_class_logits (9, 32, 5):
        input_image_meta (9,):
        log_dir:               root model_save dir to save the detected barcodes inside /seqs/ folder

    """

    #     print('target_class_ids2',target_class_ids2.shape)
    #     print(target_class_ids2)
    #     print('target_class_ids',target_class_ids.shape)
    #     print(target_class_ids)
    #     mi 0.3560548676568811
    # bc_match True
    # (288,) (288,)
    #     epoch=0
    #     print("input_image_meta",input_image_meta)

    #     AP50=compute_ap_bbox(target_bbox, target_class_ids,pred_boxes, target_class_ids2, mrcnn_class_logits,\
    #                iou_threshold=0.5);

    #     AP50=compute_ap_bbox(gt_boxes, gt_class_ids,pred_boxes, pred_class_ids, pred_scores,\
    #                iou_threshold=0.5):

    input_im_id = input_image_meta[:, 0]
    input_im_site = input_image_meta[:, -2]
    #     epoch=np.unique(input_image_meta[:,-1])[0]
    epoch = input_image_meta[0, -1]
    print("input_im_site", input_im_site, epoch)
    #     print('active_class_ids', input_image_meta[:, 12:-2])
    if np.min(input_image_meta[:, 12:-2]) == 0:
        print("check this example")
        asdfafdf

    #     print(epoch,input_im_id) # example [2304. 2305. 2306. 2307. 2308. 2309. 2310. 2311. 2312.]

    #     print(target_class_ids.shape,target_class_ids2.shape,target_bbox.shape,mrcnn_class_logits.shape,input_im_id.shape) #(9, 32) (9, 32) (9, 32, 4) (9, 32, 5) (9,)
    #     print(target_bbox)
    #     print("target_class_ids",target_class_ids)
    #     print("target_class_ids2",target_class_ids2)

    #     target_class_max=np.max(target_class_ids,axis=0)
    target_class_min = np.min(target_class_ids, axis=0)
    #     print('target_class_sum',target_class_sum.shape)
    #     print('np.where(~target_class_sum==0)',np.where(~target_class_sum==0))
    forgG_index = np.where(target_class_min > 0)[0]

    target_class_ids2[target_class_ids2 == -10] = 0

    #     print('target_class_ids',target_class_ids)
    #     print('target_class_ids2',target_class_ids2)
    #     print("forgG_index",forgG_index)

    #     print("len",len(target_class_ids.shape),target_class_ids.shape,forgG_index)
    if (len(target_class_ids.shape) > 1) and (len(forgG_index) > 1):

        target_class_ids = target_class_ids[:, forgG_index]
        target_class_ids2 = target_class_ids2[:, forgG_index]
        target_bbox = target_bbox[:, forgG_index, :]
        mrcnn_class_logits = mrcnn_class_logits[:, forgG_index, :]

        logit_arr = mrcnn_class_logits.reshape(
            mrcnn_class_logits.shape[0] * mrcnn_class_logits.shape[1],
            mrcnn_class_logits.shape[2],
        )
        target_class_ids2_flat = target_class_ids2.reshape(
            target_class_ids2.shape[0] * target_class_ids2.shape[1],
        )
        target_class_ids2_flat2 = target_class_ids2.reshape(
            target_class_ids2.shape[0] * target_class_ids2.shape[1], 1
        )
        target_class_ids_flat = target_class_ids.reshape(
            target_class_ids.shape[0] * target_class_ids.shape[1],
        )
        target_probs = np.take_along_axis(
            logit_arr, target_class_ids2_flat2, 1
        ).reshape(target_class_ids2.shape[0], target_class_ids2.shape[1])

        #         print(log_dir.decode('UTF-8'))
        #         print(int(input_im_id[0]))
        #         print(log_dir.decode('UTF-8')+'/seqs/'+str(int(input_im_id[0])))
        #         if epoch>1:
        if target_class_ids.shape[1] > 1:
            #             print(len(target_class_ids.shape),target_class_ids.shape)
            #             print(epoch.shape,np.squeeze(target_bbox[0,:,:]).shape,target_bbox.shape)
            logit_arr2 = logit_arr[:, 1:]
            cross_ent_loss_val = np.round(
                log_loss(target_class_ids2_flat - 1, logit_arr2, labels=[0, 1, 2, 3]), 2
            )
            print("log_loss2:", cross_ent_loss_val)
            acc = accuracy_score(target_class_ids_flat, target_class_ids2_flat)
            nmi = np.float32(
                normalized_mutual_info_score(
                    target_class_ids_flat, target_class_ids2_flat
                )
            )

            if save_seqs:
                print("epoch", epoch)
                a = np.concatenate(
                    (
                        epoch * np.ones((target_class_ids.shape[1], 1), dtype=np.int8),
                        np.squeeze(target_bbox[0, :, :]),
                        target_class_ids2.T,
                        target_probs.T,
                        cross_ent_loss_val * np.ones((target_class_ids.shape[1], 1)),
                        acc * np.ones((target_class_ids.shape[1], 1)),
                        nmi * np.ones((target_class_ids.shape[1], 1)),
                    ),
                    axis=1,
                )  # ,dtype=np.float16)
                #             print('a.shape',a.shape)
                f = open(
                    log_dir.decode("UTF-8")
                    + "/"
                    + barcodeFolderName.decode("UTF-8")
                    + "/im_id_"
                    + str(int(input_im_site[0]))
                    + "_"
                    + str(int(input_im_id[0]))
                    + ".txt",
                    "a",
                )
                np.savetxt(
                    f, a, delimiter=","
                )  # ,header="epoch,y1,x1,y2,x2,0,1,2,3,4,5,6,7,8,p0,p1,p2,p3,p4,p5,p6,p7,p8,ce,acc",comments='') #delimiter=','
                #             delimiter=",", header="ID,AMOUNT",
                #            fmt="%i", comments=''
                f.close()
            #                         with open(log_dir+'/seqs/'+str(image), 'a') as f:
            #                             df.to_csv(f, header=False)

            #             sh=target_class_ids.shape
            #             target_class_ids_flatten=target_class_ids.reshape(sh[0]*sh[1],)
            #             target_class_ids2_flatten=target_class_ids2.reshape(sh[0]*sh[1],)

            #             print(target_class_ids.shape,target_class_ids2.shape) # (288,) (288,)

            #         target_class_ids2=target_class_ids2[target_class_ids>0]        NMI based on forground defind by ground truth
            #         target_class_ids=target_class_ids[target_class_ids>0]

            target_class_ids2 = target_class_ids2_flat[target_class_ids2_flat > 0]
            target_class_ids = target_class_ids_flat[target_class_ids2_flat > 0]

        #         print(target_class_ids,target_class_ids.shape) #(90,)
        #         print(target_class_ids2,target_class_ids2.shape) #(90,)
        #     print(target_class_ids)
        #     print(normalized_mutual_info_score(target_class_ids,target_class_ids2))

        #     if list(target_class_ids)==list(target_class_ids2):
        if len(forgG_index) < 2:
            nmi = np.float32(0.9666)
        else:
            #             nmi=normalized_mutual_info_score(target_class_ids,target_class_ids2)#.astype('float32')
            #             acc=accuracy_score(target_class_ids, target_class_ids2)
            print("nmi:", nmi, "acc:", acc)
            #             if nmi!=1:

            #                 nmi=nmi.astype('float32')
            #             else:
            if nmi == 1:
                acc = np.float32(0.96666)
                nmi = np.float32(0.96666)
    else:
        acc = np.float32(0.1)
        nmi = np.float32(0.1)
    return nmi


def save_barcodes_whileTrain(target_class_ids2, img_ids, mrcnn_bbox):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    target_class_ids2:    array of labels, outputted by kmeans
    img_ids:                 true labels
    n_clusters:             number of clusters in the dataset

    mrcnn_bbox:    a tuple containing the accuracy and confusion matrix,
                in that order
    """

    print("im_ids:   ", img_ids)
    print(target_class_ids2.shape)
    print(img_ids.shape)
    print(mrcnn_bbox.shape)

    #     import csv

    #     # open the file in the write mode
    #     with open('path/to/csv_file', 'w') as f:
    #         # create the csv writer
    #         writer = csv.writer(f)

    #         # write a row to the csv file
    #         writer.writerow(row)

    return np.float32(1)


# [ 0.7880706  -0.54546934 -0.43432173 -0.7956673 ]
#   [-1.0764822  -1.3608562  -0.80553406 -0.5980677 ]
#   [-1.9034969   0.5264623  -1.2145798  -0.11363877]


def instance_nucl_cell_mask(mask_im_resized, xCenter, yCenter):
    from skimage.segmentation import flood_fill

    cell_color = (255, 255, 255)
    nuclei_color = (255, 0, 0)

    indices_r = np.where(np.all(mask_im_resized == nuclei_color, axis=-1))
    indices_w = np.where(np.all(mask_im_resized == cell_color, axis=-1))

    # mask_im_2=mask_im.copy()
    # mask_im_2[indices]=0
    mask_im_2 = np.zeros(np.shape(mask_im_resized)[0:2]).astype("uint8")
    nucl_c = 100
    cell_c = 200
    mask_im_2[indices_r] = nucl_c
    mask_im_2[indices_w] = cell_c
    cellMask1 = flood_fill(mask_im_2, (yCenter, xCenter), nucl_c, connectivity=1)
    nucl_ins_mask = flood_fill(cellMask1, (yCenter, xCenter), 10, connectivity=1)
    nucl_ins_mask[nucl_ins_mask != 10] = 0

    mask_im_3 = np.zeros(np.shape(mask_im_resized)[0:2]).astype("uint8")
    mask_im_3[indices_w] = cell_c
    # cellMask1 = flood_fill(mask_im_2,(yCenter,xCenter),nucl_c,connectivity=1)
    cell_ins_mask = flood_fill(mask_im_3, (yCenter, xCenter), 20, connectivity=1)
    cell_ins_mask[cell_ins_mask != 20] = 0
    from skimage.morphology import dilation, square

    cell_ins_mask = dilation(cell_ins_mask, square(3))
    return nucl_ins_mask, cell_ins_mask


if 0:
    import pandas as pd

    metadata_dir_257 = "/storage/data/marziehhaghighi/pooledCP/workspace/metadata/20210422_6W_CP257/Barcodes.csv"
    metadata_orig0 = pd.read_csv(metadata_dir_257)
    metadata_orig = metadata_orig0[~metadata_orig0["sgRNA"].isnull()].reset_index(
        drop=True
    )
    metadata_orig["prefix9"] = metadata_orig["sgRNA"].apply(lambda x: x[0:9])
    barcode_ref_list = metadata_orig.prefix9.unique().tolist()

    map_dict = {"A": 3, "T": 4, "G": 2, "C": 1}  # like cp228
    barcode_ref_array = np.zeros((len(barcode_ref_list), 9))
    barcode_ref_num_list = []
    for ba in range(len(barcode_ref_list)):
        barc = list(barcode_ref_list[ba])
        barcode_ref_num_list.append("".join([str(map_dict[b]) for b in barc]))
        barcode_ref_array[ba, :] = [map_dict[b] for b in barc]

# config.barcode_ref_array=barcode_ref_array;


def map_to_closest_barcode(input_barcode, barcode_ref_array, mrcnn_class_logits):
    """
    barcode_ref_array: (n_barcodes, 9) np array
    input_barcode: (9,1) np array
    spot_class_probs: (9,4)
    """
    #     print("input_barcode",input_barcode)
    #     input_barcode=[4., 1., 3., 3., 3., 3., 1., 2., 0]
    #     print("mrcnn_class_logits",mrcnn_class_logits)
    #     distances=[]
    #     for i in range(barcode_ref_array.shape[0]):
    #         distances.append(distance.hamming(input_barcode, barcode_ref_array[i,:]))
    #
    #     candide_bc_ind=np.where((np.array(distances)==np.min(distances)))[0]

    #   a MUCH faster implementation for hamming dist calculations
    distances = (input_barcode != barcode_ref_array).sum(axis=1)
    candide_bc_ind = np.argwhere((np.array(distances) == np.min(distances))).flatten()
    #     candide_bc_ind=np.argmin(distances)

    if candide_bc_ind.size > 1:  # faster comparing to next par implementation
        #         print("candide_bc_ind",candide_bc_ind)

        potential_bcs = barcode_ref_array[candide_bc_ind, :].astype(int) - 1
        print("potential_bcs", potential_bcs + 1)

        potential_bcs_probs = mrcnn_class_logits[
            np.arange(len(mrcnn_class_logits)), potential_bcs
        ]

        #         print("potential_bcs_probs",potential_bcs_probs.shape)
        #         potential_bcs_match_prob=np.sum(potential_bcs_probs,axis=1)
        potential_bcs_match_prob = np.prod(potential_bcs_probs, axis=1)
        print("potential_bcs_match_prob", potential_bcs_match_prob)

        matched_bc = (
            potential_bcs[
                np.argwhere(
                    np.array(potential_bcs_match_prob)
                    == np.max(potential_bcs_match_prob)
                ),
                :,
            ][0]
            + 1
        )
        selected_bc_prob = np.max(potential_bcs_match_prob)

        ###################### Below implementation was slower
        #     if candide_bc_ind.size>1:
        #         potential_bcs=barcode_ref_array[candide_bc_ind,:]
        #         total_p=[]
        #         for c in range(candide_bc_ind.size):
        #             potential_bc=potential_bcs[c,:]
        #             potential_bc1hot=np.zeros((potential_bc.size, 4))
        #             potential_bc1hot[np.arange(potential_bc.size),potential_bc.astype(int)-1] = 1
        #             total_p.append(np.sum(np.multiply(potential_bc1hot, mrcnn_class_logits)))

        #         matched_bc=potential_bcs[np.where((np.array(total_p)==np.max(total_p)))[0],:]

        # if there are multiple barcodes with the same probabilities then randomly pick one
        if matched_bc.shape[0] > 1:
            print("matched_bc.shape[0]", matched_bc.shape[0])
            matched_bc = matched_bc[np.random.choice(matched_bc.shape[0]), :]
    #             matched_bc=potential_bcs[np.where((np.array(total_p)==np.max(total_p)))[0],:]

    else:
        potential_bcs_probs = mrcnn_class_logits[
            np.arange(len(mrcnn_class_logits)),
            barcode_ref_array[candide_bc_ind[0], :].astype(int) - 1,
        ]

        selected_bc_prob = np.prod(potential_bcs_probs)
        #         print("xx",potential_bcs_probs,selected_bc_prob)
        #     print("candide_bc_ind",candide_bc_ind)
        #     for c in range(candide_bc_ind):
        #         candide_bc_ind[c]
        #     print(candide_bc_ind)
        matched_bc = barcode_ref_array[candide_bc_ind[0], :]

    return matched_bc, selected_bc_prob


# def map_to_barcode_min_Hdist(input_barcode,barcode_ref_array):
#     """
#     map input barcode to a barcode with results min toral distance to all the points in the ref library
#     barcode_ref_array: (n_barcodes, 9) np array
#     input_barcode: (1,9) np array
#     """


# #     input_barcode=[4., 1., 3., 3., 3., 3., 1., 2., 0]
#     distances=[]
#     for i in range(barcode_ref_array.shape[0]):
#         distances.append(distance.hamming(input_barcode, barcode_ref_array[i,:]))

#     matched_bc=barcode_ref_array[distances.index(np.min(distances)),:]
#     return matched_bc , np.min(distances)


def map_to_barcode_min_Hdist(input_barcode, barcode_ref_array):
    """
    map input barcode to a barcode with results min toral distance to all the points in the ref library
    barcode_ref_array: (n_barcodes, 9) np array
    input_barcode: (1,9) np array
    """

    #     input_barcode=[4., 1., 3., 3., 3., 3., 1., 2., 0]
    distances = []
    #     for i in range(barcode_ref_array.shape[0]):
    #         distances.append(distance.hamming(input_barcode, barcode_ref_array[i,:]))
    #     matched_bc=barcode_ref_array[distances.index(np.min(distances)),:]

    distances = (input_barcode != barcode_ref_array).sum(axis=1)
    candide_bc_ind = np.argwhere((np.array(distances) == np.min(distances))).flatten()
    potential_bcs = barcode_ref_array[candide_bc_ind, :].astype(int)

    return potential_bcs[0], np.min(distances)
