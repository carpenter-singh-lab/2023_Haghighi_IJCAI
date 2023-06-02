"""
Author:
Marzieh Haghighi
"""

import itertools
import os
import pickle
import shutil
import sys
import time
import zipfile
from ast import literal_eval
# from skimage.draw import disk,circle
from collections import defaultdict

import numpy as np
import pandas as pd
# from barcodefit.dataobjects import CellClass
import skimage.io
from skimage.draw import disk
from skimage.morphology import dilation, square
from skimage.util import img_as_ubyte

# from barcodefit import model as modellib, utils
# from barcodefit.model import barcode_calling as modellib
from barcodefit.model import model_utils as utils
from barcodefit.model.config import Config

############################################################
#  Spot Class Configurations
############################################################


class spotConfig(Config):
    """
    configs for multichannel ISS images containing many spots
    Derives from the base Config class and overrides values specific
    to the spot dataset.
    """

    # Give the configuration a recognizable name
    NAME = "spotsISS"

    BACKBONE = "resnet50"  # "resnet50"

    #  Batch size is 8-->2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 9

    # Flag indicating if barcode projection step is enforced
    BC_MATCH = True

    rpn_clustering = False

    #     TOP_DOWN_PYRAMID_SIZE = 64 #256
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 bases

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256  # 1024
    IMAGE_MAX_DIM = 256  # 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    #     BACKBONE_STRIDES = [4,8]

    ROI_POSITIVE_RATIO = 0.8  # 0.33*2#0.33*2
    #     ROI_POSITIVE_RATIO = 0.33

    USE_RPN_ROIS = True

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 9000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9  # 0.7

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 3000
    POST_NMS_ROIS_INFERENCE = 100
    FPN_CLASSIF_FC_LAYERS_SIZE = 64
    #     FPN_CLASSIF_FC_LAYERS_SIZE=512

    TRAIN_ROIS_PER_IMAGE = 32 * 3 * 2
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32 * 3 * 2
    #     RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_RATIOS = [1]
    DETECTION_MIN_CONFIDENCE = 0.7

    #     TRAIN_BN = True

    DETECTION_NMS_THRESHOLD = 0.01

    STEPS_PER_EPOCH = (
        484  # 50   ### should be at least equal to the number of crops in each image
    )

    VALIDATION_STEPS = 5
    #    head='def'; #'def','unet'

    MAX_GT_INSTANCES = 200  # ? check this
    IMAGE_CHANNEL_COUNT = 4

    DETECTION_MAX_INSTANCES = 200

    #     for classification
    CLASS_LOSS_WEIGHTS = {
        "rpn_class_loss": 1,
        "rpn_bbox_loss": 0.1,
        "mrcnn_class_loss": 0.1,
        "mrcnn_bbox_loss": 0.1,
    }

    CLUS_LOSS_WEIGHTS = {
        "rpn_class_loss": 1,
        "rpn_bbox_loss": 1,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1,
    }

    # Pooled ROIs
    POOL_SIZE = 7

    positive_roi_iou_min_thr = 0.5  # (defualt is 0.5)
    negative_roi_iou_max_thr = 0.5  # (defualt is 0.5)

    USE_MINI_MASK = False

    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    #     MEAN_PIXEL =np.zeros((IMAGE_CHANNEL_COUNT,), dtype=int);
    MEAN_PIXEL = np.array([0, 0, 0, 0])

    # LEARNING_RATE = 0.001
    # LEARNING_MOMENTUM = 0.9

    IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES + 2

    pretrained_model_path = (
        "/spotsISS/B_Well4/pretrained_scratch_train_all_lr_0.01_class/"
        "spotsiss20220731T2307/mask_rcnn_spotsiss_0002.h5"
    )

    stage5_enabled = True
    save_seqs = 1
    save_seqs2 = 0
    n_workers_to_use = 9
    max_q_size_generator = 50
    starting_site = -1
    model_save_epoch_period = 58
    #     tau_g=1e-20
    tau_g = (
        1e-8  # confidence thresholding for barcode (group of samples) pseudo-labling
    )
    tau_s = (
        0.9  # confidence thresholding for single sample (base letter) pseudo-labling
    )

    GRADIENT_CLIP_NORM = 1
    load_cropped_presaved = False
    load_tiff_crop_online = True



############################################################
# Spots Dataset
############################################################


class spotsDataset(utils.Dataset):
    def load_spots(
        self, dfInfo, subset, class_ids=None, class_map=None, return_cell=False
    ):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        dfInfo: dataframe containg all data and mask addresses and annotations
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        if subset == "minival" or subset == "valminusminival":
            subset = "val"

        #         if subset == "train":
        #         df_Info_t = df_Info[df_Info['subset_label']=="train"];

        dfInfo = dfInfo[dfInfo["subset_label"] == subset]

        self.add_class("spot", 1, "C")
        self.add_class("spot", 2, "G")
        self.add_class("spot", 3, "A")
        self.add_class("spot", 4, "T")

        # Add images
        image_ids = np.sort(dfInfo["image_id"].unique())

        for i in image_ids:
            dfInfo_i = dfInfo[dfInfo["image_id"] == i].reset_index(drop=True)
            #             print(self.imgs[i]['im_paths'],len(cell.imgs[i]['im_paths']))
            #             print(i,self.imgs[i]["im_Center_X"],cell.imgs[i]["im_Center_Y"])
            self.add_image(
                "spot",
                image_id=i,
                site=dfInfo_i["Metadata_Site"].values[0],
                path=dfInfo_i["im_paths"].values[0],
                # overlay_dir=dfInfo_i["overlay_dir"].values[0],
                #                 path=os.path.join(image_dir, cell.imgs[i]['file_name']),
                im_Center_X=dfInfo_i["im_Center_X"].values[0],
                im_Center_Y=dfInfo_i["im_Center_Y"].values[0],
                annotations=dfInfo_i[["bbox", "cat_id"]],
            )

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,nChannels] Numpy array."""
        # Load images

        listOfPaths = literal_eval(self.image_info[image_id]["path"])

        im_Center_X, im_Center_Y = (
            round(self.image_info[image_id]["im_Center_X"]),
            round(self.image_info[image_id]["im_Center_Y"]),
        )
        #         print(im_Center_X,im_Center_Y)
        center_list = [im_Center_X, im_Center_Y]
        cr_image_corr = read_crop_image(listOfPaths, center_list)
        return cr_image_corr

    def load_image_site(self, image_id):
        """Load the specified image and return a [H,W,nChannels] Numpy array."""
        # Load images
        originalURLpath = "dataParentFolder"
        toBeReplaced = "/data1/data/marziehhaghighi/pooledCP/"
        listOfPaths = literal_eval(self.image_info[image_id]["path"])

        imagesList = []
        for imPath in listOfPaths:
            #             print("imPath",imPath)
            imPath = imPath.replace(originalURLpath, toBeReplaced)
            #         print(imPath)
            im_uint16 = np.squeeze(skimage.io.imread(imPath))  # images are 'uint16'
            im_max_2scale = im_uint16.max()
            im_uint8 = ((im_uint16 / im_max_2scale) * 255).astype("uint8")
            imagesList.append(im_uint8)

        image = np.stack(imagesList, axis=-1)

        return image


    def load_image_memmap(self, image_id,site_image,IMAGES_PER_GPU):
        """Load the specified image and return a [H,W,nChannels] Numpy array.
        """

        cycle=image_id%IMAGES_PER_GPU
        crop_y_cent,crop_x_cent=round(self.image_info[image_id]['im_Center_X']),round(self.image_info[image_id]['im_Center_Y'])
        
        cropped_im_dim=256;
        cropped_im_dim_h=int(cropped_im_dim/2)

        cr_br_x_b=int(crop_x_cent-cropped_im_dim_h)
        cr_br_y_b=int(crop_y_cent-cropped_im_dim_h)
        cr_br_x_t=int(crop_x_cent+cropped_im_dim_h)
        cr_br_y_t=int(crop_y_cent+cropped_im_dim_h)

        cr_image=np.squeeze(site_image[cycle,cr_br_x_b:cr_br_x_t,cr_br_y_b:cr_br_y_t,:])

        cr_shapeX,cr_shapeY = cr_image.shape[0], cr_image.shape[1];
        if cr_shapeX<cropped_im_dim or cr_shapeY<cropped_im_dim:
            print('crxy:',cropped_im_dim-cr_shapeX,cropped_im_dim-cr_shapeY)
            cr_image_corr = np.pad(cr_image, ((0, cropped_im_dim-cr_shapeX), (0, cropped_im_dim-cr_shapeY),\
             (0, 0)),'constant', constant_values=(0))
        else:
            cr_image_corr=cr_image


        return cr_image_corr        




    def load_bbox(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        #         orig_im_w=4500; #cp074
        #         orig_im_w=5500; #cp228

        image_info = self.image_info[image_id]
        #         instance_masks = []
        instance_bboxes = []
        #         objectOrders=[]
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        #         cropped_im_dim=256;
        #         cropped_im_dim_h=int(cropped_im_dim/2)
        margin = 4
        #         for annotation in annotations:
        for a in range(annotations.shape[0]):

            class_id = annotations.loc[a, "cat_id"]

            if class_id:
                bboxx = literal_eval(annotations.loc[a, "bbox"])

                instance_bboxes.append(
                    (
                        bboxx[1] - margin,
                        bboxx[0] - margin,
                        bboxx[1] + bboxx[3] + margin,
                        bboxx[0] + bboxx[2] + margin,
                    )
                )
                #                 print((bboxx[1]-4,bboxx[0]-4,bboxx[1]+bboxx[3]+4,bboxx[0]+bboxx[2]+4))  12x12 box
                class_ids.append(class_id)

        cropped_im_dim = 256
        if class_ids:
            bboxes = np.clip(
                np.stack(instance_bboxes, axis=0), 0, cropped_im_dim
            ).astype(np.int16)
            class_ids = np.array(class_ids, dtype=np.int32)
            return bboxes, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)
        

    def create_bbox(self, images3D):
        """Take input ISS multi-channel image of one cycle
            - max project across channels and
            - by thresholding detects and lables spots

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        #         from skimage.draw import disk,circle
        # max_proj_im=np.percentile(images3D,(25),axis=-1)

        cropped_im_dim = 256
        cropped_im_dim_h = int(cropped_im_dim / 2)
        bbox_half_len = 2

        max_proj_im = np.max(images3D, axis=-1)

        thrsh = np.percentile(max_proj_im, (99))
        index_spots = np.where(max_proj_im > thrsh)

        if index_spots[0].shape[0] > 0:
            armaxx = np.argmax(images3D[index_spots[0], index_spots[1], :], axis=-1)

            centers = []
            instance_bboxes = []
            class_ids = []

            for ri in range(len(armaxx)):
                center_y, center_x = index_spots[1][ri], index_spots[0][ri]

                if ri > 0:
                    if (abs(center_x - centers[-1][0]) > bbox_half_len * 2) or (
                        abs(center_y - centers[-1][1]) > bbox_half_len * 2
                    ):
                        instance_bboxes.append(
                            (
                                center_x - bbox_half_len,
                                center_y - bbox_half_len,
                                center_x + bbox_half_len,
                                center_y + bbox_half_len,
                            )
                        )
                        class_ids.append(armaxx[ri] + 1)
                        centers.append([center_x, center_y])

                else:
                    instance_bboxes.append(
                        (
                            center_x - bbox_half_len,
                            center_y - bbox_half_len,
                            center_x + bbox_half_len,
                            center_y + bbox_half_len,
                        )
                    )
                    class_ids.append(armaxx[ri] + 1)
                    centers.append([center_x, center_y])

            # channels1=["C","G","A","T"]

            bboxes = np.clip(
                np.stack(instance_bboxes, axis=0), 0, cropped_im_dim
            ).astype(np.int16)
            class_ids = np.array(class_ids, dtype=np.int32)
            return bboxes, class_ids
        else:
            return super(CocoDataset, self).load_mask(image_id)


    def load_overlay(self, image_id):

        import skimage.io
        from skimage.transform import downscale_local_mean, rescale, resize

        #         orig_im_w=4500; #cp074
        orig_im_w = 5500  # cp228

        overlay_dir = self.image_info[image_id]["overlay_dir"]
        ov_im = resize(
            skimage.io.imread(overlay_dir),
            (orig_im_w, orig_im_w),
            mode="constant",
            preserve_range=True,
            order=0,
        ).astype("uint8")

        #         print(ov_im.shape)
        crop_y_cent, crop_x_cent = (
            round(self.image_info[image_id]["im_Center_X"]),
            round(self.image_info[image_id]["im_Center_Y"]),
        )

        cropped_im_dim = 256
        cropped_im_dim_h = int(cropped_im_dim / 2)

        cr_br_x_b = crop_x_cent - cropped_im_dim_h
        cr_br_y_b = crop_y_cent - cropped_im_dim_h
        cr_br_x_t = np.min([crop_x_cent + cropped_im_dim_h, orig_im_w])
        cr_br_y_t = np.min([crop_y_cent + cropped_im_dim_h, orig_im_w])

        cr_image_corr = np.zeros((cropped_im_dim, cropped_im_dim, 3))
        cr_image_corr[: (cr_br_x_t - cr_br_x_b), : (cr_br_y_t - cr_br_y_b), :] = ov_im[
            cr_br_x_b:cr_br_x_t, cr_br_y_b:cr_br_y_t, :
        ]

        return cr_image_corr


########################### AUX functions


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def convertAnnot2dict(dfInfo):
    dataset = {}

    pwsc_colName = dfInfo.columns[dfInfo.columns.str.contains("P-")].values[0]
    loc_im_colNames = dfInfo.columns[dfInfo.columns.str.contains("im_Center")].tolist()
    loc_colNames = dfInfo.columns[dfInfo.columns.str.contains("Center")].tolist()
    ########## form images dict
    imageDf = pd.DataFrame(
        columns=[
            "id",
            "im_paths",
            pwsc_colName,
            "height",
            "width",
            "overlay_dir",
            "Metadata_Site",
        ]
        + loc_im_colNames
    )
    imageDf[
        [
            "id",
            "im_paths",
            pwsc_colName,
            "height",
            "width",
            "overlay_dir",
            "Metadata_Site",
        ]
        + loc_im_colNames
    ] = dfInfo[
        [
            "image_id",
            "im_paths",
            pwsc_colName,
            "height",
            "width",
            "overlay_dir",
            "Metadata_Site",
        ]
        + loc_im_colNames
    ].drop_duplicates(
        ignore_index=True
    )
    dataset["images"] = imageDf.to_dict(orient="records")

    ########## form annot dict
    annotDf = pd.DataFrame(
        columns=[
            "image_id",
            "bbox",
            "category_id",
            "id",
            pwsc_colName,
            "ObjectNumber",
            "mask",
        ]
        + loc_colNames
    )
    annotDf[
        ["image_id", "bbox", "category_id", pwsc_colName, "ObjectNumber", "mask"]
        + loc_colNames
    ] = dfInfo[
        ["image_id", "bbox", "cat_id", pwsc_colName, "ObjectNumber", "mask"]
        + loc_colNames
    ]
    annotDf["id"] = annotDf.index + 1
    dataset["annotations"] = annotDf.to_dict(orient="records")
    #     print('hey',dataset['annotations'])

    ########## form cat dict
    categDf = pd.DataFrame(columns=["supercategory", "id", "name"])
    categDf[["id", "name"]] = dfInfo[["cat_id", "Metadata_Label"]].drop_duplicates(
        ignore_index=True
    )
    categDf["supercategory"] = "cell"
    dataset["categories"] = categDf.to_dict(orient="records")

    return dataset


def convertAnnot2dict2(dfInfo3):
    ### When using compact dfInfo

    ##### Expand the compact format
    list_ofCycles = []

    # dfInfo42=dfInfo3.copy()
    for i in range(9):
        print(i)
        dfInfo42 = dfInfo3.copy()
        dfInfo42["Metadata_Cycle"] = i + 1
        list_ofCycles.append(dfInfo42)

    dfInfo = pd.concat(list_ofCycles).reset_index(drop=True)
    dfInfo["cat_id"] = dfInfo.apply(
        lambda x: eval(x["BarcodeList_cat_id"])[x["Metadata_Cycle"] - 1], axis=1
    )
    dct = {"1": "A", "2": "T", "3": "C", "4": "G"}
    dfInfo["Metadata_Label"] = dfInfo["cat_id"].apply(
        lambda x: list(map(dct.get, str(x)))
    )
    ###############

    dataset = {}

    pwsc_colName = dfInfo.columns[dfInfo.columns.str.contains("P-")].values[0]
    loc_im_colNames = dfInfo.columns[dfInfo.columns.str.contains("im_Center")].tolist()
    loc_colNames = dfInfo.columns[dfInfo.columns.str.contains("Center")].tolist()
    ########## form images dict
    imageDf = pd.DataFrame(
        columns=["id", "im_paths", pwsc_colName, "height", "width"] + loc_im_colNames
    )
    imageDf[
        ["id", "im_paths", pwsc_colName, "height", "width", "overlay_dir"]
        + loc_im_colNames
    ] = dfInfo[
        ["image_id", "im_paths", pwsc_colName, "height", "width", "overlay_dir"]
        + loc_im_colNames
    ].drop_duplicates(
        ignore_index=True
    )
    dataset["images"] = imageDf.to_dict(orient="records")

    ########## form annot dict
    annotDf = pd.DataFrame(
        columns=[
            "image_id",
            "bbox",
            "category_id",
            "id",
            pwsc_colName,
            "ObjectNumber",
            "mask",
        ]
        + loc_colNames
    )
    annotDf[
        ["image_id", "bbox", "category_id", pwsc_colName, "ObjectNumber", "mask"]
        + loc_colNames
    ] = dfInfo[
        ["image_id", "bbox", "cat_id", pwsc_colName, "ObjectNumber", "mask"]
        + loc_colNames
    ]
    annotDf["id"] = annotDf.index + 1
    dataset["annotations"] = annotDf.to_dict(orient="records")
    #     print('hey',dataset['annotations'])

    ########## form cat dict
    categDf = pd.DataFrame(columns=["supercategory", "id", "name"])
    categDf[["id", "name"]] = dfInfo[["cat_id", "Metadata_Label"]].drop_duplicates(
        ignore_index=True
    )
    categDf["supercategory"] = "cell"
    dataset["categories"] = categDf.to_dict(orient="records")

    return dataset


def read_crop_image(listOfPaths, center_list):

    originalURLpath = "dataParentFolder"
    #     toBeReplaced='/data1/data/marziehhaghighi/pooledCP/'
    #         toBeReplaced='/storage/data/marziehhaghighi/pooledCP/'
    toBeReplaced = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"

    crop_y_cent, crop_x_cent = center_list[0], center_list[1]

    imagesList = []
    #         print("ID",image_id,listOfPaths)
    for imPath in listOfPaths:
        #             print("imPath",imPath)
        imPath = imPath.replace(originalURLpath, toBeReplaced)
        #         print(imPath)
        im_uint16 = np.squeeze(skimage.io.imread(imPath))  # images are 'uint16'
        im_max_2scale = im_uint16.max()
        im_uint8 = ((im_uint16 / im_max_2scale) * 255).astype("uint8")
        imagesList.append(im_uint8)

    image = np.stack(imagesList, axis=-1)
    mean_array = np.array([np.mean(image[:, :, i]) for i in range(image.shape[2])])


    cropped_im_dim = 256
    cropped_im_dim_h = int(cropped_im_dim / 2)

    cr_br_x_b = int(crop_x_cent - cropped_im_dim_h)
    cr_br_y_b = int(crop_y_cent - cropped_im_dim_h)
    cr_br_x_t = int(crop_x_cent + cropped_im_dim_h)
    cr_br_y_t = int(crop_y_cent + cropped_im_dim_h)
    print(image.shape)
    print(cr_br_x_b, cr_br_y_b, cr_br_x_t, cr_br_y_t)
    cr_image = image[cr_br_x_b:cr_br_x_t, cr_br_y_b:cr_br_y_t, :]



    cr_shapeX, cr_shapeY = cr_image.shape[0], cr_image.shape[1]
    if cr_shapeX < cropped_im_dim or cr_shapeY < cropped_im_dim:
        print("crxy:", cropped_im_dim - cr_shapeX, cropped_im_dim - cr_shapeY)
        cr_image_corr = np.pad(
            cr_image,
            ((0, cropped_im_dim - cr_shapeX), (0, cropped_im_dim - cr_shapeY), (0, 0)),
            "constant",
            constant_values=(0),
        )
    else:
        cr_image_corr = cr_image

    return cr_image_corr
