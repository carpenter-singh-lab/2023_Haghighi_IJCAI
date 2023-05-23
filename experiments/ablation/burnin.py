"""
This script performs genrates initial pseudo-labler for the burn-in stage and saves the models which used LQ and HQ labels in init_teacher_by_lq
and init_teacher_by_hq folders


Data Used for Burn-in stage:

- Labeled Data: D_l (one site of plate B) is used for burnin stage


Author:
Marzieh Haghighi
"""


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pdb
import time

import numpy as np
import pandas as pd

import barcodefit.model.barcode_calling_KD2 as modellib
from barcodefit.dataobjects import spot, spot_utils

label_quality = ["LQ"]  # "LQ" or "HQ"

batch = "20210124_6W_CP228"
plate = "B"
well = "Well4"


which_gpu = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

d_inf = [batch, plate, well]

# read metadata
(
    dfInfo,
    dfInfo_comp,
    dataset_train_ls,
    dataset_val,
    barcode_ref_array,
) = spot_utils.read_metadata(d_inf, "train")


############################ config model  #####################
batch = d_inf[0]
config = spot.spotConfig()
config.batchplate_well = d_inf[0].split("_")[-1] + d_inf[1] + "_" + d_inf[2]

config.lr = 10 * config.LEARNING_RATE
config.LEARNING_MOMENTUM = 0.5

config.barcode_ref_array = barcode_ref_array
config.load_cropped_presaved = True


if label_quality == "LQ":
    MODEL_DIR = "./experiments/ablation/temp/init_teacher_by_lq"
    config.create_mask = True
elif label_quality == "HQ":
    MODEL_DIR = "./experiments/ablation/temp/init_teacher_by_hq"
    config.create_mask = False


# ########################## burn-in, using labeled samples (4356-(484*9) images are in each site)

# config.pretrained_model_type='class'
config.assign_label_mode = "classification"
config.rpn_clustering = False
config.BC_MATCH = False
config.img_aug = False
config.init_with = "scratch"
config.layers_to_tune = "all"
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.train(
    dataset_train_ls[20:21],
    dataset_val,
    learning_rate=config.lr,
    epochs=3,
    layers=config.layers_to_tune,
)
