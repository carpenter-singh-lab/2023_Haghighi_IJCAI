"""
This script takes pseudo-labler trained on D_l and take each site of the unlabled well and:
 - Pseudo-label and save the results  
 - update the model using the pseudo-labels enhanced by priviledged information 

Notes:
- This in an incremental learning strategy which is computationally efficient for incorporation of unlabeled data and applying/saving the results

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

batch='20210124_6W_CP228';batch_abbrev='CP228'
plate = "B"
well = "Well4"

# plate='A';
# well='Well3';

which_gpu = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

d_inf=[[batch,batch_abbrev],plate,well]


##################### read metadata
# (
#     dfInfo,
#     dfInfo_comp,
#     dataset_train_ls,
#     dataset_val,
#     barcode_ref_array,
# ) = spot_utils.read_metadata(d_inf,dataset_dir, "train")

metadata_dir = "./resource/"
barcode_ref_list, codebook, barcode_ref_array = spot_utils.read_barcode_list(metadata_dir)

##################### config model
# batch = d_inf[0]
config = spot.spotConfig()
config.batchplate_well=batch_abbrev+d_inf[1]+'_'+d_inf[2]

config.init_with = "fixed"
config.assign_label_mode = "clustering"
config.lr = 10 * config.LEARNING_RATE
config.LEARNING_MOMENTUM = 0.5

config.barcode_ref_array = barcode_ref_array
# config.load_cropped_presaved = True
config.pretrained_model_path = "./experiments/plepi-iss/pseudo-labeler"
MODEL_DIR = "./experiments/plepi-iss/temp/"

dataset_dir='/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/'
config.im_Dir=dataset_dir+'/'+batch+'/images_aligned_cropped/'
config.dl_meta_Dir=dataset_dir+'/workspace/DL_meta/'+batch+'/'

########################## Using unlabled data in a incremental online learning fashion
config.pretrained_model_type = "class"
config.assign_label_mode = "clustering"
config.img_aug = True
config.rpn_clustering = True
config.create_mask = False
config.BC_MATCH = True

config.GRADIENT_CLIP_NORM = 10
config.save_seqs = 0
config.save_seqs2 = 1
config.tau_g = 1e-04
config.layers_to_tune = "heads"
config.STEPS_PER_EPOCH = 484
config.update_teacher_n_batch = 1
config.TRAIN_ROIS_PER_IMAGE = 32 * 3 * 4
config.RPN_TRAIN_ANCHORS_PER_IMAGE = 32 * 3 * 4
config.list_of_sites=[21,16,60,19,47]
config.val_list_of_sites=[49]
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# print(len(dataset_train_ls),len(dataset_val))
# pdb.set_trace()
model.train(
    learning_rate=config.lr,
    epochs=5,
    layers=config.layers_to_tune,
)

########################## spot_level_to_cell_level_assignments
matched_flag = "2"
epoch_filter_list = list(range(105))

model_direc = model.log_dir
model_params = [model_direc, epoch_filter_list]
test_sites = [60]
(
    ngs_match,
    cell_recovery_rate,
    call_dl_df,
) = spot_utils.spot_level_to_cell_level_assignments(
    d_inf, test_sites,dataset_dir, model_params, matched_flag
)


call_dl_df_site = call_dl_df[call_dl_df["Barcodes_called_medP"] > 0.9].reset_index(
    drop=True
)
call_dl_df_site = call_dl_df_site[
    call_dl_df_site["Barcodes_called_prodP"] > 0.2
].reset_index(drop=True)

call_dl_df_site.to_csv("./results/paper_results/PLePI-ISS_B_Well4_60.csv")


print("ngs_match=", ngs_match)
print("cell_recovery_rate=", cell_recovery_rate)

with open("./results/paper_results/table3_plepi_results.txt", "a") as f:
    f.write(f"ngs_match={ngs_match}, cell_recovery_rate={cell_recovery_rate}\n")
