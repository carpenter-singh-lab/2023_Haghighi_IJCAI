"""
This script performs ablation experiments for PLePI-ISS model and should reproduce the results in Table 2 of the paper

Data Splits:

- Labeled Data: D_l (one site of plate B) is used for pretraining for all the models
- Unlabeled Data: D_u (two sites of plate A) are exploited to improve the the initialized teacher by PLePI
- Test 

Author:
Marzieh Haghighi
"""


import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pdb
import time

import numpy as np
import pandas as pd

import barcodefit.model.barcode_calling_KD2 as modellib
from barcodefit.dataobjects import spot, spot_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Table 2. Ablation studies")

    parser.add_argument("--dataset_dir", help="the root dir for the dataset tiff images")
    
    parser.add_argument("--work_dir", help="the dir to save logs and models")

    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--PL_mode", type=str, default="-")

    parser.add_argument("--label_quality", type=str, default="HQ")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        MODEL_DIR = args.work_dir
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        MODEL_DIR = "./experiments/ablation/temp/"
        

    if args.gpu_id is not None:
        which_gpu = args.gpu_id

    label_quality = args.label_quality
    PL_mode = args.PL_mode

    # MODEL_DIR='./experiments/ablation/temp/'

    batch='20210124_6W_CP228';batch_abbrev='CP228'
    plate = "A"
    well = "Well3"

    # label_quality='HQ'

    # PL_mode=['-','PLepi','PLePI']

    # which_gpu="6"
    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    d_inf=[[batch,batch_abbrev],plate,well]

    # unlabled_site_ind = [42, 43]
    # test_sites_ind = [30]
    
    unlabled_site_ls=[12,25]
    test_sites_ls=[70]
    

    ####### read metadata
    # (
    #     dfInfo,
    #     dfInfo_comp,
    #     dataset_train_ls,
    #     dataset_val,
    #     barcode_ref_array,
    # ) = spot_utils.read_metadata(d_inf, args.dataset_dir, "train")

    metadata_dir = "./resource/"
    barcode_ref_list, codebook, barcode_ref_array = spot_utils.read_barcode_list(metadata_dir)
    
    ####### config model
    # batch = d_inf[0]
    config = spot.spotConfig()
    config.batchplate_well=batch_abbrev+d_inf[1]+'_'+d_inf[2]

    config.init_with = "fixed"
    config.assign_label_mode = "clustering"
    config.lr = 10 * config.LEARNING_RATE
    config.LEARNING_MOMENTUM = 0.5

    config.rpn_clustering = True
    config.barcode_ref_array = barcode_ref_array
    
    config.im_Dir=args.dataset_dir+'/'+batch+'/images_aligned_cropped/'
    config.dl_meta_Dir=args.dataset_dir+'/workspace/DL_meta/'+batch+'/'
    

    ##########################
    if label_quality == "LQ":
        config.pretrained_model_path = "./experiments/ablation/init_teacher_by_lq"
        config.create_mask = True
    elif label_quality == "HQ":
        config.pretrained_model_path = "./experiments/ablation/init_teacher_by_hq"
        config.create_mask = False

    ############################ pseudo-labeling using unlabled data
    if PL_mode != "-":
        config.pretrained_model_type = "class"
        config.assign_label_mode = "clustering"
        config.img_aug = True
        config.rpn_clustering = True
        if PL_mode == "PLepi":
            config.BC_MATCH = False
        elif PL_mode == "PLePI":
            config.BC_MATCH = True

        config.GRADIENT_CLIP_NORM = 10
        config.save_seqs = 1
        config.save_seqs2 = 1
        config.tau_g = 1e-04
        config.layers_to_tune = "heads"
        config.STEPS_PER_EPOCH = 484
        config.update_teacher_n_batch = 1
        
        config.list_of_sites=unlabled_site_ls
        config.val_list_of_sites=[61]

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        print(len(dataset_train_ls), len(dataset_val))
        # pdb.set_trace()
        config.display()
        model.train(
            learning_rate=config.lr,
            epochs=len(unlabled_site_ls),
            layers=config.layers_to_tune,
        )

    ########################## applicatin of the model on the test set

    config.BC_MATCH = False
    config.rpn_clustering = True
    config.img_aug = False
    config.init_with == "fixed"
    config.assign_label_mode = "clustering"

    if PL_mode != "-":
        config.pretrained_model_path = model.log_dir + "/final_model"
        config.pretrained_model_type = "clust"

    else:
        config.pretrained_model_type = "class"

    config.STEPS_PER_EPOCH = 484
    config.tau_g = 0
    config.create_mask = False
    config.save_seqs = 1

    config.TRAIN_ROIS_PER_IMAGE = 32 * 3 * 3
    config.RPN_TRAIN_ANCHORS_PER_IMAGE = 32 * 3 * 3
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    
    
    for site_i in test_sites_ls:
        config.list_of_sites=[site_i]
        out = model.evaluate_saved_model(
            learning_rate=config.lr,
            layers="all",
            pretrained_model_path=config.pretrained_model_path,
        )

    ########################## spot_level_to_cell_level_assignments
    matched_flag = ""
    epoch_filter_list = list(range(105))
    #     pdb.set_trace()

    model_direc = model.log_dir
    model_params = [model_direc, epoch_filter_list]

    (
        ngs_match,
        cell_recovery_rate,
        call_dl_df,
    ) = spot_utils.spot_level_to_cell_level_assignments(
        d_inf, test_sites_ls, args.dataset_dir, model_params, matched_flag
    )

    print("ngs_match=", ngs_match)
    print("cell_recovery_rate=", cell_recovery_rate)

    with open("./results/paper_results/table2_results.txt", "a") as f:
        f.write(
            f"PL_mode={PL_mode}, label_quality={label_quality}, ngs_match={ngs_match}, cell_recovery_rate={cell_recovery_rate}\n"
        )


if __name__ == "__main__":
    main()
