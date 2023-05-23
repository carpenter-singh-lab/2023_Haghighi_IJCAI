import copy
import math
import os
import pdb
import pickle
import random
import re
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from skimage.transform import downscale_local_mean, rescale, resize
from skimage.util import img_as_float, img_as_ubyte, img_as_uint

from ..model import model_utils as utils
from ..model.config import Config
from . import spot


###############################################################
def read_metadata(d_inf, mode):

    if len(d_inf) > 3:
        return read_metadata_site(d_inf, mode)

    else:
        return read_metadata_well(d_inf, mode)


###############################################################
def read_metadata_site(d_inf, mode):

    batch, plate, well, site = d_inf[0], d_inf[1], d_inf[2], d_inf[3]

    batch_abbrv = batch.split("_")[-1]
    n_rand_val = 20
    # number of randomly selected samples for validation

    #### directories
    #     dataset_dir='/storage/data/marziehhaghighi/pooledCP/'
    dataset_dir = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"

    #### all data for a batch-well-site information in one dataframe
    if batch_abbrv == "CP074":
        dfInfo3 = pd.read_csv(
            dataset_dir
            + "/workspace/DL_meta/"
            + batch_abbrv
            + "/df-Info-pcp-"
            + batch
            + plate
            + "-"
            + well
            + "-"
            + site
            + "-compact_new.csv"
        )  # cp074
    elif batch_abbrv == "CP228":
        dfInfo3 = pd.read_csv(
            dataset_dir
            + "/workspace/DL_meta/"
            + batch_abbrv
            + "/df-Info-pcp-"
            + batch
            + plate
            + "-"
            + well
            + "-"
            + site
            + "-cp.csv"
        )  # cp228
        #         dfInfo3 = pd.read_csv(dataset_dir+'/workspace/DL_meta/'+batch_abbrv+'/df-Info-pcp-'+batch+plate+'-'+well+'-'+site+'-simple.csv')   #cp228

        #         dfInfo3['im_paths']=dfInfo3['im_paths'].apply(lambda x: x.replace('CP228_','CP228A_'))
        dfInfo3["im_paths"] = dfInfo3["im_paths"].apply(
            lambda x: correct_address(eval(x))
        )

    #     dfInfo3=dfInfo3.rename(columns={"Barcode_BarcodeCalled":"Barcode_BarcodeCalled_simple"})
    #     dfInfo3=dfInfo3.rename(columns={"Barcode_BarcodeCalled_cp":"Barcode_BarcodeCalled"})

    ###### Expand the compact format
    list_ofCycles = []

    # dfInfo42=dfInfo3.copy()
    for i in range(9):
        #         print(i)
        dfInfo42 = dfInfo3.copy()
        dfInfo42["Metadata_Cycle"] = i + 1
        list_ofCycles.append(dfInfo42)

    dfInfo = pd.concat(list_ofCycles).reset_index(drop=True)

    dfInfo["image_id2"] = dfInfo["image_id"]

    dfInfo["cat_id"] = dfInfo.apply(
        lambda x: eval(x["BarcodeList_cat_id"])[x["Metadata_Cycle"] - 1], axis=1
    )
    #     dfInfo["image_id"]=dfInfo.apply(lambda x: x['image_id']+(x['Metadata_Cycle']-1)*323, axis=1)

    dfInfo["image_id"] = np.nan
    uniq_figs = dfInfo.groupby(["image_id2", "Metadata_Cycle"]).size().reset_index()
    for r in range(uniq_figs.shape[0]):
        imi, cy = uniq_figs.loc[r, ["image_id2", "Metadata_Cycle"]].values
        dfInfo.loc[
            (dfInfo["image_id2"] == imi) & (dfInfo["Metadata_Cycle"] == cy), "image_id"
        ] = r

    dfInfo["im_paths"] = dfInfo.apply(
        lambda x: str(
            [
                y.replace("1_", str(x["Metadata_Cycle"]) + "_")
                for y in eval(x["im_paths"])
            ]
        ),
        axis=1,
    )
    #     dct={'1':'A','2':'T','3':'C','4':'G'}

    if batch_abbrv == "CP074":
        dct = {"1": "A", "2": "T", "3": "C", "4": "G"}  # was used for cp074
    elif batch_abbrv == "CP228":
        dct = {"1": "C", "2": "G", "3": "A", "4": "T"}  # was used for cp0228
        map_dict = {"A": 3, "T": 4, "G": 2, "C": 1}

    #     dct={'A':1,'T':2,'C':3,'G':4} #was used for cp074
    #     dct={'C':1,'G':2,'A':3,'T':4} #used for cp228

    dfInfo["Metadata_Label"] = dfInfo["cat_id"].apply(
        lambda x: list(map(dct.get, str(x)))[0]
    )

    # # ###############

    ### read the raw file for the df Info to form test datset for unannotated images by Marzi
    # dfInfo_raw = pd.read_csv(dataset_dir+'PILOT_1_maxproj/scLebelsImgs-'+annotDate+'.csv')
    # dfInfo.loc[(dfInfo['Metadata_Location2']=='na') | (dfInfo['Metadata_Location2']=='?'),'subset_label']="test"

    if mode == "train":
        #         dfInfo = dfInfo[(dfInfo['Metadata_Cycle']==1) | (dfInfo['Metadata_Cycle']==6)]
        #         dfInfo = dfInfo[(dfInfo['image_id2']==1)]
        dfInfo.loc[(dfInfo["image_id2"] == 10), "subset_label"] = "test"
        dfInfo.loc[(dfInfo["image_id2"] == 30), "subset_label"] = "val"

    elif mode == "inference":
        dfInfo["subset_label"] = ""

    if batch_abbrv == "CP074":
        overlay_dir = (
            dataset_dir
            + "workspace/analysis/"
            + batch
            + "/"
            + batch_abbrv
            + plate
            + "_"
            + well
            + "-Site_"
            + site
            + "/Site_"
            + site
            + "_Overlay.png"
        )  # CP074
    elif batch_abbrv == "CP228":
        overlay_dir = (
            dataset_dir
            + "workspace/analysis/"
            + batch
            + "/"
            + batch_abbrv
            + plate
            + "-"
            + well
            + "-"
            + site
            + "/CorrDNA_Site_"
            + site
            + "_Overlay.png"
        )  # CP228

    dfInfo["overlay_dir"] = overlay_dir
    dfInfo["bbox"] = dfInfo["bbox1"]
    dfInfo["Location_Center_X"] = dfInfo["Location_Center_X1"]
    dfInfo["Location_Center_Y"] = dfInfo["Location_Center_Y1"]

    ######### partition dataset to train and validation
    # dfInfo=dfInfo[dfInfo['subset_label']!='test'].reset_index(drop=True)
    testImIds = dfInfo[dfInfo["subset_label"] == "test"]["image_id"].unique().tolist()
    valImIds = dfInfo[dfInfo["subset_label"] == "val"]["image_id"].unique().tolist()
    #     valImIds=list(np.random.choice(dfInfo[dfInfo['subset_label']!='test']['image_id'].unique().tolist(),n_rand_val))
    #     trainImIds=list(set(dfInfo['image_id'].unique().tolist())-set(valImIds)-set(testImIds))
    trainImIds = list(set(dfInfo["image_id"].unique().tolist()) - set(testImIds))

    dfInfo.loc[dfInfo["image_id"].isin(trainImIds), "subset_label"] = "train"
    #     dfInfo.loc[dfInfo['image_id'].isin(valImIds),'subset_label']="val"

    # prepare train dataset
    dataset_train = spot.spotsDataset()
    dataset_train.load_spots(dataset_dir, dfInfo, "train")
    dataset_train.prepare()

    # prepare validation dataset
    dataset_val = spot.spotsDataset()
    dfInfo.loc[dfInfo["image_id"].isin(valImIds), "subset_label"] = "val"
    dataset_val.load_spots(dataset_dir, dfInfo, "val")
    dataset_val.prepare()

    ############ Read reference barcode list
    #     batch='20190919_6W_CP074A'
    # batch='20190922_6W_CP074B'
    metadata_dir = dataset_dir + "workspace/metadata/" + batch + "/"
    metadata_orig = pd.read_csv(metadata_dir + "Barcodes.csv")
    metadata_orig["prefix9"] = metadata_orig["sgRNA"].apply(lambda x: x[0:9])
    barcode_ref_list = metadata_orig.prefix9.tolist()

    #     map_dict={'A':1,'T':2,'G':3,'C':4}

    barcode_ref_array = np.zeros((len(barcode_ref_list), 9))
    barcode_ref_num_list = []
    for ba in range(len(barcode_ref_list)):
        barc = list(barcode_ref_list[ba])
        barcode_ref_num_list.append("".join([str(map_dict[b]) for b in barc]))
        barcode_ref_array[ba, :] = [map_dict[b] for b in barc]

    return dfInfo, dfInfo3, dataset_train, dataset_val, barcode_ref_array


###############################################################
def read_save_metadata_well(d_inf):
    # def read_save_metadata_well(d_inf,mode):
    batch, plate, well = d_inf[0], d_inf[1], d_inf[2]

    batch_abbrv = batch.split("_")[-1]
    #     n_rand_val=20; # number of randomly selected samples for validation

    #### directories
    #     dataset_dir='/home/ubuntu/bucket/projects/2018_11_20_Periscope_Calico/'
    #     dataset_dir='/storage/data/marziehhaghighi/pooledCP/'
    dataset_dir = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"

    #### all data for a batch-well-site information in one dataframe
    #     if batch_abbrv=='CP074':
    #         dfInfo3 = pd.read_csv(dataset_dir+'/workspace/DL_meta/'+batch_abbrv+'/df-Info-pcp-'+batch+plate+'-'+well+'-'+site+'-compact_new.csv') #cp074
    #     elif batch_abbrv=='CP228':

    dl_meta_all = os.listdir(dataset_dir + "/workspace/DL_meta/" + batch_abbrv)
    dl_meta_well = [
        f for f in dl_meta_all if batch + plate + "-" + well in f and "-cp" in f
    ]
    ls_data_train = []
    ls_data_val = []
    #     ls_dfInfo=[]
    for sfn in dl_meta_well:
        dfInfo3 = pd.read_csv(
            dataset_dir + "/workspace/DL_meta/" + batch_abbrv + "/" + sfn
        )
        dfInfo3["im_paths"] = dfInfo3["im_paths"].apply(
            lambda x: correct_address(eval(x))
        )

        #     dfInfo3=dfInfo3.rename(columns={"Barcode_BarcodeCalled":"Barcode_BarcodeCalled_simple"})
        #     dfInfo3=dfInfo3.rename(columns={"Barcode_BarcodeCalled_cp":"Barcode_BarcodeCalled"})

        ###### Expand the compact format
        list_ofCycles = []

        # dfInfo42=dfInfo3.copy()
        for i in range(9):
            #         print(i)
            dfInfo42 = dfInfo3.copy()
            dfInfo42["Metadata_Cycle"] = i + 1
            list_ofCycles.append(dfInfo42)

        dfInfo = pd.concat(list_ofCycles).reset_index(drop=True)

        dfInfo["image_id2"] = dfInfo["image_id"]

        dfInfo["cat_id"] = dfInfo.apply(
            lambda x: eval(x["BarcodeList_cat_id"])[x["Metadata_Cycle"] - 1], axis=1
        )
        #     dfInfo["image_id"]=dfInfo.apply(lambda x: x['image_id']+(x['Metadata_Cycle']-1)*323, axis=1)

        dfInfo["image_id"] = np.nan
        uniq_figs = dfInfo.groupby(["image_id2", "Metadata_Cycle"]).size().reset_index()
        for r in range(uniq_figs.shape[0]):
            imi, cy = uniq_figs.loc[r, ["image_id2", "Metadata_Cycle"]].values
            dfInfo.loc[
                (dfInfo["image_id2"] == imi) & (dfInfo["Metadata_Cycle"] == cy),
                "image_id",
            ] = r

        dfInfo["im_paths"] = dfInfo.apply(
            lambda x: str(
                [
                    y.replace("1_", str(x["Metadata_Cycle"]) + "_")
                    for y in eval(x["im_paths"])
                ]
            ),
            axis=1,
        )
        #     dct={'1':'A','2':'T','3':'C','4':'G'}

        if batch_abbrv == "CP074":
            dct = {"1": "A", "2": "T", "3": "C", "4": "G"}  # was used for cp074
        elif batch_abbrv == "CP228":
            dct = {"1": "C", "2": "G", "3": "A", "4": "T"}  # was used for cp0228
            map_dict = {"A": 3, "T": 4, "G": 2, "C": 1}
            im_orig_size = 5500

        #     dct={'A':1,'T':2,'C':3,'G':4} #was used for cp074
        #     dct={'C':1,'G':2,'A':3,'T':4} #used for cp228

        dfInfo["Metadata_Label"] = dfInfo["cat_id"].apply(
            lambda x: list(map(dct.get, str(x)))[0]
        )

        # # ###############

        if batch_abbrv == "CP074":
            overlay_dir = (
                dataset_dir
                + "workspace/analysis/"
                + batch
                + "/"
                + batch_abbrv
                + plate
                + "_"
                + well
                + "-Site_"
                + site
                + "/Site_"
                + site
                + "_Overlay.png"
            )  # CP074
        elif batch_abbrv == "CP228":
            #             overlay_dir=dataset_dir+"workspace/analysis/"+batch+"/"+batch_abbrv+plate+"-"+well+"-"+site+"/CorrDNA_Site_"+site+"_Overlay.png" # CP228
            dfInfo["overlay_dir"] = (
                dataset_dir
                + "workspace/analysis/"
                + batch
                + "/"
                + batch_abbrv
                + plate
                + "-"
                + well
                + "-"
                + dfInfo["Metadata_Site"].astype(str)
                + "/CorrDNA_Site_"
                + dfInfo["Metadata_Site"].astype(str)
                + "_Overlay.png"
            )  # CP228

        #         dfInfo["overlay_dir"]=overlay_dir
        dfInfo["bbox"] = dfInfo["bbox1"]
        dfInfo["Location_Center_X"] = dfInfo["Location_Center_X1"]
        dfInfo["Location_Center_Y"] = dfInfo["Location_Center_Y1"]

        #     read whole site image
        #         originalURLpath="dataParentFolder/20210124_6W_CP228/images_corrected_cropped/"
        #         toBeReplaced='/home/ubuntu/calbucket/projects/2018_11_20_Periscope_Calico/20210124_6W_CP228/images_aligned_stitched/images_corrected_cropped/'

        originalURLpath = "dataParentFolder"
        #         toBeReplaced='/storage/data/marziehhaghighi/pooledCP/'
        #         toBeReplaced='/data1/data/marziehhaghighi/pooledCP/'
        toBeReplaced = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"

        listOfPaths = eval(dfInfo3.im_paths.unique()[0])

        cropped_im_dim = 256
        cropped_im_dim_h = int(cropped_im_dim / 2)

        site = dfInfo["Metadata_Site"].values[0]
        batchplate_well = batch_abbrv + plate + "_" + well

        #         save_folder='/home/ubuntu/bucket/projects/2018_11_20_Periscope_Calico/workspace/dl_presaved/20210124_6W_CP228/im_cr_presaved/'+batchplate_well+'/'
        #         save_folder='/storage/data/marziehhaghighi/pooledCP/20210124_6W_CP228/im_cr_presaved/'+batchplate_well+'/'

        im_target_size = (int(im_orig_size / cropped_im_dim) + 1) * cropped_im_dim

        if 1:
            image = np.zeros((im_target_size, im_target_size, 4, 9), dtype="uint8")
            for chi in range(len(listOfPaths)):
                #             print("imPath",imPath)
                imPath = listOfPaths[chi].replace(originalURLpath, toBeReplaced)

                for ci in range(9):
                    imPath2 = imPath.replace("Cycle01", "Cycle0" + str(ci + 1))
                    im_uint16 = skimage.io.imread(imPath2)  # images are 'uint16'
                    im_max_2scale = im_uint16.max()
                    im_uint8 = ((im_uint16 / im_max_2scale) * 255).astype("uint8")
                    image[:im_orig_size, :im_orig_size, chi, ci] = im_uint8

        ######## If you want to save each image ID uncomment below line
        if 1:
            save_folder = (
                dataset_dir
                + "/20210124_6W_CP228/im_cr_presaved/"
                + batchplate_well
                + "/"
            )
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            for crimid in np.arange(0, r + 1):
                crop_y_cent, crop_x_cent, cr_cyc = (
                    dfInfo.loc[
                        dfInfo["image_id"] == crimid,
                        ["im_Center_X", "im_Center_Y", "Metadata_Cycle"],
                    ]
                    .astype(int)
                    .values[0]
                )

                cr_br_x_b = crop_x_cent - cropped_im_dim_h
                cr_br_y_b = crop_y_cent - cropped_im_dim_h
                cr_br_x_t = crop_x_cent + cropped_im_dim_h
                cr_br_y_t = crop_y_cent + cropped_im_dim_h

                cr_image = image[
                    cr_br_x_b:cr_br_x_t, cr_br_y_b:cr_br_y_t, :, cr_cyc - 1
                ]

                with open(
                    save_folder + "im_" + str(site) + "_" + str(int(crimid)) + ".npy",
                    "wb",
                ) as f:
                    np.save(f, cr_image)

        ######## If you want to save the whole batch uncomment below line
        if 0:
            save_folder = (
                dataset_dir
                + "/20210124_6W_CP228/im_cr_batch_presaved/"
                + batchplate_well
                + "/"
            )
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            for crimid in np.arange(0, r + 1, 9):
                crop_y_cent, crop_x_cent = (
                    dfInfo.loc[
                        dfInfo["image_id"] == crimid, ["im_Center_X", "im_Center_Y"]
                    ]
                    .astype(int)
                    .values[0]
                )

                cr_br_x_b = crop_x_cent - cropped_im_dim_h
                cr_br_y_b = crop_y_cent - cropped_im_dim_h
                cr_br_x_t = crop_x_cent + cropped_im_dim_h
                cr_br_y_t = crop_y_cent + cropped_im_dim_h

                cr_image = image[cr_br_x_b:cr_br_x_t, cr_br_y_b:cr_br_y_t, :, :]

                with open(
                    save_folder + "im_" + str(site) + "_" + str(int(crimid)) + ".npy",
                    "wb",
                ) as f:
                    np.save(f, cr_image)

        ######### partition dataset to train and validation
        # dfInfo=dfInfo[dfInfo['subset_label']!='test'].reset_index(drop=True)
        #         testImIds=dfInfo[dfInfo['subset_label']=='test']['image_id'].unique().tolist()
        #         valImIds=dfInfo[dfInfo['subset_label']=='val']['image_id'].unique().tolist()
        #     valImIds=list(np.random.choice(dfInfo[dfInfo['subset_label']!='test']['image_id'].unique().tolist(),n_rand_val))
        #     trainImIds=list(set(dfInfo['image_id'].unique().tolist())-set(valImIds)-set(testImIds))
        #         trainImIds=list(set(dfInfo['image_id'].unique().tolist())-set(testImIds))
        trainImIds = set(dfInfo["image_id"].unique().tolist())

        dfInfo.loc[dfInfo["image_id"].isin(trainImIds), "subset_label"] = "train"
        #     dfInfo.loc[dfInfo['image_id'].isin(valImIds),'subset_label']="val"

        # prepare train dataset
        dataset_train = spot.spotsDataset()
        dataset_train.load_spots(dfInfo, "train")
        #         dataset_train.load_spots(dataset_dir,dfInfo,"train")
        dataset_train.prepare()
        ls_data_train.append(dataset_train)

        ###### To save as a single Dataset instead of the list
    #         dfInfo['subset_label']="train"
    #         dfInfo['image_id']=dfInfo['image_id']+id_2add

    #         ls_dfInfo.append(dfInfo)
    #         id_2add=dfInfo.image_id.unique().max()

    #         ls_dfInfo.append(dfInfo)

    #     ###### To save as a single Dataset instead of the list
    #     dataset_train = spot.spotsDataset()
    #     dataset_train.load_spots(pd.concat(ls_dfInfo),"train")
    #     dataset_train.prepare()

    save_do_dir = dataset_dir + "workspace/DL_WellSiteObjects/" + batch_abbrv + "/"

    if not os.path.isdir(save_do_dir):
        os.mkdir(save_do_dir)

    with open(save_do_dir + plate + "_" + well + "_do_t.dat", "wb") as f:
        pickle.dump(ls_data_train, f)

    #     with open(save_do_dir+plate+'_'+well+'_dfInfo.dat', "wb") as f:
    #         pickle.dump(ls_dfInfo, f)

    return


###############################################################


def read_metadata_well(d_inf, mode):
    batch, plate, well = d_inf[0], d_inf[1], d_inf[2]

    batch_abbrv = batch.split("_")[-1]
    #     n_rand_val=20; # number of randomly selected samples for validation

    #### directories
    #     dataset_dir='/storage/data/marziehhaghighi/pooledCP/'
    dataset_dir = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"

    save_do_dir_do = (
        dataset_dir
        + "workspace/DL_WellSiteObjects/"
        + batch_abbrv
        + "/"
        + plate
        + "_"
        + well
        + "_do.dat"
    )

    with open(save_do_dir_do, "rb") as f:
        ls_data_train = pickle.load(f)

    metadata_dir = "./resource/"
    barcode_ref_list, codebook, barcode_ref_array = read_barcode_list(metadata_dir)

    #     return [],[],ls_data_train, ls_data_val,barcode_ref_array
    if mode == "train":
        return [], [], ls_data_train, ls_data_train[60:62], barcode_ref_array

    else:
        save_do_dir_dfInfo = (
            dataset_dir
            + "workspace/DL_WellSiteObjects/"
            + batch_abbrv
            + "/"
            + plate
            + "_"
            + well
            + "_dfInfo.dat"
        )
        with open(save_do_dir_dfInfo, "rb") as f:
            ls_dfInfo = pickle.load(f)

        return [], ls_dfInfo, ls_data_train, [], barcode_ref_array


def config_model(d_inf):

    batch = d_inf[0]
    save_model_subdir = "_".join(d_inf[1:])

    config = spot.spotConfig()
    config.batchplate_well = d_inf[0].split("_")[-1] + d_inf[1] + "_" + d_inf[2]
    ####################### Model Parameters
    # use of pretrained models --> "imagenet","coco","scratch","last","fixed"
    config.init_with = "fixed"

    # layers to train --> "all" , "heads"
    config.layers_to_tune = "all"

    # assign_label_mode --> "clustering","classification"
    config.assign_label_mode = "clustering"

    # learning rate
    #     config.lr=10*config.LEARNING_RATE
    config.lr = 10 * config.LEARNING_RATE
    config.LEARNING_MOMENTUM = 0.5
    config.display()
    #     print("lr",lr)

    #     MODEL_Root_DIR ="/storage/data/marziehhaghighi/DL_trained_models/mrcnn/"
    MODEL_Root_DIR = "/dgx1nas1/storage/data/marziehhaghighi/DL_trained_models/mrcnn/"

    ###########################
    #     config.project_name=project_name
    #     config.head=maskBranchNet;
    #     config.assign_label_mode =assign_label_mode#"clustering","classification"
    # fileNumModel=assign_label_mode+'_init_with_'+init_with+'_maskNet_'+maskBranchNet+'_train_'+layers_to_tune;
    # MODEL_DIR = os.path.join(model_dir+fileNumModel+"/")

    fileNumModel = (
        "pretrained_"
        + config.init_with
        + "_train_"
        + config.layers_to_tune
        + "_lr_"
        + str(config.lr)
        + "_"
        + config.assign_label_mode[0:5]
    )

    MODEL_DIR = (
        MODEL_Root_DIR + config.NAME + "/" + save_model_subdir + "/" + fileNumModel
    )
    return MODEL_DIR, config


###############################################################


def check_saved_model_config_params(model_direc, config_params_ls):
    pkl_file = open(model_direc + "/config.pkl", "rb")
    inference_config_loaded = pickle.load(pkl_file)
    for p in config_params_ls:
        print(p, inference_config_loaded[p])
    return


###############################################################


def train_model(config, MODEL_DIR, dataset_train, dataset_val, which_gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu

    import barcodefit.model.barcode_calling as modellib

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    #     MODEL_Root_DIR ="/storage/data/marziehhaghighi/DL_trained_models/mrcnn/"
    MODEL_Root_DIR = "/dgx1nas1/storage/data/marziehhaghighi/DL_trained_models/mrcnn/"
    #         model.load_weights(model_fixed, by_name=True)

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.lr,
        epochs=config.epochs,
        layers=config.layers_to_tune,
    )

    return


###############################################################
def correct_address(x_list):
    y_list = []
    for x in x_list:
        y = x.split("/")  # .replace('Site','sdfdsfs')
        #         del y[-4]
        y[-1] = y[-2] + "_" + y[-1]
        y_list.append("/".join(y))
    return str(y_list)


###############################################################
def read_results_to_df(training_results_file, epoch_filter_list):
    """
    Read saved barcodes during training for the a given txt file corresponding to
    a cropped image to a pandas df and compute x,y cordinates of each barcode

    """

    cols = [
        "epoch",
        "y1",
        "x1",
        "y2",
        "x2",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "cel",
        "acc",
        "nmi",
    ]
    # df = pd.read_csv(address,index_col=None,sep=",")
    res_arr = np.loadtxt(
        training_results_file,
        skiprows=0,
        delimiter=",",
        dtype={
            "names": tuple(cols),
            "formats": (
                "i4",
                "f4",
                "f4",
                "f4",
                "f4",
                "i4",
                "i4",
                "i4",
                "i4",
                "i4",
                "i4",
                "i4",
                "i4",
                "i4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
                "f4",
            ),
        },
    )
    res_df0 = pd.DataFrame(res_arr, columns=cols)

    res_df0[["0", "1", "2", "3", "4", "5", "6", "7", "8"]] = res_df0[
        ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    ].astype(int)

    res_df = res_df0[res_df0["epoch"].isin(epoch_filter_list)].reset_index(drop=True)

    # print('here:',image.shape, molded_images[i].shape,windows[i])
    # (256, 256, 4) (256, 256, 4)
    #     unq_epochs=res_df.epoch.unique()
    #     res_df=res_df[res_df['epoch']==unq_epochs[0]].reset_index(drop=True)

    boxes = res_df[["y1", "x1", "y2", "x2"]].values

    image_shape = (256, 256, 4)
    original_image_shape = (256, 256, 4)
    window = [0, 0, 256, 256]
    window = utils.norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

    res_df[["y1_", "x1_", "y2_", "x2_"]] = boxes

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0
    )[0]

    exclude_ix_big_area = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 20
    )[0]

    # res_df[res_df]
    update_df = res_df.drop(exclude_ix)

    # update_df2 = update_df.drop(exclude_ix_big_area)
    #     update_df

    # if exclude_ix.shape[0] > 0:
    #     boxes = np.delete(boxes, exclude_ix, axis=0)
    #     class_probs = np.delete(class_probs, exclude_ix, axis=0)
    #     class_ids = np.delete(class_ids, exclude_ix, axis=0)
    #     scores = np.delete(scores, exclude_ix, axis=0)
    #     masks = np.delete(masks, exclude_ix, axis=0)
    #     N = class_ids.shape[0]

    #     height = box[:, 2] - box[:, 0]
    height = res_df["y2_"] - res_df["y1_"]
    width = res_df["x2_"] - res_df["x1_"]
    #     width = box[:, 3] - box[:, 1]
    update_df["bb_center_y"] = res_df["y1_"] + 0.5 * height
    update_df["bb_center_x"] = res_df["x1_"] + 0.5 * width

    return update_df


# #######################################################
def map_barcodes_to_cells_by_overlays_whole_site(
    ds_site, d_inf, dataset_dir, model_params, barcode_ref_array, matched_flag
):
    """
    Assigns barcodes to cells

    steps:
    - Read Overlay.png and Nuclei.csv files for the whole site
        The for each cropped image:
        - Form address for each image id barcodes detected
        - Read saved barcodes and locations through read_results_to_df call
        - Read the overlay to assign barcodes to cells through masks and detected centers

    """

    batch, plate, well = d_inf[0], d_inf[1], d_inf[2]
    batch_abbrv = batch.split("_")[-1]

    cropped_im_dim = 256
    orig_im_w = 5500
    cropped_im_dim_h = int(cropped_im_dim / 2)

    image_ids_inside_site = ds_site.image_ids
    #     img_site=int(ds_site.image_info[image_ids_inside_site[0]]["overlay_dir"].split('_')[-2])
    img_site = int(ds_site.image_info[image_ids_inside_site[0]]["site"])
    print("img_site", img_site)

    overlay_dir = (
        dataset_dir
        + "workspace/analysis/"
        + batch
        + "/"
        + batch_abbrv
        + plate
        + "-"
        + well
        + "-"
        + str(img_site)
        + "/CorrDNA_Site_"
        + str(img_site)
        + "_Overlay.png"
    )
    ov_im = resize(
        skimage.io.imread(overlay_dir),
        (orig_im_w, orig_im_w),
        mode="constant",
        preserve_range=True,
        order=0,
    ).astype("uint8")

    nucl_csv = pd.read_csv(
        dataset_dir
        + "workspace/analysis/"
        + batch
        + "/"
        + batch_abbrv
        + plate
        + "-"
        + well
        + "-"
        + str(img_site)
        + "/Nuclei.csv"
    )
    # print('nucl_csv',nucl_csv.shape,nucl_csv[nucl_csv['ObjectNumber']!=0].shape)
    ############ Read reference barcode list
    #     metadata_dir=dataset_dir+'workspace/metadata/'+batch+'/'
    #     metadata_orig= pd.read_csv(metadata_dir+'Barcodes.csv')
    #     metadata_orig['prefix9']=metadata_orig["sgRNA"].apply(lambda x: x[0:9])
    #     barcode_ref_list=metadata_orig.prefix9.tolist()

    bins = np.array(range(128, 5633, 256))
    single_feature_vals_discrete_x = np.digitize(
        nucl_csv.Location_Center_X.values, range(256, 5633, 256), right=True
    )
    single_feature_vals_discrete_y = np.digitize(
        nucl_csv.Location_Center_Y.values, range(256, 5633, 256), right=True
    )

    nucl_csv["im_Center_X"] = [bins[s] for s in single_feature_vals_discrete_x]
    nucl_csv["im_Center_Y"] = [bins[s] for s in single_feature_vals_discrete_y]
    nucl_csv["Nuclei_Location_Center_X1"] = (
        nucl_csv["Location_Center_X"] - nucl_csv["im_Center_X"] + 128
    )
    nucl_csv["Nuclei_Location_Center_Y1"] = (
        nucl_csv["Location_Center_Y"] - nucl_csv["im_Center_Y"] + 128
    )

    nucl_csv = nucl_csv.rename(
        columns={
            "ObjectNumber": "Parent_Cells",
            "Location_Center_X": "Nuclei_Location_Center_X",
            "Location_Center_Y": "Nuclei_Location_Center_Y",
        }
    )

    nucl_csv.loc[
        nucl_csv["Nuclei_Location_Center_Y1"] > 255, "Nuclei_Location_Center_Y1"
    ] = 255
    nucl_csv.loc[
        nucl_csv["Nuclei_Location_Center_X1"] > 255, "Nuclei_Location_Center_X1"
    ] = 255
    #     dfInfo_site=dfInfo_comp[site_idx]

    cells_df_site = nucl_csv[
        [
            "Parent_Cells",
            "Nuclei_Location_Center_X1",
            "Nuclei_Location_Center_Y1",
            "Nuclei_Location_Center_X",
            "Nuclei_Location_Center_Y",
            "im_Center_X",
            "im_Center_Y",
        ]
    ].astype(int)

    model_direc, epoch_filter_list = model_params

    site_results_list = []
    saved_image_ids_inside_site = image_ids_inside_site[0::9]
    for cr_im in saved_image_ids_inside_site:

        training_results_file = (
            model_direc
            + "/seqs"
            + matched_flag
            + "/im_id_"
            + str(img_site)
            + "_"
            + str(cr_im)
            + ".txt"
        )
        if os.path.exists(training_results_file):
            ### Read results saved during training
            update_df = read_results_to_df(training_results_file, epoch_filter_list)
            #         update_df=update_df[update_df['epoch']==52].reset_index(drop=True)

            if 1:
                #                 find the closest reflin barcode
                #                 map_dict={'A':3,'T':4,'G':2,'C':1}
                update_df["dist_2_ref"] = update_df.apply(
                    lambda x: utils.map_to_barcode_min_Hdist(
                        [
                            int(x["0"]),
                            int(x["1"]),
                            int(x["2"]),
                            int(x["3"]),
                            int(x["4"]),
                            int(x["5"]),
                            int(x["6"]),
                            int(x["7"]),
                            int(x["8"]),
                        ],
                        barcode_ref_array,
                    )[1],
                    axis=1,
                )
            ### Read the overlay to group the results for cell calling
            crop_y_cent, crop_x_cent = (
                round(ds_site.image_info[cr_im]["im_Center_X"]),
                round(ds_site.image_info[cr_im]["im_Center_Y"]),
            )
            ovly = np.zeros((cropped_im_dim, cropped_im_dim, 3))

            cr_br_x_b = crop_x_cent - cropped_im_dim_h
            cr_br_y_b = crop_y_cent - cropped_im_dim_h
            cr_br_x_t = np.min([crop_x_cent + cropped_im_dim_h, orig_im_w])
            cr_br_y_t = np.min([crop_y_cent + cropped_im_dim_h, orig_im_w])

            ovly[: (cr_br_x_t - cr_br_x_b), : (cr_br_y_t - cr_br_y_b), :] = ov_im[
                cr_br_x_b:cr_br_x_t, cr_br_y_b:cr_br_y_t, :
            ]
            #         ovly = ds_site.load_overlay(int(cr_im))

            #             dfInfo_site_cr=dfInfo_site[dfInfo_site['image_id']==cr_im]

            ###

            cells_df = cells_df_site[
                (cells_df_site["im_Center_X"] == crop_y_cent)
                & (cells_df_site["im_Center_Y"] == crop_x_cent)
            ].reset_index(drop=True)
            #             hjdsf
            #             cells_df = dfInfo_site_cr.groupby(["Parent_Cells","Nuclei_Location_Center_X1","Nuclei_Location_Center_Y1",\
            #                         "Nuclei_Location_Center_X","Nuclei_Location_Center_Y",\
            #                        "im_Center_X","im_Center_Y"]).size().reset_index().astype(int)

            #             cells_df.loc[cells_df['Nuclei_Location_Center_Y1']<0,'Nuclei_Location_Center_Y1']=0
            #             cells_df.loc[cells_df['Nuclei_Location_Center_X1']<0,'Nuclei_Location_Center_X1']=0
            #             cells_df.loc[cells_df['Nuclei_Location_Center_Y1']>255,'Nuclei_Location_Center_Y1']=255
            #             cells_df.loc[cells_df['Nuclei_Location_Center_X1']>255,'Nuclei_Location_Center_X1']=255

            from skimage.segmentation import flood_fill

            cell_color = (255, 255, 255)
            cell_bound = np.copy(ovly)
            indices_not_w = np.where(~np.all(cell_bound == cell_color, axis=-1))
            cell_bound[indices_not_w] = 0
            colored_cells = cell_bound[:, :, 0].astype(int)
            parent_cells = cells_df.Parent_Cells.unique().tolist()
            for p in parent_cells:
                cent_x, cent_y = cells_df.loc[
                    cells_df["Parent_Cells"] == p,
                    ["Nuclei_Location_Center_X1", "Nuclei_Location_Center_Y1"],
                ].values[0]
                colored_cells = flood_fill(
                    colored_cells, (cent_y, cent_x), p, connectivity=1
                )

            dct = {1: "C", 2: "G", 3: "A", 4: "T"}  # was used for cp0228
            map_dict = {"A": 3, "T": 4, "G": 2, "C": 1}

            for c in range(9):
                update_df = update_df.replace({str(c): dct})

            update_df["Barcodes_called_dl"] = update_df[
                ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            ].sum(axis=1)
            update_df["Barcodes_called_sumP"] = update_df[
                ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            ].sum(axis=1)
            update_df["Barcodes_called_medP"] = update_df[
                ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            ].median(axis=1)
            update_df["Barcodes_called_prodP"] = update_df[
                ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
            ].product(axis=1)

            bb_ys = update_df["bb_center_y"].astype(int).values
            bb_xs = update_df["bb_center_x"].astype(int).values

            update_df["Parent_Cells"] = colored_cells[bb_ys, bb_xs]
            update_df["im_Center_X"] = crop_y_cent
            update_df["im_Center_Y"] = crop_x_cent

            #             print("Number of cells with barcode using CP: ",dfInfo_site_cr.Parent_Cells.unique().shape[0])
            #             print("Number of cells with barcode using DL: ",update_df.Parent_Cells.unique().shape[0])

            #         ### Spot calling for the croped image
            #         crop_pp=(dfInfo_site_cr[dfInfo_site_cr["Barcode_BarcodeCalled"].isin(barcode_ref_list)].shape[0])/dfInfo_site_cr.shape[0]
            #         print("crop_spot_calling_cp: ",crop_pp)

            #         crop_pp=(update_df[update_df["Barcodes_called_dl"].isin(barcode_ref_list)].shape[0])/update_df.shape[0]
            #         print("crop_spot_calling_dl: ",crop_pp)

            update_df["Metadata_Site"] = img_site
            update_df["image_id"] = cr_im
            #             update_df=map_barcodes_to_cells_by_overlays(img_site, cr_im, ds_site, model_direc);
            #             if update_df:
            site_results_list.append(update_df)

    #         else:
    #             print("Warning! "+'im_id_'+str(img_site)+'_'+str(cr_im)+'.txt doesnt exist!')

    return site_results_list, nucl_csv.shape[0]


import skimage.io
from skimage.transform import downscale_local_mean, rescale, resize
from skimage.util import img_as_float, img_as_ubyte, img_as_uint


def read_ngs_counts_4target_well(ngs_csv_file, well):
    samples_to_keep = [
        "d4 +dox 1",
        "d4 +dox 2",
        "d4 +dox 3",
        "d4 +dox 4",
        "d7 +dox 1",
        "d7 +dox 2",
        "d7 +dox 3",
        "d7 +dox 4",
    ]
    ngs_df_0 = pd.read_csv(ngs_csv_file)
    ngs_df = ngs_df_0[ngs_df_0["Sample"].isin(samples_to_keep)].reset_index(drop=True)
    ngs_df["Day"] = ngs_df["Sample"].apply(lambda x: x.split(" ")[0])

    ngs_counts_all = (
        ngs_df.groupby(["Barcode", "Day"])["Fraction_of_Reads"].mean().reset_index()
    )

    if well in ["Well1", "Well2", "Well3"]:
        day = "d4"
    elif well in ["Well4", "Well5", "Well6"]:
        day = "d7"

    ngs_counts = ngs_counts_all[ngs_counts_all["Day"] == day].reset_index(drop=True)
    ngs_counts["prefix9"] = ngs_counts["Barcode"].apply(lambda x: x[0:9])
    return ngs_counts


def read_resize_overlay_pooled(overlay_dir, orig_im_w):
    """
    This function read and resizes overlay to the size of the original image
    """

    import skimage.io
    from skimage.transform import downscale_local_mean, rescale, resize

    ov_im = resize(
        skimage.io.imread(overlay_dir),
        (orig_im_w, orig_im_w),
        mode="constant",
        preserve_range=True,
        order=0,
    ).astype(
        "int"
    )  # .astype('uint8')
    return img_as_ubyte(ov_im)


#     return ov_im


def read_barcode_list(metadata_dir):

    # #     map_dict={'A':1,'T':2,'G':3,'C':4}
    #     if batch_abbrv=='CP074':
    #         dct={'1':'A','2':'T','3':'C','4':'G'} #was used for cp074
    #     elif batch_abbrv=='CP228':
    dct = {"1": "C", "2": "G", "3": "A", "4": "T"}  # was used for cp0228
    map_dict = {"A": 3, "T": 4, "G": 2, "C": 1}

    metadata_orig = pd.read_csv(metadata_dir + "CP228_Experimental_Codebook.csv")
    metadata_orig["prefix9"] = metadata_orig["sgRNA"].apply(lambda x: x[0:9])
    barcode_ref_list = metadata_orig.prefix9.tolist()

    codebook_input = pd.read_csv(metadata_dir + "Codebook.csv")[
        "barcode_bases"
    ].tolist()

    barcode_ref_array = np.zeros((len(codebook_input), 9))
    barcode_ref_num_list = []
    for ba in range(len(codebook_input)):
        barc = list(codebook_input[ba])
        barcode_ref_num_list.append("".join([str(map_dict[b]) for b in barc]))
        barcode_ref_array[ba, :] = [map_dict[b] for b in barc]

    return barcode_ref_list, codebook_input, barcode_ref_array


def spot_level_to_cell_level_assignments(
    dataset_train_ls, d_inf, sites_ind, model_params, matched_flag
):

    well = d_inf[2]

    ngs_csv_file = "./resource/CP228_NGS_Reads_And_Library_Mapped.csv"
    ngs_counts = read_ngs_counts_4target_well(ngs_csv_file, well)

    metadata_dir = "./resource/"
    barcode_ref_list, codebook, barcode_ref_array = read_barcode_list(metadata_dir)

    well_results_list = []
    # sites_ind=[30,31]
    #     sites_ind=list(range(len(dataset_train_ls)))
    dataset_dir = "/dgx1nas1/cellpainting-datasets/2018_11_20_Periscope_Calico/"
    # pdb.set_trace()
    for site_idx in sites_ind:

        start = time.time()
        ds_site = dataset_train_ls[site_idx]
        print("site:", site_idx)
        site_results_list, cell_count = map_barcodes_to_cells_by_overlays_whole_site(
            ds_site,
            d_inf,
            dataset_dir,
            model_params,
            barcode_ref_array.astype(int),
            matched_flag,
        )

        # site_results_list, cell_count = spot_utils.map_barcodes_to_cells_by_overlays_whole_site(ds_site, d_inf, dataset_dir,model_params,barcode_ref_array.astype(int),matched_flag)
        well_results_list += site_results_list
        print("time elapsed for the site cell calling: ", (time.time() - start) / 60)

    #         pdb.set_trace()
    #         if site_results_list:
    #         all_results_df = pd.concat(site_results_list, axis=0)
    all_results_df = pd.concat(well_results_list, axis=0)

    #     all_results_df=all_results_df[all_results_df['Barcodes_called_dl'].isin(barcode_ref_list)].reset_index(drop=True)

    if 1:
        # Assign highest probable barcode to each cell
        cells_called_perWell = (
            all_results_df.sort_values(["Barcodes_called_prodP"], ascending=False)
            .groupby(["Metadata_Site", "Parent_Cells"])
            .head(1)
            .reset_index()
            .sort_values(by=["Metadata_Site", "Parent_Cells"])
        )

    else:

        ## If SSL without constraint, uncomment the below line
        # Assign closest to reflib barcode to each cell
        cells_called_perWell = (
            all_results_df.sort_values(["dist_2_ref"], ascending=True)
            .groupby(["Metadata_Site", "Parent_Cells"])
            .head(1)
            .reset_index()
            .sort_values(by=["Metadata_Site", "Parent_Cells"])
        )

    call_dl_df = cells_called_perWell[
        cells_called_perWell["Parent_Cells"] != 0
    ].reset_index(drop=True)
    call_dl_df = call_dl_df[
        call_dl_df["Barcodes_called_dl"].isin(barcode_ref_list)
    ].reset_index(drop=True)
    call_dl_df["y"] = call_dl_df["bb_center_y"] + call_dl_df["im_Center_Y"] - 128
    call_dl_df["x"] = call_dl_df["bb_center_x"] + call_dl_df["im_Center_X"] - 128

    dl_counts = (
        call_dl_df.groupby(["Barcodes_called_dl"]).size() / call_dl_df.shape[0]
    ).reset_index()

    merged_ngs_dl = ngs_counts.merge(
        dl_counts, left_on="prefix9", right_on="Barcodes_called_dl"
    ).rename(columns={0: "Fracs"})

    ngs_match = np.round(
        merged_ngs_dl["Fracs"].corr(merged_ngs_dl["Fraction_of_Reads"]), 2
    )
    # print(site_idx,call_dl_df.shape[0])

    # print(which_gpu, config.pretrained_model_path)
    #     pdb.set_trace()
    #     print(np.round(merged_ngs_dl['Fracs'].corr(merged_ngs_dl['Fraction_of_Reads']),2),call_dl_df.shape[0])
    cell_recovery_rate = np.round(call_dl_df.shape[0] / cell_count, 2)

    return ngs_match, cell_recovery_rate, call_dl_df


def colormask_cells_to_parent_cell_number(ov_im, cell_centers_df):
    """
    given an input overlay image, and a mapping of cell centers to object numbers
        -> color each cell to its object number

    Inputs:
       ov_im: overlay image, is an RGB image with outline of nuclei and cells in the first channel
       cell_centers_df: a dataframe containing cell_centers and map to object numbers
               required columns in the df:
                  - Parent_Cells
                  - Nuclei_Location_Center_X , Nuclei_Location_Center_Y

    """

    from skimage.segmentation import flood_fill

    if ov_im.dtype == np.uint8:
        cell_color = (255, 255, 255)
    elif ov_im.dtype == np.uint16:
        cell_color = (65535, 65535, 65535)

    print(cell_color)
    cell_bound = np.copy(ov_im)
    indices_not_w = np.where(~np.all(cell_bound == cell_color, axis=-1))
    cell_bound[indices_not_w] = 0

    colored_cells = cell_bound[:, :, 0].astype(int)

    parent_cells = cell_centers_df.Parent_Cells.unique().tolist()
    for p in parent_cells:
        cent_x, cent_y = cell_centers_df.loc[
            cell_centers_df["Parent_Cells"] == p,
            ["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"],
        ].values[0]
        colored_cells = flood_fill(colored_cells, (cent_y, cent_x), p, connectivity=1)

    return colored_cells


def create_mask_batch(images4D, thrsh_perc=98.8):
    """Take input ISS multi-channel multi-cycle image and
        - max project across channels and cycles
        - then by thresholding detects and lables spots
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    from skimage.draw import circle, disk

    # max_proj_im=np.percentile(images3D,(25),axis=-1)
    #         max_proj_im=np.max(np.percentile(images4D,90,axis=-1),axis=2)
    #     max_proj_im=np.max(np.percentile(images4D,90,axis=2),axis=-1)
    max_proj_im = np.percentile(np.max(images4D, axis=3), 90, axis=0)
    #         max_proj_im=np.max(np.median(images4D,axis=-1),axis=2)

    # max_proj_im=np.max(images3D,axis=-1)

    #         thrsh=np.percentile(max_proj_im, (98.5))
    thrsh = np.percentile(max_proj_im, (thrsh_perc))
    index_spots = np.where(max_proj_im > thrsh)

    # dfInfoo=pd.DataFrame(index=range(index_spots[0].shape[0]),columns=['Location_Center_X', 'Location_Center_Y',\
    #                             "C_int","G_int","A_int","T_int","Metadata_Label"])

    # dfInfoo['Location_Center_X']=index_spots[1]
    # dfInfoo['Location_Center_Y']=index_spots[0]
    n_spots = index_spots[0].shape[0]

    #         print('n_spots',n_spots)
    if n_spots > 900:
        thrsh = np.percentile(max_proj_im, (99))
        index_spots = np.where(max_proj_im > thrsh)
        n_spots = index_spots[0].shape[0]
        print("n_spots2", n_spots)

    if n_spots > 0:

        armaxx = np.argmax(images4D[index_spots[0], index_spots[1], :, :], axis=-2)
        # dfInfoo['Metadata_Label']=armaxx
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        instance_masks = []
        class_ids = []

        cropped_im_dim = 256
        cropped_im_dim_h = int(cropped_im_dim / 2)
        bbox_half_len = 2
        for ri in range(n_spots):

            #     center_x,center_y=dfInfoo.loc[ri,['Location_Center_X','Location_Center_Y']].values

            center_x, center_y = index_spots[1][ri], index_spots[0][ri]
            OneHot2D_instance_mask = np.zeros(
                (cropped_im_dim, cropped_im_dim), dtype=np.uint8
            )
            rr, cc = disk((center_x, center_y), bbox_half_len * 2)

            rr[rr > cropped_im_dim - 1] = cropped_im_dim - 1
            cc[cc > cropped_im_dim - 1] = cropped_im_dim - 1

            rr[rr < 0] = 0
            cc[cc < 0] = 0

            OneHot2D_instance_mask[cc, rr] = 1

            if ri > 0:
                #         c = np.logical_and(OneHot2D_instance_mask, instance_masks[-1])
                c = np.logical_and(
                    OneHot2D_instance_mask,
                    np.sum(np.stack(instance_masks, axis=2), axis=-1),
                )
                if np.sum(c) == 0:
                    instance_masks.append(OneHot2D_instance_mask)
                    class_ids.append(list(armaxx[ri, :] + 1))
            #             class_ids2.append(dfInfoo.loc[ri,['Metadata_Label']].values[0]);
            else:
                instance_masks.append(OneHot2D_instance_mask)
                class_ids.append(list(armaxx[ri, :] + 1))

        # channels1=["C","G","A","T"]

        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids_batch = np.array(class_ids, dtype=np.int32)
    else:
        return [], []

    # plt.figure()
    # plt.imshow(np.max(mask,axis=-1))
    return mask, class_ids_batch
