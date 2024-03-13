# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from contextlib import ExitStack, contextmanager
import copy
import numpy as np
import torch
import random
import torchvision
from collections import defaultdict
from torchvision import transforms
from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size        # utils.comm -> primitives for multi-gpu communication
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor
from PIL import Image
import matplotlib.pyplot as plt
from inference_core import InferenceCore
from pathlib import Path
import math

def evaluate(
    model, propagation_model, fusion_model,
    data_loader, iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "worst", seed_id = 0, vis_path = None, max_frame_interactions=0
):
    """
    Run model on the data_loader and return a dict, later used to calculate all the metrics for multi-instance inteactive segmentation such as NCI,
    NFO, NFI, and Avg IoU. The model will be used in eval mode.

    Arguments:
        model (callable): a callable which takes an object from `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length. The elements it generates will be the inputs to the model.
        iou_threshold: float - Desired IoU value for each object mask
        max_interactions: int - Maxinum number of interactions per object
        sampling_strategy: int - Strategy to avaoid regions while sampling next clicks
            0: new click sampling avoids all the previously sampled click locations
            1: new click sampling avoids all locations upto radius 5 around all the previously sampled click locations
        eval_strategy: str - Click sampling strategy during refinement (random, best, worst)
        seed_id: int - Used to generate fixed seed during evaluation
        vis_path: str - Path to save visualization of masks with clicks during evaluation

    Returns:
        Dict
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))                       # 1999 (davis_2017_val)
    logger.info(f"Using {eval_strategy} evaluation strategy with random seed {seed_id}")
    
    print(f'[INFO] Loading all frames and ground truth masks from disc...')
    all_images = load_images()
    all_gt_masks = load_gt_masks()
    
    all_ious = {}
    all_interactions = {}
    all_interactions_per_instance = {}
    all_rounds = {}
    all_iou_checkpoints = defaultdict(list)


    global_iou_threshold = max(0.85, iou_threshold)                           # set IoU to 0.99
    #threshold_list = [0.85, 0.90, 0.925, 0.95, 0.99]
    if global_iou_threshold > 0.85:
        step = (global_iou_threshold - 0.85)/5
        threshold_list = list(np.arange(0.85, global_iou_threshold, step))
        threshold_list.append(global_iou_threshold)
    else:
        threshold_list = [0.85]
        
    iou_checkpoints = copy.deepcopy(threshold_list)
    max_interactions = 10                                
    max_frame_interactions = 50                          # only interact with 1st frame
    
    with ExitStack() as stack:                                           

        if isinstance(model, nn.Module):    
            stack.enter_context(inference_context(model))                           
        stack.enter_context(torch.no_grad())                             

        total_num_instances = 0                                          # in the whole dataset                               
        total_num_interactions = 0                                       # for whole dataset        

        random.seed(123456+seed_id)
        
        dataloader_dict = defaultdict(list)
        print(f'[INFO] Iterating through the Data Loader...')
        # iterate through the data_loader, one image at a time
        for idx, inputs in enumerate(data_loader):            
            curr_seq_name = inputs[0]["file_name"].split('/')[-2]
            dataloader_dict[curr_seq_name].append([idx, inputs])

        print(f'[INFO] Sequence-wise evaluation...')
        for seq in list(dataloader_dict.keys()):
            print(f'\n[INFO] Sequence: {seq}')
            
            # Initialize propagation module - per-sequence
            seq_object_ids = set(np.unique(all_gt_masks[seq][0]))
            seq_num_instances = len(seq_object_ids) - 1
            all_frames = all_images[seq]
            num_frames = len(all_frames)
            all_frames = all_frames.unsqueeze(0).float()
            processor = InferenceCore(propagation_model, fusion_model, all_frames, seq_num_instances)
            
            lowest_frame_index = 0               # frame with lowest IoU after propagation
            lowest_instance_index = None         # instance with lowest IoU after propagation
            round_num = 0
            interacted_frames = []         
            clicker_dict = {}
            predictor_dict = {}
            checkpoints_for_sequence = []
            iou_for_sequence = [0]*num_frames
            num_interactions_for_sequence = [0]*num_frames
            num_interactions_per_instance = [[]] * num_frames
            out_masks = None

            if max_frame_interactions==0:
                frame_limit_bool = False
            else:
                frame_limit_bool = True
                #frame_interaction_limit = math.round( num_frames * (max_frame_interactions / 100))
                frame_interaction_limit = max_frame_interactions

            while lowest_frame_index!=-1:
                round_num += 1
                
                print(f'[INFO] DynaMITe refining frame {lowest_frame_index} of sequence {seq}')
                idx, inputs = dataloader_dict[seq][lowest_frame_index]
                
                # objects present in this frame
                object_ids = set(np.unique(all_gt_masks[seq][lowest_frame_index]))
                                
                if lowest_frame_index not in interacted_frames:    
                    clicker = Clicker(inputs, sampling_strategy)
                    predictor = Predictor(model)
                    repeat = False
                else:                                                  
                    clicker = clicker_dict[lowest_frame_index]
                    predictor = predictor_dict[lowest_frame_index]
                    repeat = True
                

                # convert predicted masks from previous round into instance-wise masks
                num_instances = clicker.num_instances
                missing_obj_ids = None
                if num_instances!=seq_num_instances:
                    #check for missing object
                    missing_obj_ids = seq_object_ids - object_ids

                if out_masks is not None:
                    mask_H,mask_W = out_masks[lowest_frame_index].shape
                    pred_masks = np.zeros((seq_num_instances,mask_H,mask_W))                    
                    for i in range(seq_num_instances):
                        if missing_obj_ids is not None:
                            if i+1 in missing_obj_ids:
                                continue
                        pred_masks[i][np.where(out_masks[lowest_frame_index]==i+1)] = 1          
                    pred_masks = torch.from_numpy(pred_masks)
                    clicker.set_pred_masks(pred_masks)   
                else:
                    pred_masks = predictor.get_prediction(clicker)
                    clicker.set_pred_masks(pred_masks)
                
                ious = clicker.compute_iou()                                                                # instance-wise IoU                                                      
                if vis_path:
                    clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_frame_index], round_num=round_num)             

                if not repeat:
                    total_num_instances+=num_instances
                    num_interactions = num_instances
                    total_num_interactions+=(num_instances)
                    num_interactions_for_sequence[lowest_frame_index] += num_instances
                    num_interactions_per_instance[lowest_frame_index] = [1]*(num_instances+1)      
                    num_interactions_per_instance[lowest_frame_index][-1] = 0                                                               # no interaction for bg yet, so reset                    
                
                if repeat:
                    #num_interactions = num_interactions_for_sequence[lowest_frame_index]
                    num_interactions = 0         # reset for next round

                max_iters_for_image = max_interactions * num_instances                                      

                point_sampled = True
                random_indexes = list(range(len(ious)))

                #interative refinement loop
                while (num_interactions<max_iters_for_image):
                    if all(iou >= threshold_list[0] for iou in ious):
                        t = threshold_list.pop(0)
                        print(f'[INFO] DynaMITe Refinement - all instances have IoU >= {t} in frame {lowest_frame_index}.')
                        break

                    if eval_strategy == "worst":
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                    elif eval_strategy == "best":                    
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices
                    elif eval_strategy == "random":
                        random.shuffle(random_indexes)
                        indexes = random_indexes
                    else:
                        assert eval_strategy in ["worst", "best", "random"]

                    point_sampled = False
                    for i in indexes:                        
                        if ious[i]<threshold_list[0]:                                                                
                            obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions)                                                           
                            num_interactions_per_instance[lowest_frame_index][i]+=1                                                           
                            point_sampled = True
                            break
                    if point_sampled:
                        num_interactions+=1
                        total_num_interactions+=1
                        num_interactions_for_sequence[lowest_frame_index] += 1

                        pred_masks = predictor.get_prediction(clicker)
                        clicker.set_pred_masks(pred_masks)
                        ious = clicker.compute_iou()
                        
                        if vis_path:
                            clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_frame_index], round_num=round_num)

                clicker_dict[lowest_frame_index] = clicker
                predictor_dict[lowest_frame_index] = predictor
                interacted_frames.append(lowest_frame_index)
                if pred_masks.shape[0] != seq_num_instances:
                    dummy = np.zeros((seq_num_instances, pred_masks.shape[1], pred_masks.shape[2]))
                    j = 0
                    for i in range(seq_num_instances):                        
                        if i+1 in missing_obj_ids:
                            continue
                        dummy[i][np.where(pred_masks[j]==1)] = 1
                        j +=1
                    pred_masks = torch.from_numpy(dummy)

                # compute background mask                                                                           # MiVOS propagation expects (num_instances+1, 1, H, W)
                bg_mask = np.ones(pred_masks.shape[-2:])                
                for i in range(seq_num_instances):
                    bg_mask[np.where(pred_masks[i]==1.)]=0   
                bg_mask = torch.from_numpy(bg_mask).unsqueeze(0)                                                            # H,W -> 1,H,W
                pred_masks = torch.cat((bg_mask,pred_masks),dim=0)                                                          # [bg, inst1, inst2, ..]
                pred_masks = pred_masks.unsqueeze(1).float()                                                                # num_inst+1, H, W -> num_inst+1,1, H, W
                
                # Propagate
                #print(f'[INFO] Temporal propagation on its way...')
                out_masks = processor.interact(pred_masks,lowest_frame_index)                
                #np.save(os.path.join(vis_path, f'output_masks_round_{round_num}_refined_frame_{lowest_frame_index}_seq_{seq}.npy'), out_masks)

                # Frame-level IoU for the sequence
                iou_for_sequence = compute_iou_for_sequence(out_masks, all_gt_masks[seq])
                iou_copy = copy.deepcopy(iou_for_sequence)
                avg_iou = sum(iou_for_sequence)/len(iou_for_sequence)
                print(f'[INFO] Round {round_num} scores: Max:{max(iou_for_sequence)}, Min: {min(iou_for_sequence)}, Avg: {avg_iou} ')
                while avg_iou > iou_checkpoints[0]:
                    t = iou_checkpoints.pop(0)
                    print(f'[INFO] Round {round_num}: IoU checkpoint {t} for sequence {seq} reached!')
                    checkpoints_for_sequence.append([t,round_num])

                while True:
                    min_iou = min(iou_copy)
                    if min_iou < global_iou_threshold:
                        if frame_limit_bool and round_num == frame_interaction_limit:
                            print(f'[INFO] Maximum frame interaction limit ({frame_interaction_limit}) reached!')
                            lowest_frame_index = -1
                            break
                        lowest_frame_index = 0
                        break
                        # lowest_frame_index = iou_copy.index(min_iou)
                        # print(f'[INFO] Next index to refine: {lowest_frame_index}, IoU: {min_iou}')
                        # if num_interactions_for_sequence[lowest_frame_index] >= max_iters_for_image:                  
                        #     print(f'[INFO] Budget over - skipping frame {lowest_frame_index}.')
                        #     iou_copy.pop(lowest_frame_index)
                        #     if len(iou_copy)==0:
                        #         lowest_frame_index = -1
                        #         print(f'[INFO] Ran out of click budget for all frames!')
                        #         print(f'[INFO] IoU scores: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {sum(iou_for_sequence)/len(iou_for_sequence)} ')
                        #         break
                        # else:
                        #     break                        
                    else:
                        lowest_frame_index = -1
                        print(f'[INFO] All frames meet Global IoU requirement: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {sum(iou_for_sequence)/len(iou_for_sequence)} ')
                        break
                        

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
           
            all_ious[seq] = iou_for_sequence
            all_interactions[seq] = num_interactions_for_sequence
            all_interactions_per_instance[seq] = num_interactions_per_instance
            all_rounds[seq] = round_num
            all_iou_checkpoints[seq] = checkpoints_for_sequence
            del clicker_dict
            del predictor_dict
            del processor
            del all_frames, iou_for_sequence, num_interactions_for_sequence,num_interactions_per_instance   
   
    results = {'all_ious': all_ious,
                'all_interactions': all_interactions,
                'interactions_per_instance': all_interactions_per_instance,
                'total_num_interactions': [total_num_interactions],
                'iou_threshold': iou_threshold,
                'max_frame_interactions': frame_interaction_limit,
                'max_interactions': max_interactions,
                'number_of_rounds': all_rounds,
                'all_iou_checkpoints': all_iou_checkpoints
    }

    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def load_images(path:str ='/globalwork/roy/dynamite_video/mivos/MiVOS/datasets/DAVIS/DAVIS-2017-trainval')-> dict:
    val_set = os.path.join(path,'ImageSets/2017/val.txt')
    with open(val_set, 'r') as f:
        seqs = [line.rstrip('\n') for line in f.readlines()]
    all_images = {}
    image_path = os.path.join(path,'JPEGImages/480p')
    transform = transforms.Compose([transforms.ToTensor()])
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(image_path, s)
        for file in os.listdir(seq_path):
            if file.endswith('.jpg'):
                im = Image.open(os.path.join(seq_path, file))
                im = transform(im)
                seq_images.append(im)
        seq_images = torch.stack(seq_images)
        all_images[s] = seq_images
    return all_images

def load_gt_masks(path:str='/globalwork/roy/dynamite_video/mivos/MiVOS/datasets/DAVIS/DAVIS-2017-trainval')-> dict:
    val_set = os.path.join(path,'ImageSets/2017/val.txt')
    with open(val_set, 'r') as f:
        seqs = [line.rstrip('\n') for line in f.readlines()]
    all_gt_masks = {}
    mask_path = os.path.join(path,'Annotations/480p')
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(mask_path, s)
        for file in os.listdir(seq_path):
            if file.endswith('.png'):
                im = np.asarray(Image.open(os.path.join(seq_path, file)))
                seq_images.append(im)
        seq_images = np.asarray(seq_images)
        all_gt_masks[s] = seq_images
    return all_gt_masks

def compute_iou_for_sequence(pred: np.ndarray, gt: np.ndarray) -> list:
    ious = []
    for gt_mask, pred_mask in zip(gt, pred):
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        ious.append(intersection/union)
    return ious

def compute_instance_wise_iou_for_sequence(pred: np.ndarray, gt: np.ndarray)->np.ndarray:
    # pred - output masks after temporal propagation
    # gt - ground truth masks
    ious = []
    num_instances = len(np.unique(gt[0])) - 1
    idx = 0
    for gt_frame, pred_frame in zip(gt, pred):    # frame-level
        
        ious_frame = []
        mask_H,mask_W = gt_frame.shape
        
        gt_inst = np.zeros((num_instances,mask_H,mask_W))
        for i in range(num_instances):
            gt_inst[num_instances-i-1][np.where(gt_frame==i+1)] = 1
        
        pred_inst = np.zeros((num_instances,mask_H,mask_W))
        for i in range(num_instances):
            pred_inst[i][np.where(pred_frame==i+1)] = 1
        
        for g,p in zip(gt_inst, pred_inst):     # instance-level
            intersection = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            ious_frame.append(intersection/union)
        ious.append(ious_frame)
        idx+=1
    return np.array(ious)