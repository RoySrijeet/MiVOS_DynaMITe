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
# from ..clicker import Clicker
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor
from PIL import Image
import matplotlib.pyplot as plt
from inference_core import InferenceCore
from pathlib import Path

def evaluate(
    model, propagation_model, fusion_model,
    data_loader, iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "worst", seed_id = 0, vis_path = None, video_mode=False,
):
    """
    Run model on the data_loader and return a dict, later used to calculate
    all the metrics for multi-instance inteactive segmentation such as NCI,
    NFO, NFI, and Avg IoU.
    The model will be used in eval mode.

    Arguments:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        iou_threshold: float
            Desired IoU value for each object mask
        max_interactions: int
            Maxinum number of interactions per object
        sampling_strategy: int
            Strategy to avaoid regions while sampling next clicks
            0: new click sampling avoids all the previously sampled click locations
            1: new click sampling avoids all locations upto radius 5 around all
               the previously sampled click locations
        eval_strategy: str
            Click sampling strategy during refinement
        seed_id: int
            Used to generate fixed seed during evaluation
        vis_path: str
            Path to save visualization of masks with clicks during evaluation
        video_mode: bool
            If set to True, the input images are frames of a video sequence

    Returns:
        Dict with following keys:
            'total_num_instances': total number of instances in the dataset
            'total_num_interactions': total number of interactions/clicks sampled 
            'total_compute_time_str': total compute time for evaluating the dataset
            'iou_threshold': iou_threshold
            'num_interactions_per_image': a dict with keys as image ids and values 
             as total number of interactions per image
            'final_iou_per_object': a dict with keys as image ids and values as
             list of ious of all objects after final interaction
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))                       # 1999 (davis_2017_val)
    logger.info(f"Using {eval_strategy} evaluation strategy with random seed {seed_id}")

    total = len(data_loader)  # inference data loader must have a fixed length
   
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0
    total_eval_time = 0
    
    # VID
    print(f'[INFO] Loading all frames and ground truth masks from disc...')
    
    all_images = load_images()
    all_gt_masks = load_gt_masks()
    
    all_ious = {}
    
    with ExitStack() as stack:                                           # managing multiple context managers

        if isinstance(model, nn.Module):    
            stack.enter_context(inference_context(model))               # (context manager) set the model temporarily to .eval()
        # load propagation model
            
        stack.enter_context(torch.no_grad())                             # (context manager) disable gradient calculation

        total_num_instances = 0                                          # in the dataset                               
        total_num_interactions = 0                                       # that were sampled
        
        final_iou_per_object = defaultdict(list)                          # will store IoUs for all objects (in a list), for each image (image-id as key)
        num_interactions_per_image = {}                                    # key: image-id, value: #interactions

        random.seed(123456+seed_id)
        start_data_time = time.perf_counter()
        
        all_frames = None

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
            num_instances = len(np.unique(all_gt_masks[seq][0])) - 1
            all_frames = all_images[seq]
            num_frames = len(all_frames)
            all_frames = all_frames.unsqueeze(0).float()
            processor = InferenceCore(propagation_model, fusion_model, all_frames, num_instances)
            
            lowest_index = 0
            interacted_frames = []
            clicker_dict = {}
            predictor_dict = {}
            iou_for_sequence = [0]*num_frames
            num_interactions_for_sequence = [0]*num_frames
            out_masks = None
            round_num = 0
            weakest_instance = None

            while lowest_index!=-1:
                round_num += 1
                print(f'[INFO] DynaMITe refining frame {lowest_index} of sequence {seq}')
                idx, inputs = dataloader_dict[seq][lowest_index]
                
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                start_compute_time = time.perf_counter()
                
                if lowest_index not in interacted_frames:    
                    clicker = Clicker(inputs, sampling_strategy)
                    #predictor = Predictor(model)
                    repeat = False
                else:    
                    clicker = clicker_dict[lowest_index]
                    #predictor = predictor_dict[lowest_index]
                    repeat = True
                predictor = Predictor(model)
                
                if out_masks is not None:
                    mask_H,mask_W = out_masks[lowest_index].shape
                    prev_pred = np.zeros((num_instances,mask_H,mask_W))
                    for i in range(num_instances):
                        prev_pred[i][np.where(out_masks[lowest_index]==i+1)] = 1
                    clicker.set_pred_masks(torch.from_numpy(prev_pred))
                
                clicker_dict[lowest_index] = clicker
                predictor_dict[lowest_index] = predictor

                if vis_path:
                    clicker.save_visualization(vis_path, ious=iou_for_sequence[lowest_index], num_interactions=num_interactions_for_sequence[lowest_index], round_num=round_num)                      # num_interactions==0: ground truth masks
                
                num_instances = clicker.num_instances                                                       
                total_num_instances+=num_instances                                                          # counter for whole dataset
                total_num_interactions+=(num_instances)                                                     # we start with atleast one interaction per instance (center of each instance, on the ground truth mask)

                num_interactions = num_instances
                num_interactions_for_sequence[lowest_index] += num_instances                                                           
                
                num_clicks_per_object = [1]*(num_instances+1)                                               # +1 for background
                num_clicks_per_object[-1] = 0                                                               # no interaction for bg yet, so reset

                max_iters_for_image = max_interactions * num_instances                                      # budget defined per instance (max_interactions=10 per instance)
                                                                                                            # first call - from the clicker object, take the input sample, and max_timestamps (?) to make predictions
                if not repeat:                                                                                            # first call also populates many Predictor attributes (see inference.utils.predictor.py)
                    pred_masks = predictor.get_prediction(clicker)                                              # at orig (pre-transfn) image res (num_inst xHxW)
                    clicker.set_pred_masks(pred_masks)                                                          # clicker.pred_masks 
                
                ious = clicker.compute_iou()                                                                # compute iou (one score per channel==instance)
                if vis_path:
                    clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_index], round_num=round_num)  

                point_sampled = True

                random_indexes = list(range(len(ious)))

                #interative refinement loop
                while (num_interactions<max_iters_for_image):                                                 # if not over-budget
                    if all(iou >= iou_threshold for iou in ious):                                             # if mask quality met for all instances
                        break

                    index_clicked = [False]*(num_instances+1)                                                  # redundant - probably
                    if eval_strategy == "worst":
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices           # returns a list of indices that sorts ious list from lowest to highest
                    elif eval_strategy == "best":                    
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices            # returns a list of indices that sorts ious list from highest to lowest
                    elif eval_strategy == "random":
                        random.shuffle(random_indexes)
                        indexes = random_indexes
                    else:
                        assert eval_strategy in ["worst", "best", "random"]

                    point_sampled = False
                    if weakest_instance is not None:
                        obj_index = clicker.get_next_click(refine_obj_index=weakest_instance, time_step=num_interactions)   #num_interactions - counter over image
                        total_num_interactions+=1                                                                           # for dataset
                        
                        index_clicked[obj_index] = True
                        num_clicks_per_object[i]+=1                                                           
                        point_sampled = True                    
                    else:
                        for i in indexes:                        
                            if ious[i]<iou_threshold:                                                                # sample click on the first instance that has iou below threshold
                                obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions)   #num_interactions - counter over image
                                total_num_interactions+=1                                                             # for dataset
                                
                                index_clicked[obj_index] = True
                                num_clicks_per_object[i]+=1                                                           
                                point_sampled = True
                                break
                    if point_sampled:
                        num_interactions+=1                                                                        
                        num_interactions_for_sequence[lowest_index] += num_instances

                        pred_masks = predictor.get_prediction(clicker)
                        clicker.set_pred_masks(pred_masks)
                        ious = clicker.compute_iou()
                        
                        if vis_path:
                            clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_index], round_num=round_num)
                        # final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)

                interacted_frames.append(lowest_index)

                # compute background mask                                                                           # MiVOS temporal propagation expects (num_instances+1, 1, H, W)
                bg_mask = np.ones(pred_masks.shape[-2:])                
                for i in range(num_instances):
                    bg_mask[np.where(pred_masks[i]==1.)]=0   
                bg_mask = torch.from_numpy(bg_mask).unsqueeze(0)                                                            # H,W -> 1,H,W
                pred_masks = torch.cat((bg_mask,pred_masks),dim=0)                                                          # [bg, inst1, inst2, ..]
                pred_masks = pred_masks.unsqueeze(1).float()                                                                 # num_inst+1, H, W -> num_inst+1,1, H, W
                
                # Propagate
                print(f'[INFO] Temporal propagation on its way...')
                out_masks = processor.interact(pred_masks,lowest_index)                
                np.save(os.path.join(vis_path, f'output_masks_round_{round_num}_refined_frame_{lowest_index}_seq_{seq}.npy'), out_masks)

                # Frame-level IoU for the sequence
                # iou_for_sequence = compute_iou_for_sequence(out_masks, all_gt_masks[seq])
                # min_iou = min(iou_for_sequence)
                # if min_iou < iou_threshold:
                #     lowest_index = iou_for_sequence.index(min_iou)                    
                #     print(f'[INFO] Next index to refine: {lowest_index}, IoU: {min_iou}')
                # else:
                #     lowest_index = -1
                #     print(f'[INFO] All frames meet IoU requirement: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {sum(iou_for_sequence)/len(iou_for_sequence)} ')
                #     all_ious[seq] = iou_for_sequence   

                # Instance-level IoU for the sequence
                instance_wise_ious = compute_instance_wise_iou_for_sequence(out_masks, all_gt_masks[seq])
                min_iou = np.min(instance_wise_ious)
                if min_iou < iou_threshold:
                    min_idx = np.unravel_index(np.argmin(instance_wise_ious), instance_wise_ious.shape)
                    lowest_index = min_idx[0]
                    weakest_instance = min_idx[1]
                    print(f'[INFO] Next index to refine: {lowest_index}, instance:{weakest_instance}, IoU: {min_iou}')
                else:
                    lowest_index = -1
                    print(f'[INFO] All frames meet IoU requirement: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {sum(iou_for_sequence)/len(iou_for_sequence)} ')
                    all_ious[seq] = instance_wise_ious   

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
           
            final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)
            num_interactions_per_image[f"{inputs[0]['image_id']}_{idx}"] = num_interactions
            
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        # f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"Total instances: {total_num_instances}. "
                        f"Average interactions:{(total_num_interactions/total_num_instances):.2f}. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
            del clicker_dict
            del predictor_dict
            del processor
            del all_frames

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        ),
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = {'total_num_instances': [total_num_instances],
                'total_num_interactions': [total_num_interactions],
                'total_compute_time_str': total_compute_time_str,
                'iou_threshold': iou_threshold,
                'final_iou_per_object': [final_iou_per_object],
                'num_interactions_per_image': [num_interactions_per_image],
    }

    return results,all_ious


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