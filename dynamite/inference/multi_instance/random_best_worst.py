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
from collections import defaultdict
from torchvision import transforms
from detectron2.utils.comm import get_world_size        # utils.comm -> primitives for multi-gpu communication
from torch import nn
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor
from PIL import Image
from inference_core import InferenceCore
from metrics.j_and_f_scores import batched_jaccard,batched_f_measure

def evaluate(
    model, propagation_model, fusion_model,
    data_loader, iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "worst", seed_id = 0, vis_path = None, max_rounds=0
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
    # args
    print(f'[EVALUATOR INFO] IoU Threshold: {iou_threshold}')
    print(f'[EVALUATOR INFO] Max Interactions per Frame: {max_interactions}')
    print(f'[EVALUATOR INFO] Max Rounds per Sequence: {max_rounds}')
    print(f'[EVALUATOR INFO] DynaMITe Evaluation Strategy: {eval_strategy}')
    print(f'[EVALUATOR INFO] DynaMITe Point Sampling Strategy: {sampling_strategy}')

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))                       # 1999 (davis_2017_val)
    logger.info(f"Using {eval_strategy} evaluation strategy with random seed {seed_id}")
    
    print(f'[EVALUATOR INFO] Loading all frames from the disc...')
    all_images = load_images()
    print(f'[EVALUATOR INFO] Loading all ground truth masks from the disc...')
    all_gt_masks = load_gt_masks()
    
    all_interactions = {}
    all_interactions_per_round = {}
    all_interactions_per_instance = {}
    all_rounds = {}
    all_j_and_f = {}
    all_jaccard = {}
    all_contour = {}
    all_ious = {}
    all_instance_level_iou = {}
    copy_iou_checkpoints = [0.85, 0.90, 0.95, 0.99]
    
    with ExitStack() as stack:                                           

        if isinstance(model, nn.Module):    
            stack.enter_context(inference_context(model))                           
        stack.enter_context(torch.no_grad())                             
                           
        total_num_interactions = 0                                       # for whole dataset       
        total_num_rounds = 0                                             # for whole dataset

        random.seed(123456+seed_id)
        
        dataloader_dict = defaultdict(list)
        print(f'[EVALUATOR INFO] Iterating through the Data Loader...')
        # iterate through the data_loader, one image at a time
        for idx, inputs in enumerate(data_loader):          
            curr_seq_name = inputs[0]["file_name"].split('/')[-2]
            dataloader_dict[curr_seq_name].append([idx, inputs])

        print(f'[EVALUATOR INFO] Sequence-wise evaluation...')
        for seq in list(dataloader_dict.keys()):
            print(f'\n[SEQUENCE INFO] Sequence: {seq}')
            
            # Initialize propagation module - once per-sequence
            seq_object_ids = set(np.unique(all_gt_masks[seq][0]))       # object ids in the sequence
            seq_num_instances = len(seq_object_ids) - 1                 # remove bg
            all_frames = all_images[seq]                                # collect all image frames in the sequence
            num_frames = len(all_frames)                                # sequence length
            all_frames = all_frames.unsqueeze(0).float()                
            processor = InferenceCore(propagation_model, fusion_model, all_frames, seq_num_instances)       # initialize prop module
            
            lowest_frame_index = 0                                      # frame with lowest IoU after propagation - initially, first frame
            round_num = 0                                               # counter for number of propagation rounds
            clicker_dict = {}                                           # records clickers for interacted frames
            predictor_dict = {}                                         # records predictors for interacted frames

            num_interactions_for_sequence = [0]*num_frames              # records #interactions on each frame of the sequence
            num_interactions_per_instance_for_sequence = [[]] * num_frames           # records #interactions on each object of each frame of the sequence
            out_masks = None                                            # stores predicted masks by prop module, initially None
            all_interactions_per_round[seq] = []                           # records round-wise interaction details
            all_j_and_f[seq] = [0]                                      # records J&F metric score for the sequence
            seq_avg_iou = 0                                                 # records average IoU for the sequence
            seq_avg_jf = 0                                                  # records average J&F for the sequence
            iou_checkpoints = copy.deepcopy(copy_iou_checkpoints)            # IoU checkpoints
            
            instance_level_iou = [[]] * num_frames
            
            # start
            while lowest_frame_index!=-1:
                round_num += 1      # round start
                loop = 0

                print(f'[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] DynaMITe refining frame {lowest_frame_index}...')
                idx, inputs = dataloader_dict[seq][lowest_frame_index]                              # load frame with lowest IoU

                # (re)load Clicker and Predictor (DynaMITe)
                if num_interactions_for_sequence[lowest_frame_index] == 0:                           # if frame has been previously interacted with 
                    clicker = Clicker(inputs, sampling_strategy)
                    predictor = Predictor(model)
                    repeat = False
                else:                                                  
                    clicker = clicker_dict[lowest_frame_index]
                    predictor = predictor_dict[lowest_frame_index]
                    repeat = True
                
                #check for missing objects
                object_ids = set(np.unique(all_gt_masks[seq][lowest_frame_index]))                  # objects present in the frame                
                num_instances = clicker.num_instances
                missing_obj_ids = None
                if num_instances!=seq_num_instances:                    
                    missing_obj_ids = seq_object_ids - object_ids
                
                # if available, pred_masks from prev round is set to Clicker
                if out_masks is not None:                                                           # out_masks = [num_frames, H, W]
                    mask_H,mask_W = out_masks[lowest_frame_index].shape
                    pred_masks = np.zeros((seq_num_instances,mask_H,mask_W))                        # pred_masks = [num_instances, H, W]
                    for i in range(seq_num_instances):
                        if missing_obj_ids is not None:                                             # take into account missing objects
                            if i+1 in missing_obj_ids:
                                continue
                        pred_masks[i][np.where(out_masks[lowest_frame_index]==i+1)] = 1       
                    pred_masks = torch.from_numpy(pred_masks)
                    clicker.set_pred_masks(pred_masks)   
                # otherwise, predict segmentation mask with DynaMITe
                else:
                    pred_masks = predictor.get_prediction(clicker)
                    clicker.set_pred_masks(pred_masks)
                # instance-wise IoU
                ious = clicker.compute_iou()
                instance_level_iou[lowest_frame_index] = [iou.tolist() for iou in ious]
                frame_avg_iou = sum(ious)/len(ious)                                                        
                if vis_path:
                    clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_frame_index], round_num=round_num)             
                
                
                # record the interaction, if the frame has never been interacted with
                if not repeat:
                    total_num_interactions+=(num_instances)                                                       # counter over all dataset
                    num_interactions_for_sequence[lowest_frame_index] += num_instances              
                    num_interactions_per_instance_for_sequence[lowest_frame_index] = [1]*(num_instances+1)      
                    num_interactions_per_instance_for_sequence[lowest_frame_index][-1] = 0                        # no interaction for bg yet, so reset                    
                    
                    # round,loop,frame_idx,obj_idx,#interactions,frame_iou,seq_iou,seq_jf
                    all_interactions_per_round[seq].append([round_num, loop, 
                                                           lowest_frame_index, list(map(int,object_ids - {0})), 
                                                           num_interactions_for_sequence[lowest_frame_index],
                                                           frame_avg_iou.item(),
                                                           seq_avg_iou, seq_avg_jf])
                
                # interaction limit
                max_iters_for_image = max_interactions * num_instances                                      

                point_sampled = True
                random_indexes = list(range(len(ious)))

                #interative refinement loop                 
                while (num_interactions_for_sequence[lowest_frame_index]<max_iters_for_image):               # 1st stopping criterion - if interaction budget is over
                    
                    while all(iou >= iou_checkpoints[0] for iou in ious):                                    # IoU checkpoints
                        t = iou_checkpoints.pop(0)
                        print(f'[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] Frame {lowest_frame_index} reached IoU Checkpoint {t} after {num_interactions_for_sequence[lowest_frame_index]} interactions.')
                    
                    if all(iou >= iou_threshold for iou in ious):                                             # 2nd stopping criterion - if IoU threshold is reached
                        print(f'[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] Frame {lowest_frame_index} meets IoU Threshold {iou_threshold} after {num_interactions_for_sequence[lowest_frame_index]} interactions!')
                        break
                    
                    loop += 1
                    #print(f'[DynaMITe INFO] Round {round_num}, Loop {loop}')                    
                    
                    # According to eval strategy, select which instance to interact with
                    if eval_strategy == "worst":
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                    elif eval_strategy == "best":                    
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices
                    elif eval_strategy == "random":
                        random.shuffle(random_indexes)
                        indexes = random_indexes
                    else:
                        assert eval_strategy in ["worst", "best", "random"]

                    # DynaMITe Interaction
                    point_sampled = False
                    for i in indexes:                        
                        if ious[i]<iou_threshold:                                                                
                            obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions_for_sequence[lowest_frame_index])
                            num_interactions_per_instance_for_sequence[lowest_frame_index][i]+=1                                      
                            point_sampled = True
                            break
                    if point_sampled:
                        total_num_interactions+=1
                        num_interactions_for_sequence[lowest_frame_index] += 1
                        pred_masks = predictor.get_prediction(clicker)
                        clicker.set_pred_masks(pred_masks)
                        ious = clicker.compute_iou()
                        #print(f'[DynaMITe REFINEMENT][SEQ:{seq}][ROUND:{round_num}][LOOP: {loop}] Frame {lowest_frame_index} Instance IoUs: {ious}')
                        if vis_path:
                            clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions_for_sequence[lowest_frame_index], round_num=round_num)
                        
                        frame_avg_iou = sum(ious)/len(ious)
                        instance_level_iou[lowest_frame_index] = [iou.tolist() for iou in ious]
                        all_interactions_per_round[seq].append([round_num, loop, lowest_frame_index, int(obj_index), num_interactions_for_sequence[lowest_frame_index], frame_avg_iou.item(), seq_avg_iou, seq_avg_jf])
                    
                # store clicker and predictor, in case this frame needs to be interacted with again
                clicker_dict[lowest_frame_index] = clicker
                predictor_dict[lowest_frame_index] = predictor

                # account for missing objects
                if pred_masks.shape[0] != seq_num_instances:
                    dummy = np.zeros((seq_num_instances, pred_masks.shape[1], pred_masks.shape[2]))
                    j = 0
                    for i in range(seq_num_instances):                        
                        if i+1 in missing_obj_ids:
                            continue
                        dummy[i][np.where(pred_masks[j]==1)] = 1
                        j +=1
                    pred_masks = torch.from_numpy(dummy)

                # compute background mask                                                                                   # MiVOS propagation expects (num_instances+1, 1, H, W)
                bg_mask = np.ones(pred_masks.shape[-2:])                
                for i in range(seq_num_instances):
                    bg_mask[np.where(pred_masks[i]==1.)]=0   
                bg_mask = torch.from_numpy(bg_mask).unsqueeze(0)                                                            # H,W -> 1,H,W
                pred_masks = torch.cat((bg_mask,pred_masks),dim=0)                                                          # [bg, inst1, inst2, ..]
                pred_masks = pred_masks.unsqueeze(1).float()                                                                # num_inst+1, H, W -> num_inst+1,1, H, W
                
                # Propagate
                print(f'[PROPAGATION INFO][SEQ:{seq}][ROUND:{round_num}] Temporal propagation in progress...')
                out_masks = processor.interact(pred_masks,lowest_frame_index)                

                # metrics (mean: over instances in a frame)
                jaccard_mean, jaccard_instances = batched_jaccard(all_gt_masks[seq], out_masks, average_over_objects=True, nb_objects=seq_num_instances)
                contour_mean, contour_instances = batched_f_measure(all_gt_masks[seq], out_masks, average_over_objects=True, nb_objects=seq_num_instances)

                j_and_f = 0.5*jaccard_mean + 0.5*contour_mean
                j_and_f = j_and_f.tolist()
                seq_avg_jf = sum(j_and_f)/len(j_and_f)

                iou_for_sequence = compute_iou_for_sequence(out_masks, all_gt_masks[seq])
                seq_avg_iou = sum(iou_for_sequence)/len(iou_for_sequence)
                print(f'[PROPAGATION INFO][SEQ:{seq}][ROUND:{round_num}] Prediction results: Average IoU: {seq_avg_iou}, Average J&F: {seq_avg_jf}')
                
                all_interactions_per_round[seq].append([round_num, '-', '-', '-', sum(num_interactions_for_sequence), '-', seq_avg_iou, seq_avg_jf])
                
                # Check stopping criteria
                iou_copy = copy.deepcopy(iou_for_sequence)
                while True:
                    min_iou = min(iou_copy)
                    if min_iou < iou_threshold:         # 1. whether all frames meet IoU threshold
                        if round_num == max_rounds:     # 2. whether round budget is over
                            print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] Maximum round limit ({max_rounds}) reached!')
                            print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] IoU scores: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {seq_avg_iou} ')
                            lowest_frame_index = -1
                            break
                        lowest_frame_index = iou_copy.index(min_iou)
                        print(f'[EVALUATOR INFO][SEQ:{seq}][ROUND:{round_num}] Next index to refine: {lowest_frame_index}, IoU: {min_iou}')
                        if num_interactions_for_sequence[lowest_frame_index] >= max_iters_for_image:     # if interaction budget is over for a frame, look for another frame             
                            print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] Budget over - skipping frame {lowest_frame_index}.')
                            iou_copy.pop(lowest_frame_index)
                            if len(iou_copy)==0:                                                         # 3. whether interaction budget is over for all frames
                                lowest_frame_index = -1
                                print(f'[INFO][SEQ:{seq}][ROUND:{round_num}] Ran out of click budget for all frames!')
                                print(f'[INFO][SEQ:{seq}][ROUND:{round_num}] IoU scores: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {seq_avg_iou} ')
                                break
                        else:
                            break                        
                    else:
                        lowest_frame_index = -1
                        print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] All frames meet IoU requirement: Max:{max(iou_for_sequence)}, Min: {min_iou}, Avg: {seq_avg_iou} ')
                        break
                         

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            total_num_rounds += round_num

            # store #interactions for entire sequence
            all_interactions[seq] = num_interactions_for_sequence
            all_interactions_per_instance[seq] = num_interactions_per_instance_for_sequence
            all_instance_level_iou[seq] = instance_level_iou
            all_rounds[seq] = round_num

            # store metrics for entire sequence
            all_j_and_f[seq] = j_and_f                                                            
            all_jaccard[seq] = jaccard_mean.tolist()
            all_contour[seq] = contour_mean.tolist()
            all_ious[seq] = iou_for_sequence 
            
            del clicker_dict, predictor_dict, processor
            del all_frames, num_interactions_for_sequence, num_interactions_per_instance_for_sequence   
            del iou_for_sequence, jaccard_instances, jaccard_mean, contour_instances, contour_mean


    results = {
                'iou_threshold': iou_threshold,
                'max_interactions': max_interactions,
                'max_rounds': max_rounds,
                'iou_checkpoints': copy_iou_checkpoints,
                
                'total_num_interactions': [total_num_interactions],
                'all_interactions': all_interactions,
                'all_interactions_per_instance': all_interactions_per_instance,
                
                'total_num_rounds': [total_num_rounds],
                'all_rounds': all_rounds,
                'all_interactions_per_round': all_interactions_per_round,

                'all_instance_level_iou': all_instance_level_iou,
                'all_j_and_f' : all_j_and_f,
                'all_jaccard' : all_jaccard,
                'all_contour' : all_contour,
                'all_ious': all_ious,
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