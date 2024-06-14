import os
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

_DATASET_ROOT = "/globalwork/roy/dynamite_video/mivos_dynamite/MiVOS_DynaMITe/datasets/"
_DATASET_PATH = {
    "davis_2017_val": {
        "annotations": "DAVIS/DAVIS-2017-trainval/Annotations/480p",
        "images": "DAVIS/DAVIS-2017-trainval/JPEGImages/480p",
        "sets": "DAVIS/DAVIS-2017-trainval/ImageSets/2017/val.txt",
    },
    "mose_val": {
        "annotations": "MOSE/valid/Annotations",
        "images":"MOSE/valid/JPEGImages",
        "sets":"",
    },
    "kitti_mots_val": {
        "annotations": "KITTI_masks/val",
        "images": "KITTI_MOTS/train/images",
        "sets": "KITTI_masks/ImageSets/val.txt",
    },
    "burst_val":{
        "annotations": "BURST_masks/val",
        # "images":"BURST/all_frames/val",
        "images":"BURST/annotated_frames/val",
        "sets":"",
    }
}

def load_images(dataset_name="davis_2017_val", debug_mode=False):
    print(f'[INFO] Loading all frames from the disc...')
    image_path = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["images"])
    if dataset_name=="mose_val":
        seqs = sorted([f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path,f))])
    elif dataset_name=="burst_val":
        seqs = burst_imset()
    else:
        val_set = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["sets"])
        with open(val_set, 'r') as f:
            seqs = [line.rstrip('\n') for line in f.readlines()]
    all_images = {}    
    transform = transforms.Compose([transforms.ToTensor()])
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(image_path, s)
        imagefiles = sorted([f for f in os.listdir(seq_path)])
        for file in imagefiles:
            if file.endswith('.jpg') or file.endswith('.png'):
                im = Image.open(os.path.join(seq_path, file))
                im = transform(im)
                seq_images.append(im)
        seq_images = torch.stack(seq_images)
        all_images[s] = seq_images
        if debug_mode:
            break
    return all_images

def load_gt_masks(dataset_name="davis_2017_val", debug_mode=False):
    print(f'[INFO] Loading ground truth masks from the disc...')
    mask_path = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["annotations"])
    if dataset_name=="mose_val":
        seqs = sorted([f for f in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path,f))])
    elif dataset_name=="burst_val":
        seqs = burst_imset()
    else:
        val_set = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["sets"])
        with open(val_set, 'r') as f:
            seqs = [line.rstrip('\n') for line in f.readlines()]
    all_gt_masks = {}
    print(f'[LOADER] Found {len(seqs)} sequences!')
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(mask_path, s)
        maskfiles = sorted([f for f in os.listdir(seq_path)])
        for file in maskfiles:
            if file.endswith('.jpg') or file.endswith('.png'):
                im = np.asarray(Image.open(os.path.join(seq_path, file)))
                seq_images.append(im)
        seq_images = np.asarray(seq_images)
        all_gt_masks[s] = seq_images
        if debug_mode:
            break
    return all_gt_masks

def load_sequence_images(sequence, dataset_name):
    print(f"[LOADER] Loading frames of sequence {sequence}")
    image_root = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["images"])
    seq_path = os.path.join(image_root, sequence)
    transform = transforms.Compose([transforms.ToTensor()])
    seq_images = []
    for file in tqdm(os.listdir(seq_path)):
        if file.endswith('.jpg') or file.endswith('.png'):
            im = Image.open(os.path.join(seq_path, file))
            im = transform(im)
            seq_images.append(im)
    seq_images = torch.stack(seq_images)
    return seq_images

def load_sequence_masks(sequence, dataset_name):
    print(f"[LOADER] Loading masks of sequence {sequence}")
    mask_root = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["annotations"])
    seq_path = os.path.join(mask_root, sequence)
    seq_masks = []
    for file in tqdm(os.listdir(seq_path)):
        if file.endswith('.png'):
            im = np.asarray(Image.open(os.path.join(seq_path, file)))
            seq_masks.append(im)
    seq_masks = np.asarray(seq_masks)
    return seq_masks

def burst_imset():
    dataset_name="burst_val"
    seqs = []
    ann_root = os.path.join(_DATASET_ROOT,_DATASET_PATH[dataset_name]["annotations"])
    for dataset in os.listdir(ann_root):
        if os.path.isdir(os.path.join(ann_root, dataset)):
            assert dataset in ["ArgoVerse","AVA","BDD","Charades","HACS","LaSOT","YFCC100M"]
            dataset_path = os.path.join(ann_root,dataset)
            for video in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, video)):
                    seqs.append(f'{dataset}/{video}')
    return seqs
################################################################

import copy
import torch
from collections import defaultdict
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures.masks import PolygonMasks
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode
from dynamite.data.dataset_mappers.utils import convert_coco_poly_to_mask
from dynamite.inference.utils.eval_utils import get_gt_clicks_coords_eval
from dynamite.data.dataset_mappers.evaluation_dataset_mapper import original_res_annotations, get_instance_map

def burst_video_loader(sequence):
    dataset_name = "burst_val"
    detectron2_dictionary = DatasetCatalog.get(dataset_name)    
    print(len(detectron2_dictionary))
    
    dataloader_dict = defaultdict(list)    
    for idx, dataset_dict in enumerate(detectron2_dictionary):
        
        # key
        if dataset_dict["key"] != sequence:
            continue
        
        # see dynamite/data/dataset_mappers/evaluation_dataset_mapper.py for more info
        
        # read image
        image = utils.read_image(dataset_dict["file_name"], format='RGB')        
        utils.check_image_size(dataset_dict, image)
        orig_image_shape = image.shape[:2]
        padding_mask = np.ones(image.shape[:2])

        # apply transformations
        tfm_gens = [T.ResizeShortestEdge(short_edge_length=[800,800], max_size=1333)]
        image, transforms = T.apply_transform_gens(tfm_gens, image)

        # padding        
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        # store
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))        

        # annotations
        if "annotations" in dataset_dict and len(dataset_dict["annotations"])>0:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                original_res_annotations(obj, orig_image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            if dataset_name == "coco_2017_val":        
                instances = utils.annotations_to_instances(annos, orig_image_shape)
            else:
                instances = utils.annotations_to_instances(annos, orig_image_shape,  mask_format="bitmask")
           
            if not hasattr(instances, 'gt_masks'):
                return None                        
            # tight bbox
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            if len(instances) == 0:
                return None
            
            # masks from polygon
            h, w = instances.image_size        
            if hasattr(instances, 'gt_masks'):
                if dataset_name == "coco_2017_val":
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = instances.gt_masks.tensor

                new_gt_masks, instance_map = get_instance_map(gt_masks)
                dataset_dict['semantic_map'] = instance_map
                new_gt_classes = [0]*new_gt_masks.shape[0]
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                new_instances = Instances(image_size=image_shape)
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
               
                ignore_masks = None
                if 'ignore_mask' in dataset_dict:
                    ignore_masks = dataset_dict['ignore_mask'].to(device='cpu', dtype = torch.uint8)

                (num_clicks_per_object, fg_coords_list, orig_fg_coords_list) = get_gt_clicks_coords_eval(new_gt_masks, image_shape, ignore_masks=ignore_masks)

                # clicks
                dataset_dict["orig_fg_click_coords"] = orig_fg_coords_list
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["bg_click_coords"] = None
                dataset_dict["num_clicks_per_object"] = num_clicks_per_object
                assert len(num_clicks_per_object) == gt_masks.shape[0]
            else:
                return None

            dataset_dict["instances"] = new_instances

        else:
            # if no mask
            dataset_dict['semantic_map'] = None
            dataset_dict["orig_fg_click_coords"] = None
            dataset_dict["fg_click_coords"] = None
            dataset_dict["bg_click_coords"] = None
            dataset_dict["num_clicks_per_object"] = None
            dataset_dict["instances"] = None
        
        dataloader_dict[dataset_dict["key"]].append([idx, [dataset_dict]])

    return dataloader_dict
