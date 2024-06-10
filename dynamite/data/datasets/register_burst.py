# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import glob
import logging
import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
from pycocotools import coco
from imantics import Polygons, Mask
"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_burst", "register_burst_instances"]

# ==== Predefined splits for BURST ===========
_PREDEFINED_SPLITS_BURST = {
"burst_val": ("BURST_masks/val",            # annotations
            "BURST/all_frames/val"),        # images
}

def register_all_burst(root):
    for key, (ann_root, image_root) in _PREDEFINED_SPLITS_BURST.items():
        ann_root = os.path.join(root, ann_root)
        image_root = os.path.join(root, image_root)
        imset = []
        for dataset in os.listdir(ann_root):
            if os.path.isdir(os.path.join(ann_root, dataset)):
                assert dataset in ["ArgoVerse","AVA","BDD","Charades","HACS","LaSOT","YFCC100M"]
                dataset_path = os.path.join(ann_root,dataset)
                for video in os.listdir(dataset_path):
                    if os.path.isdir(os.path.join(dataset_path, video)):
                        imset.append((dataset,video))
        assert len(imset) == 988
        register_burst_instances(
        key,
        _get_burst_instances_meta(),
        ann_root,
        image_root,
        imset
    )
    #print("burst_val dataset registered")

def bbox_from_mask_np(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
  if len(np.where(mask)[0]) == 0:
    return np.array([-1, -1, -1, -1])
  x_min = np.where(mask)[1].min()
  x_max = np.where(mask)[1].max()

  y_min = np.where(mask)[0].min()
  y_max = np.where(mask)[0].max()

  if order == 'Y1Y2X1X2':
    return np.array([y_min, y_max, x_min, x_max])
  elif order == 'X1X2Y1Y2':
    return np.array([x_min, x_max, y_min, y_max])
  elif order == 'X1Y1X2Y2':
    return np.array([x_min, y_min, x_max, y_max])
  elif order == 'Y1X1Y2X2':
    return np.array([y_min, x_min, y_max, x_max])
  else:
    raise ValueError("Invalid order argument: %s" % order)


def _get_burst_instances_meta():
    return {}


def load_burst(annotation_root, image_root, imset, dataset_name=None, extra_annotation_keys=None):

    dataset_dicts = []
    for _video_id, (_dataset, _video) in enumerate(imset):
        #_video = line.rstrip('\n')
        img_list = np.array(glob.glob(os.path.join(image_root, _dataset, _video, '*.jpg')))
        img_list.sort()

        # filter out empty annotations during training
        mask_list = np.array(glob.glob(os.path.join(annotation_root, _dataset, _video, '*.png')))     
        mask_list.sort()
        _mask_file = mask_list[0]
        _mask = np.array(Image.open(_mask_file).convert("P"))
        height, width = _mask.shape
        num_objects = np.max(_mask)
        #for i, (_img_file, _mask_file) in enumerate(zip(img_list,mask_list)):
        for i, _img_file in enumerate(img_list):
            _mask_file=None
            if i == 0:
                _mask_file = mask_list[0]
            record = {}
            record["file_name"] = _img_file
            record["height"] = height
            record["width"] = width
            record["image_id"] = _dataset +'-'+ _video +'-'+ str(_video_id) +'-'+ _img_file.split('/')[-1][:-4]
            record["key"] = f'{_dataset}/{_video}'
            if _mask_file is not None:
                frame_objs = []
                frame_mask = np.array(Image.open(_mask_file).convert("P")).astype(np.uint8)
                for obj_id in range(1, num_objects + 1):
                    obj = {} 
                    obj_mask = (frame_mask == obj_id).astype(np.uint8)
                    bbox = bbox_from_mask_np(obj_mask, order='X1Y1X2Y2')
                    obj = {"segmentation": coco.maskUtils.encode(np.asfortranarray(obj_mask)), 'category_id': 1,
                        'iscrowd': False, 'id': obj_id}
                    
                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                record["annotations"] = frame_objs
            else:
               record["annotations"] = []
            dataset_dicts.append(record)
    return dataset_dicts


def register_burst_instances(name, metadata, ann_root, image_root, imset):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".         # 'burst_val'
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_burst(ann_root, image_root, imset, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        ann_root=ann_root, image_root=image_root, evaluator_type="burst", **metadata
    )

_root = os.getcwd()
_root = os.path.join(_root, "datasets/")
register_all_burst(_root)
