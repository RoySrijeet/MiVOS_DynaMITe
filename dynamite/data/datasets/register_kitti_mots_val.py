import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
"""
Read KITTI_MOTS dataset in COCO format and 
register it with detectron2
"""

logger = logging.getLogger(__name__)

__all__ = ["register_all_kitti_mots"]

# ==== Predefined splits for KITTI MOTS ===========
_PREDEFINED_SPLITS_KITTI_MOTS = {
"val": "KITTI_MOTS/train/val_gt_as_coco_instances.json",
"train": "KITTI_MOTS/train/train_gt_as_coco_instances.json",
"full": "KITTI_MOTS/train/full_gt_as_coco_instances.json",
"image": "KITTI_MOTS/train/images",
}

def register_all_kitti_mots(root, split='val'):
   json_dir = os.path.join(root, _PREDEFINED_SPLITS_KITTI_MOTS[split])
   image_dir = os.path.join(root, _PREDEFINED_SPLITS_KITTI_MOTS['image'])
   register_coco_instances(name=f"kitti_mots_{split}", metadata=_get_kitti_mots_meta(), json_file=json_dir, image_root=image_dir)
   #print(f"kitti_mots_{split} dataset registered")

def _get_kitti_mots_meta():
    return {}

_root = os.getcwd()
_root = os.path.join(_root, "datasets/")
register_all_kitti_mots(_root, split='val')