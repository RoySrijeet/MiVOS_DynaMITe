#Adapted by Srijeet Roy from: https://github.com/amitrana001/DynaMITe/blob/main/train_net.py

import csv
import numpy as np
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging

from typing import Any, Dict, List, Set

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
)
from dynamite.utils.misc import default_argument_parser

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dynamite import (
    COCOLVISDatasetMapper, EvaluationDatasetMapper
)

from dynamite import (
    add_maskformer2_config,
    add_hrnet_config
)

from dynamite.inference.utils.eval_utils import log_single_instance, log_multi_instance
# MiVOS
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
import os
from collections import defaultdict

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to Mask2Former.
    """

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        mapper = EvaluationDatasetMapper(cfg,False,dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)        # d2 call
    
    @classmethod
    def interactive_evaluation(cls, cfg, dynamite_model, propagation_model, fusion_model, args=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        print('[INFO] Interactive Evaluation started...')
        if not args:
            return 

        logger = logging.getLogger(__name__)

        if args and args.eval_only:
            eval_datasets = args.eval_datasets      
            vis_path = args.vis_path                
            eval_strategy = args.eval_strategy      
            seed_id = args.seed_id
            iou_threshold = args.iou_threshold
            max_interactions = args.max_interactions
            max_frame_interactions = args.max_frame_interactions
        
        assert iou_threshold>=0.80

        print(f'[INFO] Evaluation datasets: {eval_datasets}')
        print(f'[INFO] Evaluation strategy: {eval_strategy}')
        print(f'[INFO] IoU Threshold: {iou_threshold}')
        print(f'[INFO] Max interaction limit: {max_interactions}')

        for dataset_name in eval_datasets:

            if dataset_name in ["davis_2017_val","sbd_multi_insts","coco_2017_val"]:
                print(f'[INFO] Initiating Multi-Instance Evaluation on {eval_datasets}...')
                
                if eval_strategy in ["random", "best", "worst"]:
                    from dynamite.inference.multi_instance.random_best_worst import evaluate
                elif eval_strategy == "max_dt":
                    from dynamite.inference.multi_instance.max_dt import evaluate
                elif eval_strategy == "wlb":
                    from dynamite.inference.multi_instance.wlb import evaluate
                elif eval_strategy == "round_robin":
                    from dynamite.inference.multi_instance.round_robin import evaluate
                
                print(f'[INFO] Loaded Evaluation routine following {eval_strategy} evaluation strategy!')
                
                print(f'[INFO] Loading test data loader from {dataset_name}...')
                data_loader = cls.build_test_loader(cfg, dataset_name)
                print(f'[INFO] Data loader  preparation complete! length: {len(data_loader)}')
                
                print(f'[INFO] Starting evaluation...')
                results_i = evaluate(dynamite_model, propagation_model, fusion_model, data_loader, iou_threshold = iou_threshold,
                                    max_interactions = max_interactions,
                                    eval_strategy = eval_strategy, seed_id=seed_id,
                                    vis_path=vis_path, max_frame_interactions=max_frame_interactions)
                print(f'[INFO] Evaluation complete for dataset {dataset_name}!')
                import json
                with open(os.path.join(vis_path,'results.json'), 'w') as f:
                    json.dump(results_i, f)
                
                summary = summarize_results(results_i)
                with open(os.path.join(vis_path,'summary.json'), 'w') as f:
                    json.dump(summary, f)
                
                # results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                # if comm.is_main_process():
                #     # sum the values with same keys
                #     assert len(results_i) > 0
                #     res_gathered = results_i[0]
                #     results_i.pop(0)
                #     for _d in results_i:
                #         for k in _d.keys():
                #             res_gathered[k] += _d[k]
                #     log_multi_instance(res_gathered, max_interactions=max_interactions,
                #                     dataset_name=dataset_name, iou_threshold=iou_threshold)

def summarize_results(results):
    all_ious = results['all_ious']
    all_interactions = results['all_interactions']
    interactions_per_instance = results['interactions_per_instance']

    seqs = list(all_ious.keys())
    summary = defaultdict()
    summary['meta'] = {}
    summary['meta']['total_interactions_over_dataset'] = results['total_num_interactions'][0]
    summary['meta']['iou_threshold'] = results['iou_threshold']
    summary['meta']['max_interactions_per_frame'] = results['max_interactions']
    summary['meta']['max_frames_interaction'] = results['max_frame_interactions']
    avg_iou_over_dataset = []
    total_frames_over_dataset = []
    total_frames_interacted = []
    total_instances_over_dataset = []
    total_instances_interacted = []
    total_failed_sequences = 0
    total_failed_frames = []
    for seq in seqs:
        summary[seq] = {}
        ious = all_ious[seq]
        summary[seq]['max_iou'] = max(ious)
        summary[seq]['min_iou'] = min(ious)
        summary[seq]['avg_iou'] = sum(ious)/len(ious)
        avg_iou_over_dataset.append(summary[seq]['avg_iou'])
        total_failed_frames.append(sum(1 for i in ious if i < results['iou_threshold']))
        if summary[seq]['avg_iou'] < results['iou_threshold']:
            total_failed_sequences +=1

        interactions = all_interactions[seq]    

        summary[seq]['total_frames'] = len(interactions)
        total_frames_over_dataset.append(summary[seq]['total_frames'])
        summary[seq]['frames_interacted'] = np.count_nonzero(np.array(interactions))
        total_frames_interacted.append(summary[seq]['frames_interacted'])
        summary[seq]['total_interactions'] = sum(interactions)

        object_clicks = defaultdict(lambda:0)
        for clicks in interactions_per_instance[seq]:
            if len(clicks) !=0:
                for c in range(len(clicks)):
                    object_clicks[c] += clicks[c]
        summary[seq]['instance_wise_interactions'] = list(object_clicks.items())
        total_instances_over_dataset.append(len(list(object_clicks.keys())))
        total_instances_interacted.append(np.count_nonzero(np.array(list(object_clicks.values()))))

    summary['meta']['avg_iou_over_dataset'] = sum(avg_iou_over_dataset)/len(avg_iou_over_dataset)
    summary['meta']['total_failed_sequences'] = total_failed_sequences
    summary['meta']['total_frames_over_dataset'] = sum(total_frames_over_dataset)
    summary['meta']['total_frames_interacted'] = sum(total_frames_interacted)
    summary['meta']['total_failed_frames'] = sum(total_failed_frames)
    summary['meta']['total_instances_over_dataset'] = sum(total_instances_over_dataset)
    summary['meta']['total_instances_interacted'] = sum(total_instances_interacted)
    return summary



def setup(args):
    """
    Create configs and perform basic setups.
    """
    print('[INFO] Setting up DynaMITE...')
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)                 
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)       # path to config file
    cfg.merge_from_list(args.opts)
    cfg.freeze()                                # make cfg (and children) immutable
    default_setup(cfg, args)                    # D2 call
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dynamite")
    return cfg


def main(args):
    
    cfg = setup(args)       # create configs 
    print('[INFO] Setup complete!')


    # for evaluation
    if args.eval_only:
        print('[INFO] DynaMITExMiVOS Evaluation!')
        torch.autograd.set_grad_enabled(False)

        print('[INFO] Building model...')
        dynamite_model = Trainer.build_model(cfg)                                                # load model (torch.nn.Module)
        print('[INFO] Loading model weights...')                                        
        DetectionCheckpointer(dynamite_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(           # d2 checkpoint load
             cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print('[INFO] Model loaded!')

        prop_model_weights = torch.load('/globalwork/roy/dynamite_video/mivos/MiVOS/saves/propagation_model.pth')
        prop_model = PropagationNetwork().cuda().eval()
        prop_model.load_state_dict(prop_model_weights)
        print(f'[INFO] Propagation module loaded!')

        fusion_model_weights = torch.load('/globalwork/roy/dynamite_video/mivos/MiVOS/saves/fusion.pth')
        fusion_model = FusionNet().cuda().eval()
        fusion_model.load_state_dict(fusion_model_weights)
        print(f'[INFO] Fusion module loaded!')

        res = Trainer.interactive_evaluation(cfg,dynamite_model,prop_model, fusion_model, args)

        return res

    else:
        # for training
        # trainer = Trainer(cfg)
        # trainer.resume_or_load(resume=args.resume)
        # return trainer.train()
        print(f'[INFO] Training routine... Not Implemented')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("[INFO] Command Line Args:", args)
    launch(                                                                            
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )