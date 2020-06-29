# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    # CityscapesEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.events import EventStorage
from fcos.checkpoint import AdetCheckpointer
from fcos.config import get_cfg
from fcos.evaluation import COCOEvaluator

from detectron2.data.datasets import register_coco_instances
import argparse
#import multiprocessing as mp #https://github.com/roym899/abandoned_bag_detection/blob/master/demo.py#L10

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = AdetCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res




def get_parser():
    parser = argparse.ArgumentParser(description="ZeroWaste Detectron2 ML Training models")
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--img-folder', type=str, dest='img_folder', help='data folder mounting point')
    parser.add_argument('--masks-folder', type=str, dest='masks_folder', help='data folder mounting point')
    parser.add_argument('--output-folder', type=str, dest='output_folder', help='data folder mounting point')
    parser.add_argument('--config-file', type=str, dest='config_file', help='training configuraiton ad parameters')
    parser.add_argument('--num-gpus', type=int, default=1, dest='num_gpus', help='number of gpus *per machine')
    parser.add_argument("--num-machines", type=int, default=0,dest='num_machines', help="total number of machines")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



def setup_cfg(args):
    # to improve check out - https://github.com/Julienbeaulieu/iMaterialist2020-Image-Segmentation-on-Detectron2/blob/master/imaterialist/config.py
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    MNT_OUTPUT_PATH = os.path.join(args.output_folder, cfg.OUTPUT_DIR)
    cfg.OUTPUT_DIR = MNT_OUTPUT_PATH
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup_cfg(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        evaluators = [
            Trainer.build_evaluator(cfg, name)
            for name in cfg.DATASETS.TEST
        ]
        res = Trainer.test(cfg, model, evaluators)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)   # multi processing - https://github.com/roym899/abandoned_bag_detection/blob/master/demo.py
    #args = default_argument_parser().parse_args() # original - https://github.com/bongjoonhyun/fcos/blob/master/EE898_PA1_2020_rev2/skeleton/train_net.py
    args = get_parser().parse_args()

    # Register Custom Dataset
    MASKS_PATHS = args.masks_folder
    IMG_PATHS = args.img_folder
    
    TRAIN_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_train_coco_instances.json')
    VAL_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_val_coco_instances.json')
    TEST_PATH = os.path.join(MASKS_PATHS, 'ds2_3c_test_coco_instances.json')

    register_coco_instances(f"custom_dataset_train", {},TRAIN_PATH , IMG_PATHS)
    register_coco_instances(f"custom_dataset_val", {}, VAL_PATH, IMG_PATHS)
    register_coco_instances(f"custom_dataset_test", {}, TEST_PATH, IMG_PATHS)

    
    #setup_logger(name="fvcore")
    #logger = setup_logger()
    #logger.info("Arguments: " + str(args))
    # default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    args.eval_only = False
    args.resume = False
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        #machine_rank=args.machine_rank,
        machine_rank=0,
        #dist_url=args.dist_url,'tcp://127.0.0.1:50153'
        dist_url='tcp://127.0.0.1:49152',
        args=(args,),
    )
