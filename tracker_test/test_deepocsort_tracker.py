import numpy as np
import unittest
from copy import deepcopy
from typing import Dict
import torch

from detectron2.config import CfgNode as CfgNode_
from detectron2.config import instantiate
from detectron2.structures import Boxes, Instances
from detectron2.tracking.base_tracker import build_tracker_head

import cv2

from deepocsort import DeepOCSortTracker
from path_constants import TEST_IMG


class TestDeepOCSORTTracker(unittest.TestCase):
    def setUp(self) -> None:
        # for the config

        self._asso_func = 'giou'
        self._conf_thres = 0.5122620708221085
        self._delta_t = 1
        self._det_thresh = 0
        self._inertia = 0.3941737016672115
        self._iou_thresh = 0.22136877277096445
        self._max_age = 50
        self._min_hits = 1
        self._use_bytes = False

        self._model_weights = 'osnet_x0_25_msmt17.pt'
        self._device = 'cpu' # might not be neededdue to global config but eh.
        self._fp16 = False
        self._w_association_emb = 0.75
        self._alpha_fixed_emb = 0.95
        self._aw_param = 0.5
        self._embedding_off = False
        self._cmc_off = False
        self._aw_off = False
        self._new_kf_off = False

        # borrowing these from test_bbox_iou_tracker.py
        # these shouldnt matter as much as it should, really.
        # the goal of testing is to mainly test whether the updates work as intended or not.
        self._img_size = np.array([600, 851])
        self._prev_boxes = np.array(
            [
                [336, 352, 408, 408],
                [461, 401, 538, 454]
            ]
        ).astype(np.float32)

        self._prev_scores = np.array([0.9, 0.9])
        self._prev_classes = np.array([1,1])
        self._prev_masks = np.ones((2, 600, 800)).astype("uint8")
        
        self._curr_boxes = np.array(
            [
                [336, 353, 408, 409],
                [461, 400, 538, 453]
            ]
        ).astype(np.float32)
        self._curr_scores = np.array([0.95, 0.85])
        self._curr_classes = np.array([1, 1])

        self._prev_instances = {
            "image_size": self._img_size,
            "pred_boxes": self._prev_boxes,
            "scores": self._prev_scores,
            "pred_classes": self._prev_classes,
            "pred_masks": self._prev_masks,
        }
        
        self._prev_instances = self._convertDictPredictionToInstance(self._prev_instances)
        self._curr_masks = np.ones((2, 600, 800)).astype("uint8")

        self._curr_instances = {
            "image_size": self._img_size,
            "pred_boxes": self._curr_boxes,
            "scores": self._curr_scores,
            "pred_classes": self._curr_classes,
            "pred_masks": self._curr_masks,
        }
        self._curr_instances = self._convertDictPredictionToInstance(self._curr_instances)

        # for the update
        print(str(TEST_IMG))
        self._img_feats = cv2.imread(str(TEST_IMG))

    def _convertDictPredictionToInstance(self, prediction: Dict) -> Instances:
        """
        convert prediction from Dict to D2 Instances format
        """
        res = Instances(
            image_size=torch.IntTensor(prediction["image_size"]),
            pred_boxes=Boxes(torch.FloatTensor(prediction["pred_boxes"])),
            pred_masks=torch.IntTensor(prediction["pred_masks"]),
            pred_classes=torch.IntTensor(prediction["pred_classes"]),
            scores=torch.FloatTensor(prediction["scores"]),
        )
        return res
    
    def test_init(self):
        cfg = {
            "_target_": "deepocsort.DeepOCSortTracker",
            "model_weights": self._model_weights,
            "device": self._device,
            "fp16": self._fp16,
            "det_thresh": self._det_thresh,
            "max_age": self._max_age,
            "min_hits": self._min_hits,
            "iou_threshold": self._iou_thresh,
            "delta_t": self._delta_t,
            "asso_func": self._asso_func,
            "inertia": self._inertia,
            "w_association_emb": self._w_association_emb,
            "alpha_fixed_emb": self._alpha_fixed_emb,
            "aw_param": self._aw_param,
            "embedding_off": self._embedding_off,
            "cmc_off": self._cmc_off,
            "aw_off": self._aw_off,
            "new_kf_off": self._new_kf_off
        }

        tracker = instantiate(cfg)
        self.assertTrue(tracker.det_thresh == self._det_thresh)

    def test_from_config(self):
        cfg = CfgNode_()
        cfg.TRACKER_HEADS = CfgNode_()
        cfg.TRACKER_HEADS.TRACKER_NAME = 'DeepOCSortTracker'
        cfg.TRACKER_HEADS.ASSO_FUNC = 'giou'
        cfg.TRACKER_HEADS.CONF_THRES = 0.5122620708221085
        cfg.TRACKER_HEADS.DELTA_T = 1
        cfg.TRACKER_HEADS.DET_THRESH = 0
        cfg.TRACKER_HEADS.INERTIA = 0.3941737016672115
        cfg.TRACKER_HEADS.IOU_THRESH = 0.22136877277096445
        cfg.TRACKER_HEADS.MAX_AGE = 50
        cfg.TRACKER_HEADS.MIN_HITS = 1
        cfg.TRACKER_HEADS.USE_BYTE = False
        cfg.TRACKER_HEADS.MODEL_WEIGHTS = 'osnet_x0_25_msmt17.pt'
        cfg.TRACKER_HEADS.DEVICE = 'cpu' # might not be needed due to global config but eh.
        cfg.TRACKER_HEADS.FP16 = False
        cfg.TRACKER_HEADS.W_ASSOCIATION_EMB = 0.75
        cfg.TRACKER_HEADS.ALPHA_FIXED_EMB = 0.95
        cfg.TRACKER_HEADS.AW_PARAM = 0.5
        cfg.TRACKER_HEADS.EMBEDDING_OFF = False
        cfg.TRACKER_HEADS.CMC_OFF = False
        cfg.TRACKER_HEADS.AW_OFF = False
        cfg.TRACKER_HEADS.NEW_KF_OFF = False

        tracker = build_tracker_head(cfg)
        self.assertTrue(tracker.det_thresh == self._det_thresh)

    def test_initialize_extra_fields(self):
        cfg = {
            "_target_": "deepocsort.DeepOCSortTracker",
            "model_weights": self._model_weights,
            "device": self._device,
            "fp16": self._fp16,
            "det_thresh": self._det_thresh,
            "max_age": self._max_age,
            "min_hits": self._min_hits,
            "iou_threshold": self._iou_thresh,
            "delta_t": self._delta_t,
            "asso_func": self._asso_func,
            "inertia": self._inertia,
            "w_association_emb": self._w_association_emb,
            "alpha_fixed_emb": self._alpha_fixed_emb,
            "aw_param": self._aw_param,
            "embedding_off": self._embedding_off,
            "cmc_off": self._cmc_off,
            "aw_off": self._aw_off,
            "new_kf_off": self._new_kf_off
        }

        tracker = instantiate(cfg)
        instances = tracker._initialize_fields(self._curr_instances)
        self.assertTrue(instances.has("ID"))

    def test_update(self):
        cfg = {
            "_target_": "deepocsort.DeepOCSortTracker",
            "model_weights": self._model_weights,
            "device": self._device,
            "fp16": self._fp16,
            "det_thresh": self._det_thresh,
            "max_age": self._max_age,
            "min_hits": self._min_hits,
            "iou_threshold": self._iou_thresh,
            "delta_t": self._delta_t,
            "asso_func": self._asso_func,
            "inertia": self._inertia,
            "w_association_emb": self._w_association_emb,
            "alpha_fixed_emb": self._alpha_fixed_emb,
            "aw_param": self._aw_param,
            "embedding_off": self._embedding_off,
            "cmc_off": self._cmc_off,
            "aw_off": self._aw_off,
            "new_kf_off": self._new_kf_off
        }

        tracker = instantiate(cfg)
        tracker.img_feats = self._img_feats
        prev_instances = tracker.update(self._prev_instances)
        self.assertTrue(len(prev_instances) == 2)
        self.assertTrue(prev_instances.ID[0] == 0)
        self.assertTrue(prev_instances.ID[1] == 1)
        curr_instances = tracker.update(self._curr_instances)
        self.assertTrue(len(curr_instances.ID) == 2)
        self.assertTrue(curr_instances.ID[0] == 2)
        self.assertTrue(curr_instances.ID[1] == 3)

if __name__ == "__main__":
    unittest.main()