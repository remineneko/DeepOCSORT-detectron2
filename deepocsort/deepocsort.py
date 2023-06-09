from detectron2.detectron2.config.config import CfgNode

from .constants import WEIGHTS_FOLDER

from detectron2.detectron2.tracking.base_tracker import BaseTracker, TRACKER_HEADS_REGISTRY

from detectron2.detectron2.config import configurable
from detectron2.detectron2.structures import Instances, Boxes

from typing import List
import torch

import numpy as np
from .association import *
from .cmc import CMCComputer
from .reid_multibackend import ReIDDetectMultiBackend

from copy import deepcopy


class DeviceNotFoundError(Exception):
    pass


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        ((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2, (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2)
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, cls, delta_t=3, orig=False, emb=None, alpha=0, new_kf=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
        else:
            from filterpy.kalman import KalmanFilter
        self.cls = cls
        self.conf = bbox[-1]
        self.new_kf = new_kf
        if new_kf:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.F = np.array(
                [
                    # x y w h x' y' w' h'
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            # Process and measurement uncertainty happen in functions
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    # x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.history_observations = []
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t

        self.emb = emb

        self.frozen = False

    def update(self, bbox, cls):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(bbox)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb.cpu()

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0

            # Stop velocity, will update in kf during OOS
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


# to be absolutely fair this should be in detectron2/detectron2/tracking folder,
# but I figured this might be slightly easier to navigate.

@TRACKER_HEADS_REGISTRY.register()
class DeepOCSortTracker(BaseTracker):
    """
    A (hopefully decent) implementation of DeepOCSORT for Detectron2 based on mikel-brostorm's DeepOCSORT implementation for YOLOv8. 
    """
    @configurable
    def __init__(
        self,
        model_weights: str = 'osnet_x0_25_msmt17.pt',
        device='cpu',
        fp16=False,
        det_thresh=0,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0
        
        model_path = WEIGHTS_FOLDER / model_weights
        real_device = torch.device(device) if device in ['cpu', 'cuda'] else None
        if not real_device:
            raise DeviceNotFoundError(f"Device {device} cannot be found. Available devices: 'cpu', 'cuda'.")
        self.embedder = ReIDDetectMultiBackend(weights=model_path, device=device, fp16=fp16)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off

        # these are for the update function
        # this is a hack.
        # This is a TERRIBLE hack
        # but I would love to keep the integrity of the .update() function
        # so this is a sacrifice of sort to the Devil I guess.

        self._image_numpy = None
        self.update_tag = 'blub'

    @classmethod
    def from_config(cls, cfg: CfgNode):
        """
        Initialize the tracker using CfgNode.
        If there is no TRACKER_HEADS in config, this function will automatically set the values up for the tracker.

        Args:
            cfg (CfgNode): Config file
        """

        # The following values are based on mikel's config files for DeepOCSORT in the implementation for YOLOv8
        # Should you need to improvise, add and modify these numbers in the config.

        asso_func = cfg.TRACKER_HEADS.get('ASSO_FUNC', 'giou')
        conf_thres = cfg.TRACKER_HEADS.get('CONF_THRES', 0.5122620708221085)
        delta_t = cfg.TRACKER_HEADS.get('DELTA_T',1)
        det_thresh = cfg.TRACKER_HEADS.get('DET_THRESH', 0)
        inertia = cfg.TRACKER_HEADS.get('INERTIA', 0.3941737016672115)
        iou_thresh = cfg.TRACKER_HEADS.get('IOU_THRESH', 0.22136877277096445)
        max_age = cfg.TRACKER_HEADS.get('MAX_AGE', 50)
        min_hits = cfg.TRACKER_HEADS.get('MIN_HITS', 1)
        use_bytes = cfg.TRACKER_HEADS.get('USE_BYTE', False)

        # This _might_ not be necessary, but I will leave it here anyway.

        model_weights = cfg.TRACKER_HEADS.get('MODEL_WEIGHTS', 'osnet_x0_25_msmt17.pt')
        device = cfg.TRACKER_HEADS.get('DEVICE', 'cpu') # might not be needed due to global config but eh.
        fp16 = cfg.TRACKER_HEADS.get('FP16', False)
        w_association_emb = cfg.TRACKER_HEADS.get('W_ASSOCIATION_EMB', 0.75),
        alpha_fixed_emb = cfg.TRACKER_HEADS.get('ALPHA_FIXED_EMB', 0.95)
        aw_param = cfg.TRACKER_HEADS.get('AW_PARAM', 0.5)
        embedding_off = cfg.TRACKER_HEADS.get('EMBEDDING_OFF', False)
        cmc_off = cfg.TRACKER_HEADS.get('CMC_OFF', False)
        aw_off = cfg.TRACKER_HEADS.get('AW_OFF', False)
        new_kf_off = cfg.TRACKER_HEADS.get('NEW_KF_OFF', False)

        return {
            "_target_": "detectron2.tracking.deepocsort.DeepOCSORT",
            "model_weights": model_weights,
            "device": device,
            "fp16": fp16,
            "det_thresh": det_thresh,
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_thresh,
            "delta_t": delta_t,
            "asso_func": asso_func,
            "inertia": inertia,
            "w_association_emb": w_association_emb,
            "alpha_fixed_emb": alpha_fixed_emb,
            "aw_param": aw_param,
            "embedding_off": embedding_off,
            "cmc_off": cmc_off,
            "aw_off": aw_off,
            "new_kf_off": new_kf_off
        }

    @property
    def img_feats(self) -> np.ndarray:
        """
        The representation of the image as a numpy array.

        Returns:
            np.ndarray: The array that represents the image.
        """
        return self._image_numpy
    
    @img_feats.setter
    def img_feats(self, features: np.ndarray):
        self._image_numpy = deepcopy(features)

    def update(self, instances: Instances) -> Instances:
        """
        Updates the instances in accordance with DeepOCSort Algorithm.
        This implementation is adapted from mikel-brostorm's implementation of DeepOCSort in YOLOv8,
            slightly modified to fit the BaseTracker class in Detectron2.

        Args:
            instances (Instances): _description_

        Returns:
            Instances: _description_
        """
        instances = self._initialize_extra_fields(instances)
        if self._prev_instances:
            # to be fair, this doesn't matter... though I would love to try and incorporate this to modify this function.
            # mostly because mikel's implementation already stores the tracks in the tracker, which renders this almost redundant.
            # then again, i could be wrong.
            # TODO: after the base base version is completed, come back to this.
            self._untracked_prev_idx = set(range(len(self._prev_instances)))
            
            # Basically, if I understand this correctly
            # now we have two things to deal with - the current instances extracted
            # and the old untracked ids.

            # if so, one solution would be to deal with them sequentially, no?
            # deal with the current ones first, then the ones that havent been dealt with

            # However, first things first - we extract what the current instances yield:
            boxes: Boxes = instances.pred_boxes
            scores: torch.Tensor = instances.scores
            classes: torch.Tensor = instances.classes
            ids: torch.Tensor = instances.ID

            n_boxes = boxes.tensor.numpy()
            n_scores = scores.numpy()
            n_classes = classes.numpy()
            n_ids = ids.numpy()

            # filter out tracks that doesn't meet the min threshold.
            filtered_indices = n_scores > self.det_thresh
            n_boxes = n_boxes[filtered_indices]
            n_scores = n_scores[filtered_indices]
            n_classes = n_classes[filtered_indices]
            n_ids = n_ids[filtered_indices]

            # a terrible hack to see if it works with mikel's code from yours truly
            # this is to handle the second round of association by OCR
            # though I could force another concat to make it the same as the original implementation by mikel,
            # I think this no-more-concat approach would be more readable.
            dets = np.concatenate((n_boxes, n_scores.T), axis = 1)

            # Embedding
            if self.embedding_off or n_boxes.shape[0] == 0:
                boxes_embeds = np.ones((n_boxes.shape[0], 1))
            else:
                boxes_embeds = self._get_features(n_boxes)

            # CMC
            if not self.cmc_off:
                transform = self.cmc.compute_affine(self.img_feats, n_boxes, self.tag)
                for tracker in self.trackers:
                    tracker.apply_affine_correction(transform)
            
            
            trust = (n_scores - self.det_thresh) / (1 - self.det_thresh)
            af = self.alpha_fixed_emb

            detects_alpha = af + (1 - af) * (1 - trust)

            trks = np.zeros((len(self.trackers), 5))
            trk_embs = []
            to_del = []
            ret = []
            for t, trk in enumerate(trks):
                pos = self.trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
                else:  
                    trk_embs.append(self.trackers[t].get_emb())
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

            if len(trk_embs) > 0:
                trk_embs = np.vstack(trk_embs)
            else:
                trk_embs = np.array(trk_embs)

            for t in reversed(to_del):
                self.trackers.pop(t)

            velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
            last_boxes = np.array([trk.last_observation for trk in self.trackers])
            k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

            """
                First round of association
            """
            # (M detections X N tracks, final score)
            if self.embedding_off or n_boxes.shape[0] == 0 or trk_embs.shape[0] == 0:
                stage1_emb_cost = None
            else:
                stage1_emb_cost = boxes_embeds @ trk_embs.T
            matched, unmatched_dets, unmatched_trks = associate(
                n_boxes,
                trks,
                self.iou_threshold,
                velocities,
                k_observations,
                self.inertia,
                stage1_emb_cost,
                self.w_association_emb,
                self.aw_off,
                self.aw_param,
            )
            
            for m in matched:
                self.trackers[m[1]].update(n_classes[m[0]], n_classes[m[0]])
                self.trackers[m[1]].update_emb(boxes_embeds[m[0]], alpha=detects_alpha[m[0]])

            """
                Second round of associaton by OCR
            """
            
            if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
                left_dets = n_boxes[unmatched_dets]
                left_dets_embs = boxes_embeds[unmatched_dets]
                left_trks = last_boxes[unmatched_trks]
                left_trks_embs = trk_embs[unmatched_trks]

                iou_left = self.asso_func(left_dets, left_trks)
                # TODO: is better without this
                emb_cost_left = left_dets_embs @ left_trks_embs.T
                if self.embedding_off:
                    emb_cost_left = np.zeros_like(emb_cost_left)
                iou_left = np.array(iou_left)
                if iou_left.max() > self.iou_threshold:
                    """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                    """
                    rematched_indices = linear_assignment(-iou_left)
                    to_remove_det_indices = []
                    to_remove_trk_indices = []
                    for m in rematched_indices:
                        det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue

                        self.trackers[trk_ind].update(dets[det_ind, :5], n_classes[det_ind])
                        self.trackers[trk_ind].update_emb(boxes_embeds[det_ind], alpha=detects_alpha[det_ind])
                        to_remove_det_indices.append(det_ind)
                        to_remove_trk_indices.append(trk_ind)
                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

            for m in unmatched_trks:
                self.trackers[m].update(None, None)

            # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(
                    dets[i, :5], n_classes[i], delta_t=self.delta_t, emb=boxes_embeds[i], alpha=detects_alpha[i], new_kf=not self.new_kf_off
                )
                self.trackers.append(trk)
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                if trk.last_observation.sum() < 0:
                    d = trk.get_state()[0]
                else:
                    """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                    """
                    d = trk.last_observation[:4]
                if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    # +1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
                i -= 1
                # remove dead tracklet
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i) 
            
            # now the remaining jobs are... hopefully simple.
            # assign the new values into instances?
            instances.ID = torch.as_tensor(n_ids)
            instances.pred_boxes = Boxes(torch.as_tensor(n_boxes))
            instances.classes = torch.as_tensor(n_classes)
            instances.scores = torch.as_tensor(n_scores)

        self._prev_instances = deepcopy(instances)
        return instances

    def _initialize_extra_fields(self, instances: Instances) -> Instances:
        """
        Per BaseTracker's update() function, Instances will have the following fields:
          .pred_boxes               (shape=[N, 4])
          .scores                   (shape=[N,])
          .pred_classes             (shape=[N,])
          .pred_keypoints           (shape=[N, M, 3], Optional)
          .pred_masks               (shape=List[2D_MASK], Optional)   2D_MASK: shape=[H, W]
          .ID                       (shape=[N,])

        with N being the number of detected bboxes, and H and W being the height and width, respectively, of 2D mask.

        For this tracker, if the instances are missing 'pred_boxes', 'scores', 'pred_classes', and 'ID',
        this function will create and initialize these fields.

        Contrary to existing trackers in this repo, I don't think this needs 'ID_period' - Kalam

        Args:
            instances (Instances): _description_

        Returns:
            Instances: _description_
        """

        # based on the documentations
        # the intended types for the fields are:
        #  - “pred_boxes”: Boxes object storing N boxes, one for each detected instance.
        #  - “scores”: Tensor, a vector of N confidence scores.
        #  - “pred_classes”: Tensor, a vector of N labels in range [0, num_categories).
        #  - “pred_masks”: a Tensor of shape (N, H, W), masks for each detected instance.
        #  - “pred_keypoints”: a Tensor of shape (N, num_keypoint, 3). Each row in the last dimension is (x, y, score). Confidence scores are larger than 0.

        if not instances.has("ID"):
            instances.set("ID", np.zeros((len(instances),))) # shape=[N,]
        if not instances.has("pred_boxes"):
            instances.set("pred_boxes", Boxes(torch.as_tensor(np.zeros((len(instances), 4))))) # shape=[N,4]
        if not instances.has("scores"):
            instances.set("scores", torch.as_tensor(np.zeros((len(instances),)))) # shape=[N,]
        if not instances.has('pred_classes'):
            instances.set('pred_classes', torch.as_tensor(np.zeros((len(instances)), ))) # shape=[N,]
        if not self._prev_instances:
            # this is supposed to handle the cases where no predictions are made.
            # to be honest, I am still unsure, so I'll do a first draft and see if it goes wrong
            # TODO: Reimplement this part, IF IT IS NEEDED ONLY.
            
            # to be fair, I haven't entirely understand the whole... Instances shebang.
            # TODO: Read the models again, see how they actually work.
            instances.ID = np.array(list(range(len(instances))))
            self._id_count += len(instances)
            instances.pred_boxes = Boxes(torch.as_tensor(np.zeros((len(instances), 4))))
            instances.scores = torch.as_tensor(np.zeros((len(instances), )))
            instances.pred_classes = torch.as_tensor(np.zeros((len(instances),)))
        return instances

    def _xywh_to_xyxy(self, bbox_xywh: np.ndarray):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2
    
    def _get_features(self, bbox_xywh: np.ndarray):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = self.img_feats[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.embedder(im_crops).cpu()
        else:
            features = np.array([])
        
        return features

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh

        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti(
            dets,
            trks,
            cates,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
            The re-association stage by OCR.
            NOTE: at this stage, adding other strategy might be able to continue improve
            the performance, such as BYTE association by ByteTrack.
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:, 4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                        """
                        For some datasets, such as KITTI, there are different categories,
                        we have to avoid associate them together.
                        """
                        cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if trk.time_since_update < 1:
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id + 1], [trk.cls], [trk.conf])).reshape(1, -1))
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i + 2)]
                        ret.append(
                            (
                                np.concatenate(
                                    (
                                        prev_observation[:4],
                                        [trk.id + 1],
                                        [trk.cls],
                                        [trk.conf],
                                    )
                                )
                            ).reshape(1, -1)
                        )
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()