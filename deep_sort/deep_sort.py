import numpy as np
import torch
import cv2

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)           # CNN
        # features = self._get_features_hog_paper(bbox_xywh, ori_img)   # HOG paper
        # features = self._get_features_hog(bbox_xywh, ori_img)       # HOG
        # features = self._get_features_sift(bbox_xywh, ori_img)       # SIFT
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _get_features_hog_paper(self, bbox_xywh, ori_img):

        features = np.zeros((len(bbox_xywh),256))    # 512 256 128
        for i,box in enumerate(bbox_xywh):
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            if im.shape[0]>0 and im.shape[1]>0:
                image=cv2.resize(im, (64,128))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert the original image to gray scale
                cell_size = (16, 16)                  # 512: (8, 16)  256: (16, 16) 128: (16, 32)
                num_cells_per_block = (2, 2)
                block_size = (num_cells_per_block[0] * cell_size[0],num_cells_per_block[1] * cell_size[1])
                x_cells = gray_image.shape[1] // cell_size[0]
                y_cells = gray_image.shape[0] // cell_size[1]
                h_stride = 2
                v_stride = 2
                block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
                num_bins = 8
                win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])
                hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
                feature = hog.compute(gray_image)
                features[i]=feature.reshape(256)   # 512 256 128

            else:
                features[i] = np.array([])
        return features

    def _get_features_hog(self, bbox_xywh, ori_img):

        features = np.zeros((len(bbox_xywh),512))   # 512 256 128
        for i,box in enumerate(bbox_xywh):
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            if im.shape[0]>0 and im.shape[1]>0:
                image=cv2.resize(im, (64,128))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert the original image to gray scale
                gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1)
                height = gray_image.shape[0]
                width = gray_image.shape[1]
                split_index_h = int(height / 2)
                split_index_w = int(width / 2)
                mag, ang = cv2.cartToPolar(gx, gy)
                bin_n = 128                           # 128:32 256:64 512:128
                bin = np.int32(bin_n * ang / (2 * np.pi))
                bin_cells = bin[:split_index_w, :split_index_h], bin[split_index_w:, :split_index_h], \
                            bin[:split_index_w, split_index_h:], bin[split_index_w:, split_index_h:]
                mag_cells = mag[:split_index_w, :split_index_h], mag[split_index_w:, :split_index_h], \
                            mag[:split_index_w, split_index_h:], mag[split_index_w:, split_index_h:]
                hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
                hist = np.hstack(hists)
                # transform to Hellinger kernel
                eps = 1e-7
                hist /= hist.sum() + eps
                hist = np.sqrt(hist)
                hist /= np.linalg.norm(hist) + eps
                feature = np.float32(hist)
                features[i]=feature.reshape(512)    # 512 256 128

            else:
                features[i] = np.array([])
        return features

    def _get_features_sift(self, bbox_xywh, ori_img):

        features = np.zeros((len(bbox_xywh),512))    # 512 256 128
        for i,box in enumerate(bbox_xywh):
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            if im.shape[0]>0 and im.shape[1]>0:
                image=cv2.resize(im, (64,128))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert the original image to gray scale
                sift = cv2.xfeatures2d_SIFT.create(4)
                if sift:
                    kp,des = sift.detectAndCompute(image, None)    # des=4*128
                    feature=np.hstack(des)
                    print(feature.shape)
                    features[i]=feature[:512]        # 512 256 128
                else:
                    continue


            else:
                features[i] = np.array([])
        return features
