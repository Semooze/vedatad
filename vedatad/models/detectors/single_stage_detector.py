import torch
import torch.nn as nn

from vedacore.misc import registry
from ..builder import build_backbone, build_head, build_neck
from .base_detector import BaseDetector
import numpy as np
import os
from pathlib import Path

@registry.register_module('detector')
class SingleStageDetector(BaseDetector):

    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.head = build_head(head)

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.head.init_weights()

    def forward_impl(self, x, video_metas):
        print("Features")

        # feats = self.backbone(x)
        # if self.neck:
        #     feats = self.neck(feats)
        # for idx, m_data in enumerate(video_metas):
        #     folder_name = "/home/semooze/Products/is-project/vedatad/features/train/rgb"
        #     file_name = f"{m_data['ori_video_name']}.npy"
        #     print(m_data['ori_video_name'], file_name)
        #     np.save(os.path.join(folder_name, file_name), feats[idx].permute(1,0).data.cpu().numpy())
        # feats = self.head(feats)

        # For test augmentation only not train
        folder_name = "/home/semooze/Products/is-project/vedatad/features/test/rgb"
        file_name = f"{video_metas[0][0]['ori_video_name']}.npy"
        path = os.path.join(folder_name, file_name)
        # print(file_name, len(x))
        isFile = os.path.isfile(path)
        if isFile:
            return x
        print(file_name, len(x))
        feats = [ self.backbone(overlap_img) for overlap_img in x ]
        if self.neck:
            feats = [ self.neck(frame).squeeze(0).permute(1, 0) for frame in feats ]
        merged_feats = torch.cat(feats)
        np.save(path, merged_feats.data.cpu().numpy())
        return feats

    def forward(self, x, video_metas, train=True):
        if train:
            self.eval()
            with torch.no_grad():
                feats = self.forward_impl(x, video_metas)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_impl(x, video_metas)
        return feats
