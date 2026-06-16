# -*- coding: utf-8 -*-
"""
FroFA-style feature augmentation for episodic few-shot classification.

This module implements the core idea from "Frozen Feature Augmentation for
Few-Shot Image Classification" (CVPR 2024) inside LibFewShot's metric-learning
interface: normalize support features into a bounded feature space, apply simple
feature-space augmentations, map them back, and classify queries with augmented
class prototypes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class FroFAProtoNet(MetricModel):
    def __init__(
        self,
        augmentations=("brightness",),
        alpha=0.2,
        num_aug=2,
        distance="euclidean",
        freeze_emb_func=False,
        learnable_scale=True,
        eps=1e-6,
        **kwargs
    ):
        super(FroFAProtoNet, self).__init__(**kwargs)
        if isinstance(augmentations, str):
            augmentations = [augmentations]
        self.augmentations = list(augmentations)
        self.alpha = float(alpha)
        self.num_aug = int(num_aug)
        self.distance = distance
        self.freeze_emb_func = freeze_emb_func
        self.eps = eps
        self.loss_func = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.tensor(0.0), requires_grad=learnable_scale)

        if self.freeze_emb_func:
            for param in self.emb_func.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(FroFAProtoNet, self).train(mode)
        if self.freeze_emb_func:
            self.emb_func.eval()
        return self

    def _extract_features(self, images):
        if self.freeze_emb_func:
            with torch.no_grad():
                return self.emb_func(images)
        return self.emb_func(images)

    def _to_unit_range(self, support_feat):
        feat_min = support_feat.amin(dim=(1, 2), keepdim=True)
        feat_max = support_feat.amax(dim=(1, 2), keepdim=True)
        scale = (feat_max - feat_min).clamp_min(self.eps)
        return (support_feat - feat_min) / scale, feat_min, scale

    def _sample_strengths(self, count, device, dtype):
        if self.training:
            return (torch.rand(count, device=device, dtype=dtype) * 2.0 - 1.0) * self.alpha
        if count == 1:
            return torch.tensor([self.alpha], device=device, dtype=dtype)
        return torch.linspace(-self.alpha, self.alpha, count, device=device, dtype=dtype)

    def _apply_one_aug(self, feat01, aug_name, strength):
        if aug_name == "brightness":
            return (feat01 + strength).clamp(0.0, 1.0)
        if aug_name == "contrast":
            center = feat01.mean(dim=-1, keepdim=True)
            factor = (1.0 + strength).clamp_min(0.05)
            return ((feat01 - center) * factor + center).clamp(0.0, 1.0)
        if aug_name == "posterize":
            levels = max(2, int(round(8 - float(torch.abs(strength).detach().cpu()) * 6)))
            bins = float((2 ** levels) - 1)
            return torch.round(feat01 * bins) / bins
        raise ValueError("Unsupported FroFA augmentation: {}".format(aug_name))

    def _augment_support(self, support_feat):
        episode_size, _, channels = support_feat.size()
        support_feat = support_feat.view(episode_size, self.way_num, self.shot_num, channels)
        if self.num_aug <= 0 or len(self.augmentations) == 0:
            return support_feat

        support01, feat_min, scale = self._to_unit_range(support_feat)
        augmented = [support_feat]
        strengths = self._sample_strengths(
            self.num_aug, support_feat.device, support_feat.dtype
        )

        for aug_name in self.augmentations:
            for strength in strengths:
                aug01 = self._apply_one_aug(support01, aug_name, strength)
                augmented.append(aug01 * scale + feat_min)

        return torch.cat(augmented, dim=2)

    def _classify(self, support_feat, query_feat):
        episode_size, query_count, channels = query_feat.size()
        support_aug = self._augment_support(support_feat)
        proto_feat = support_aug.mean(dim=2)
        query_feat = query_feat.view(episode_size, self.way_num * self.query_num, channels)

        if self.distance == "cos_sim":
            logits = torch.matmul(
                F.normalize(query_feat, p=2, dim=-1),
                F.normalize(proto_feat, p=2, dim=-1).transpose(-1, -2),
            )
        else:
            logits = -torch.sum(
                torch.pow(query_feat.unsqueeze(2) - proto_feat.unsqueeze(1), 2),
                dim=3,
            )
        return logits * self.logit_scale.exp()

    def set_forward(self, batch):
        images, _ = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self._extract_features(images)
        support_feat, query_feat, _, query_target = self.split_by_episode(feat, mode=1)
        output = self._classify(support_feat, query_feat).reshape(
            episode_size * self.way_num * self.query_num, self.way_num
        )
        acc = accuracy(output, query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        images, _ = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self._extract_features(images)
        support_feat, query_feat, _, query_target = self.split_by_episode(feat, mode=1)
        output = self._classify(support_feat, query_feat).reshape(
            episode_size * self.way_num * self.query_num, self.way_num
        )
        target = query_target.reshape(-1)
        loss = self.loss_func(output, target)
        acc = accuracy(output, target)
        return output, acc, loss
