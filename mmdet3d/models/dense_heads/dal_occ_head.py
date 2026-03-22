import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from .dal_head import DALHead


@HEADS.register_module()
class DALOccHead(DALHead):
    """First-pass DAL head with sparse pillar occupancy branch.

    Design constraints:
        1. box regression remains LiDAR-only;
        2. occupancy only enhances shared BEV / dense heatmap / classification;
        3. no occupancy feature is injected into regression.
    """

    def __init__(self,
                 occ_enabled=True,
                 occ_num_classes=2,
                 occ_z_bins=4,
                 occ_topk_ratio=0.1,
                 occ_prop_threshold=0.3,
                 occ_use_gt_mask=True,
                 occ_feedback='none',
                 occ_prop_weight=1.0,
                 occ_detach_feedback=False,
                 loss_occ_proposal=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     reduction='mean',
                     loss_weight=1.0),
                 loss_occ=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        super().__init__(**kwargs)
        self.occ_enabled = occ_enabled
        self.occ_num_classes = occ_num_classes
        self.occ_z_bins = occ_z_bins
        self.occ_topk_ratio = occ_topk_ratio
        self.occ_prop_threshold = occ_prop_threshold
        self.occ_use_gt_mask = occ_use_gt_mask
        self.occ_feedback = occ_feedback
        self.occ_prop_weight = occ_prop_weight
        self.occ_detach_feedback = occ_detach_feedback
        self.loss_occ_proposal = build_loss(loss_occ_proposal)
        self.loss_occ = build_loss(loss_occ)

        if not self.occ_enabled:
            return

        hidden_channel = kwargs['hidden_channel']
        self.occ_bev_proposal_head = nn.Sequential(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d')),
            nn.Conv2d(hidden_channel, 1, kernel_size=1))
        self.occ_bev_feedback = ConvModule(
            hidden_channel + self.occ_z_bins,
            hidden_channel,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'))
        self.occ_heatmap_feedback = ConvModule(
            self.num_classes + self.occ_z_bins,
            self.num_classes,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'))
        self.occ_cls_feedback = ConvModule(
            hidden_channel + self.occ_z_bins,
            hidden_channel,
            kernel_size=1,
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'))
        self.occ_z_embedding = nn.Embedding(self.occ_z_bins, hidden_channel)
        self.occ_decoder = nn.Sequential(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=1,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d')),
            nn.Conv1d(hidden_channel, occ_num_classes, kernel_size=1))

        for module in [
                self.occ_bev_proposal_head,
                self.occ_bev_feedback,
                self.occ_heatmap_feedback,
                self.occ_cls_feedback]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

    def _get_num_occ_cells(self, total_cells):
        num_cells = max(1, int(total_cells * self.occ_topk_ratio))
        return min(total_cells, num_cells)

    def _build_occ_targets(self, voxel_semantics, out_h, out_w, device):
        semantics = voxel_semantics.long().to(device)
        occupied = (semantics > 0).float()
        occupied_bev = occupied.max(dim=-1).values
        occ_prop_target = F.adaptive_max_pool2d(occupied_bev.unsqueeze(1),
                                                (out_h, out_w))

        occ_volume = occupied.permute(0, 3, 1, 2).contiguous().float()
        occ_bin_target = F.adaptive_max_pool3d(
            occ_volume.unsqueeze(1), (self.occ_z_bins, out_h, out_w)).squeeze(1)
        occ_bin_target = occ_bin_target.permute(0, 2, 3, 1).contiguous()
        return occ_prop_target, occ_bin_target.long()

    def _pool_mask_to_occ(self, mask_camera, out_h, out_w, device):
        if mask_camera is None:
            return None
        mask = mask_camera.to(device=device, dtype=torch.float32)
        mask = mask.permute(0, 3, 1, 2).contiguous()
        mask = F.adaptive_max_pool3d(
            mask.unsqueeze(1), (self.occ_z_bins, out_h, out_w)).squeeze(1)
        mask = mask.permute(0, 2, 3, 1).contiguous()
        return mask

    def _sparse_pillar_forward(self, bev_feat_lidar, occ_prop_logits):
        b, _, h, w = occ_prop_logits.shape
        proposal_score = occ_prop_logits.sigmoid().view(b, -1)
        num_select = self._get_num_occ_cells(proposal_score.shape[-1])
        topk_score, topk_index = proposal_score.topk(num_select, dim=-1)
        valid_mask = topk_score > self.occ_prop_threshold
        if valid_mask.sum() == 0:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)

        bev_feat_flat = bev_feat_lidar.view(b, bev_feat_lidar.shape[1], -1)
        pillar_feat = bev_feat_flat.gather(
            dim=-1,
            index=topk_index[:, None, :].expand(-1, bev_feat_flat.shape[1], -1))

        z_embed = self.occ_z_embedding.weight.t()[None, :, :, None]
        z_embed = z_embed.expand(b, -1, -1, pillar_feat.shape[-1])
        pillar_feat = pillar_feat.unsqueeze(2) + z_embed.permute(0, 2, 1, 3)
        pillar_feat = pillar_feat.reshape(b * self.occ_z_bins, -1,
                                          pillar_feat.shape[-1])

        sparse_occ_logits = self.occ_decoder(pillar_feat)
        sparse_occ_logits = sparse_occ_logits.view(b, self.occ_z_bins,
                                                   self.occ_num_classes, -1)
        sparse_occ_logits = sparse_occ_logits.permute(0, 2, 1, 3).contiguous()

        dense_occ_logits = sparse_occ_logits.new_zeros(
            (b, self.occ_num_classes, self.occ_z_bins, h * w))
        scatter_index = topk_index[:, None, None, :].expand(
            -1, self.occ_num_classes, self.occ_z_bins, -1)
        dense_occ_logits.scatter_(dim=-1, index=scatter_index,
                                  src=sparse_occ_logits)
        dense_occ_logits = dense_occ_logits.view(b, self.occ_num_classes,
                                                 self.occ_z_bins, h, w)
        sparse_occ_prob = sparse_occ_logits.softmax(dim=1)[:, 1]
        occ_feedback = sparse_occ_prob.new_zeros((b, self.occ_z_bins, h * w))
        occ_feedback.scatter_(dim=-1, index=topk_index[:, None, :].expand(
            -1, self.occ_z_bins, -1), src=sparse_occ_prob)
        occ_feedback = occ_feedback.view(b, self.occ_z_bins, h, w)
        return dense_occ_logits, occ_feedback, topk_index, valid_mask

    def forward_single(self, inputs, img_inputs, bev_feat_img=None):
        batch_size = inputs.shape[0]
        bev_feat_lidar = self.shared_conv(inputs)
        bev_feat_lidar_flatten = bev_feat_lidar.view(batch_size,
                                                     bev_feat_lidar.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(bev_feat_lidar.device)

        dense_fuse_feat = torch.cat([bev_feat_lidar, bev_feat_img], dim=1)
        dense_fuse_feat = self.dense_heatmap_fuse_convs(dense_fuse_feat)[0]

        occ_feedback_for_det = None
        occ_outputs = dict()
        if self.occ_enabled:
            occ_prop_logits = self.occ_bev_proposal_head(bev_feat_lidar)
            occ_logits, occ_feedback, occ_topk_index, occ_valid_mask = \
                self._sparse_pillar_forward(bev_feat_lidar, occ_prop_logits)
            occ_feedback_for_det = occ_feedback.detach() \
                if self.occ_detach_feedback else occ_feedback
            occ_outputs.update(
                occ_prop_logits=occ_prop_logits,
                occ_logits=occ_logits,
                occ_topk_index=occ_topk_index,
                occ_valid_mask=occ_valid_mask)
            if self.occ_feedback == 'bev':
                dense_fuse_feat = self.occ_bev_feedback(
                    torch.cat([dense_fuse_feat, occ_feedback_for_det], dim=1))

        dense_heatmap = self.heatmap_head(dense_fuse_feat)
        if self.occ_enabled and self.occ_feedback == 'heatmap':
            dense_heatmap = self.occ_heatmap_feedback(
                torch.cat([dense_heatmap, occ_feedback_for_det], dim=1))
        heatmap = dense_heatmap.detach().sigmoid()

        top_proposals_class, top_proposals_index = self.extract_proposal(heatmap)
        self.query_labels = top_proposals_class

        index = top_proposals_index.expand(-1, bev_feat_lidar_flatten.shape[1], -1)
        query_feat_lidar = bev_feat_lidar_flatten.gather(index=index, dim=-1)

        one_hot = F.one_hot(top_proposals_class,
                            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat_lidar += query_cat_encoding

        query_pos_index = top_proposals_index.permute(0, 2, 1)
        query_pos_index = query_pos_index.expand(-1, -1, bev_pos.shape[-1])
        query_pos = bev_pos.gather(index=query_pos_index, dim=1)

        res = dict()
        for task in ['height', 'center', 'dim', 'rot', 'vel']:
            res[task] = self.prediction_heads[0].__getattr__(task)(query_feat_lidar)
        res['center'] += query_pos.permute(0, 2, 1)

        query_feat_img = self.extract_instance_img_feat(res, img_inputs)

        bev_feat_img = bev_feat_img.view(batch_size, bev_feat_img.shape[1], -1)
        index = top_proposals_index.expand(-1, bev_feat_img.shape[1], -1)
        query_feat_img_bev = bev_feat_img.gather(index=index, dim=-1)

        query_feat_fuse = torch.cat(
            [query_feat_lidar, query_feat_img, query_feat_img_bev], dim=1)
        query_feat_fuse = self.fuse_convs(query_feat_fuse)
        if self.occ_enabled and self.occ_feedback == 'cls':
            occ_query_feat = occ_feedback_for_det.view(batch_size,
                                                       self.occ_z_bins, -1)
            occ_query_feat = occ_query_feat.gather(
                dim=-1,
                index=top_proposals_index.expand(-1, self.occ_z_bins, -1))
            query_feat_fuse = self.occ_cls_feedback(
                torch.cat([query_feat_fuse, occ_query_feat], dim=1))

        res['heatmap'] = self.prediction_heads[0].__getattr__('heatmap')(
            query_feat_fuse)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        res['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index.expand(-1, self.num_classes, -1), dim=-1)
        res['dense_heatmap'] = dense_heatmap
        res.update(occ_outputs)
        return [res]

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self,
             gt_bboxes_3d,
             gt_labels_3d,
             preds_dicts,
             img_metas=None,
             voxel_semantics=None,
             mask_camera=None,
             **kwargs):
        loss_dict = super().loss(
            gt_bboxes_3d, gt_labels_3d, preds_dicts, img_metas=img_metas, **kwargs)
        if not self.occ_enabled or voxel_semantics is None:
            return loss_dict

        preds_dict = preds_dicts[0][0]
        occ_prop_logits = preds_dict['occ_prop_logits']
        occ_logits = preds_dict['occ_logits']
        _, _, _, out_h, out_w = occ_logits.shape

        occ_prop_target, occ_bin_target = self._build_occ_targets(
            voxel_semantics, out_h, out_w, occ_prop_logits.device)
        occ_mask = self._pool_mask_to_occ(mask_camera, out_h, out_w,
                                          occ_prop_logits.device)

        loss_occ_prop = self.loss_occ_proposal(
            occ_prop_logits.reshape(-1, 1),
            occ_prop_target.reshape(-1, 1),
            avg_factor=max(float(occ_prop_target.sum().item()), 1.0))

        occ_target = occ_bin_target.reshape(-1)
        occ_logits = occ_logits.permute(0, 3, 4, 2, 1).reshape(
            -1, self.occ_num_classes)
        occ_weight = occ_prop_target.expand(-1, self.occ_z_bins, -1, -1).permute(
            0, 2, 3, 1).reshape(-1)
        if self.occ_use_gt_mask and occ_mask is not None:
            occ_weight = occ_weight * occ_mask.reshape(-1)

        loss_dict['loss_occ_prop'] = self.occ_prop_weight * loss_occ_prop
        loss_dict['loss_occ'] = self.loss_occ(
            occ_logits,
            occ_target,
            occ_weight,
            avg_factor=max(float(occ_weight.sum().item()), 1.0))
        return loss_dict
