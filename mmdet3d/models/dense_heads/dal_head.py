import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn
from .transfusion_head import TransFusionHead
from mmdet3d.models.builder import HEADS
from .. import builder

__all__ = ["DALHead"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class LocalSelfAttention2D(nn.Module):
    """Lightweight local self-attention in BEV using local depthwise context."""

    def __init__(self, channels):
        super(LocalSelfAttention2D, self).__init__()
        self.q_proj = build_conv_layer(
            dict(type='Conv2d'), channels, channels, kernel_size=1, stride=1, padding=0)
        self.k_proj = build_conv_layer(
            dict(type='Conv2d'),
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels)
        self.v_proj = build_conv_layer(
            dict(type='Conv2d'), channels, channels, kernel_size=1, stride=1, padding=0)
        self.out_proj = build_conv_layer(
            dict(type='Conv2d'), channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.sigmoid(q * k)
        out = self.out_proj(attn * v)
        return x + out


@HEADS.register_module()
class DALHead(TransFusionHead):
    def __init__(self,
                 img_feat_dim=128,
                 feat_bev_img_dim=32,
                 sparse_fuse_layers=2,
                 dense_fuse_layers=2,
                 use_spar_p=False,
                 spar_p_alpha=0.5,
                 spar_p_mid_channels=None,
                 spar_p_learnable_alpha=False,
                 spar_p_use_local_attn=False,
                 use_spar_c=False,
                 spar_c_kernel=3,
                 spar_c_pool_modes=('max', 'avg'),
                 spar_c_ctx_dim=32,
                 spar_c_mlp_layers=2,
                 proposal_use_threshold_topk=False,
                 proposal_score_threshold=0.0,
                 **kwargs):
        super(DALHead, self).__init__(**kwargs)
        self.use_spar_p = use_spar_p
        self.spar_p_mid_channels = spar_p_mid_channels or kwargs['hidden_channel']
        self.spar_p_learnable_alpha = spar_p_learnable_alpha
        self.spar_p_use_local_attn = spar_p_use_local_attn
        if self.spar_p_learnable_alpha:
            self.spar_p_alpha = nn.Parameter(torch.tensor(float(spar_p_alpha)))
        else:
            self.spar_p_alpha = float(spar_p_alpha)
        self.use_spar_c = use_spar_c
        self.spar_c_kernel = spar_c_kernel
        self.spar_c_pool_modes = tuple(spar_c_pool_modes)
        self.spar_c_ctx_dim = spar_c_ctx_dim
        self.spar_c_mlp_layers = spar_c_mlp_layers
        self.proposal_use_threshold_topk = proposal_use_threshold_topk
        self.proposal_score_threshold = proposal_score_threshold

        # fuse net for first stage dense prediction
        cfg = dict(
            type='CustomResNet',
            numC_input=kwargs['hidden_channel'] + feat_bev_img_dim,
            num_layer=[dense_fuse_layers+1, ],
            num_channels=[kwargs['hidden_channel'], ],
            stride=[1, ],
            backbone_output_ids=[0, ])
        self.dense_heatmap_fuse_convs = builder.build_backbone(cfg)

        if self.use_spar_p:
            self.spar_p_refiner = nn.Sequential(
                ConvModule(
                    kwargs['hidden_channel'],
                    self.spar_p_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias='auto',
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d')),
                ConvModule(
                    self.spar_p_mid_channels,
                    self.spar_p_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias='auto',
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d')))
            self.spar_p_residual = nn.Sequential(
                ConvModule(
                    self.spar_p_mid_channels,
                    self.spar_p_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias='auto',
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d')))
            if self.spar_p_use_local_attn:
                self.spar_p_local_attn = LocalSelfAttention2D(self.spar_p_mid_channels)
            self.spar_p_out = build_conv_layer(
                dict(type='Conv2d'),
                self.spar_p_mid_channels,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)

        # fuse net for second stage sparse prediction
        fuse_convs = []
        c_in = img_feat_dim + kwargs['hidden_channel'] + feat_bev_img_dim
        if self.use_spar_c:
            if not set(self.spar_c_pool_modes).issubset({'max', 'avg'}):
                raise ValueError('spar_c_pool_modes only supports max/avg')
            pooled_feat_scale = len(self.spar_c_pool_modes)
            if self.spar_c_mlp_layers < 1:
                raise ValueError('spar_c_mlp_layers should be >= 1')
            ctx_layers = []
            c_in_ctx = kwargs['hidden_channel'] * pooled_feat_scale
            for _ in range(self.spar_c_mlp_layers):
                ctx_layers.append(
                    ConvModule(
                        c_in_ctx,
                        self.spar_c_ctx_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias='auto',
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d')))
                c_in_ctx = self.spar_c_ctx_dim
            self.spar_c_ctx_proj = nn.Sequential(*ctx_layers)
            c_in += self.spar_c_ctx_dim
        for i in range(sparse_fuse_layers - 1):
            fuse_convs.append(
                ConvModule(
                    c_in,
                    c_in,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias='auto',
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type="BN1d")))
        fuse_convs.append(
            ConvModule(
                c_in,
                kwargs['hidden_channel'],
                kernel_size=1,
                stride=1,
                padding=0,
                bias='auto',
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type="BN1d")))
        self.fuse_convs = nn.Sequential(*fuse_convs)
        self._init_weights()

    def _init_weights(self):
        for m in self.dense_heatmap_fuse_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        if self.use_spar_p:
            for m in self.spar_p_refiner.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.spar_p_residual.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            if self.spar_p_use_local_attn:
                for m in self.spar_p_local_attn.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
            kaiming_init(self.spar_p_out)

    def extract_local_context_feat(self, bev_feat, top_proposals_index):
        """Extract local BEV context features for top-k proposals."""
        batch_size, channel, _, _ = bev_feat.shape
        kernel = self.spar_c_kernel
        if kernel % 2 == 0:
            raise ValueError('spar_c_kernel should be odd.')
        gather_index = top_proposals_index.unsqueeze(1).expand(
            -1, channel, -1)  # [B,C,num_proposals]
        pooled_ctx = []
        if 'max' in self.spar_c_pool_modes:
            max_map = F.max_pool2d(
                bev_feat, kernel_size=kernel, stride=1, padding=kernel // 2)
            max_map = max_map.view(batch_size, channel, -1)
            pooled_ctx.append(max_map.gather(index=gather_index, dim=-1))
        if 'avg' in self.spar_c_pool_modes:
            avg_map = F.avg_pool2d(
                bev_feat, kernel_size=kernel, stride=1, padding=kernel // 2)
            avg_map = avg_map.view(batch_size, channel, -1)
            pooled_ctx.append(avg_map.gather(index=gather_index, dim=-1))
        if not pooled_ctx:
            raise ValueError('spar_c_pool_modes can not be empty.')
        ctx_feat = torch.cat(pooled_ctx, dim=1)
        return self.spar_c_ctx_proj(ctx_feat)

    @force_fp32()
    def extract_img_feat_from_3dpoints(self, points, img_inputs_list, fuse=True):
        if not isinstance(img_inputs_list[0], list):
            img_inputs_list = [img_inputs_list]
        global2keyego = torch.inverse(img_inputs_list[0][2][:,0,:,:].unsqueeze(1).to(torch.float64))
        point_img_feat_list = []

        b, p, _ = points.shape
        points = points.view(b, 1, -1, 3, 1)
        for img_inputs in img_inputs_list:
            img_feats = img_inputs[0].permute(0, 2, 1, 3, 4).contiguous()
            _, c, n, h, w = img_feats.shape
            with torch.no_grad():

                sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = \
                    img_inputs[1:]
                currego2global = ego2global[:,0,:,:].unsqueeze(1).to(torch.float64)
                currego2keyego = global2keyego.matmul(currego2global).to(torch.float32)

                # aug ego to cam
                augego2cam = torch.inverse(bda.view(b, 1, 4, 4).matmul(currego2keyego).matmul(sensor2ego))
                augego2cam = augego2cam.view(b, -1, 1, 4, 4)
                points_cam = augego2cam[..., :3, :3].matmul(points)
                points_cam += augego2cam[:, :, :, :3, 3:4]

                valid = points_cam[..., 2, 0] > 0.5
                points_img = points_cam/points_cam[..., 2:3, :]
                points_img = cam2imgs.view(b, -1, 1, 3, 3).matmul(points_img)

                points_img_x = points_img[..., 0, 0]
                points_img_x = points_img_x * valid
                select_cam_ids = \
                    torch.argmin(torch.abs(points_img_x -
                                           cam2imgs[:, :, 0, 2:3]), dim=1)

                points_img = post_rots.view(b, -1, 1, 3, 3).matmul(points_img) + \
                             post_trans.view(b, -1, 1, 3, 1)

                points_img[..., 2, 0] = points_cam[..., 2, 0]

                points_img = points_img[..., :2, 0]
                index = select_cam_ids[:, None, :, None].expand(-1, -1, -1, 2)
                points_img_selected = \
                    points_img.gather(index=index, dim=1).squeeze(1)

                # img space to feature space
                points_img_selected /= self.test_cfg['img_feat_downsample']

                grid = torch.cat([points_img_selected,
                                  select_cam_ids.unsqueeze(-1)], dim=2)

                normalize_factor = torch.tensor([w - 1.0, h - 1.0, n - 1.0]).to(grid)
                grid = grid / normalize_factor.view(1, 1, 3) * 2.0 - 1.0
                grid = grid.view(b, p, 1, 1, 3)
            point_img_feat = \
                F.grid_sample(img_feats, grid,
                              mode='bilinear',
                              align_corners=True).view(b,c,p)
            point_img_feat_list.append(point_img_feat)
        if not fuse:
            point_img_feat = point_img_feat_list[0]
        else:
            point_img_feat = point_img_feat_list
        return point_img_feat

    def extract_instance_img_feat(self, res_layer, img_inputs, fuse=False):
        center = res_layer["center"]
        height = res_layer["height"]
        center_x = center[:, 0:1, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[0] + self.bbox_coder.pc_range[0]
        center_y = center[:, 1:2, :] * self.bbox_coder.out_size_factor * \
                   self.bbox_coder.voxel_size[1] + self.bbox_coder.pc_range[1]

        ref_points = torch.cat([center_x, center_y, height], dim=1).permute(0, 2, 1)

        img_feat = self.extract_img_feat_from_3dpoints(ref_points, img_inputs, fuse=fuse)
        return img_feat

    def extract_proposal(self, heatmap):
        batch_size = heatmap.shape[0]
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, stride=1, padding=0,
                                       kernel_size=self.nms_kernel_size)
        local_max[:, :, padding:(-padding), padding:(-padding)] = \
            local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[:, 8,] = F.max_pool2d(heatmap[:, 8], kernel_size=1,
                                            stride=1, padding=0)
            local_max[:, 9,] = F.max_pool2d(heatmap[:, 9], kernel_size=1,
                                            stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":
            # for Pedestrian & Cyclist in Waymo
            local_max[:, 1,] = F.max_pool2d(heatmap[:, 1], kernel_size=1,
                                            stride=1, padding=0)
            local_max[:, 2,] = F.max_pool2d(heatmap[:, 2], kernel_size=1,
                                            stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        all_proposals = heatmap.view(batch_size, -1)
        if self.proposal_use_threshold_topk and self.proposal_score_threshold > 0:
            global_topk = all_proposals.argsort(dim=-1, descending=True)[..., :self.num_proposals]
            threshold_topk = []
            for b in range(batch_size):
                valid_index = torch.nonzero(
                    all_proposals[b] >= self.proposal_score_threshold,
                    as_tuple=False).squeeze(-1)
                if valid_index.numel() == 0:
                    top_idx = global_topk[b]
                else:
                    merged_index = torch.cat([valid_index, global_topk[b]], dim=0)
                    merged_index = torch.unique(merged_index)
                    merged_score = all_proposals[b].gather(dim=0, index=merged_index)
                    merged_order = merged_score.argsort(descending=True)
                    top_idx = merged_index[merged_order][:self.num_proposals]
                    if top_idx.numel() < self.num_proposals:
                        pad_idx = global_topk[b][:self.num_proposals - top_idx.numel()]
                        top_idx = torch.cat([top_idx, pad_idx], dim=0)
                threshold_topk.append(top_idx)
            top_proposals = torch.stack(threshold_topk, dim=0)
        else:
            top_proposals = all_proposals.argsort(dim=-1, descending=True)
            top_proposals = top_proposals[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        top_proposals_index = top_proposals_index.unsqueeze(1)
        return top_proposals_class, top_proposals_index

    def forward_single(self, inputs, img_inputs, bev_feat_img=None):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]

        bev_feat_lidar = self.shared_conv(inputs)
        bev_feat_lidar_flatten = bev_feat_lidar.view(batch_size, bev_feat_lidar.shape[1], -1)  # [BS, C, H*W]

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(bev_feat_lidar.device)

        # predict dense heatmap
        dense_fuse_feat = torch.cat([bev_feat_lidar, bev_feat_img],
                                             dim=1)
        dense_fuse_feat = \
            self.dense_heatmap_fuse_convs(dense_fuse_feat)[0]
        dense_heatmap = self.heatmap_head(dense_fuse_feat)
        if self.use_spar_p:
            spar_feat = self.spar_p_refiner(dense_fuse_feat)
            spar_feat = F.relu(self.spar_p_residual(spar_feat) + spar_feat)
            if self.spar_p_use_local_attn:
                spar_feat = self.spar_p_local_attn(spar_feat)
            spar_prior_map = self.spar_p_out(spar_feat)
            alpha = self.spar_p_alpha.sigmoid() if self.spar_p_learnable_alpha \
                else self.spar_p_alpha
            dense_heatmap = dense_heatmap + alpha * spar_prior_map
        heatmap = dense_heatmap.detach().sigmoid()

        # generate proposal
        top_proposals_class, top_proposals_index = self.extract_proposal(heatmap)
        self.query_labels = top_proposals_class

        # prepare sparse lidar feat of proposal
        index = top_proposals_index.expand(-1, bev_feat_lidar_flatten.shape[1],
                                           -1)
        query_feat_lidar = bev_feat_lidar_flatten.gather(index=index, dim=-1)

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat_lidar += query_cat_encoding

        query_pos_index = top_proposals_index.permute(0, 2, 1)
        query_pos_index = query_pos_index.expand(-1, -1, bev_pos.shape[-1])
        query_pos = bev_pos.gather(index=query_pos_index, dim=1)

        # Prediction
        res = dict()
        for task in ['height', 'center', 'dim', 'rot', 'vel']:
            res[task] = \
                self.prediction_heads[0].__getattr__(task)(query_feat_lidar)
        res['center'] += query_pos.permute(0, 2, 1)

        # generate sparse fuse feat
        query_feat_img = self.extract_instance_img_feat(res, img_inputs)

        bev_feat_img = bev_feat_img.view(batch_size, bev_feat_img.shape[1], -1)
        index = top_proposals_index.expand(-1, bev_feat_img.shape[1], -1)
        query_feat_img_bev = bev_feat_img.gather(index=index, dim=-1)

        query_feat_fuse = torch.cat([query_feat_lidar, query_feat_img,
                                     query_feat_img_bev], dim=1)
        if self.use_spar_c:
            query_feat_ctx = self.extract_local_context_feat(
                dense_fuse_feat, top_proposals_index.squeeze(1))
            query_feat_fuse = torch.cat([query_feat_fuse, query_feat_ctx], dim=1)
        query_feat_fuse = self.fuse_convs(query_feat_fuse)
        res['heatmap'] = \
            self.prediction_heads[0].__getattr__('heatmap')(query_feat_fuse)

        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        res["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index.expand(-1,  self.num_classes, -1),
            dim=-1)  # [bs, num_classes, num_proposals]
        res["dense_heatmap"] = dense_heatmap

        return [res]

    def forward(self, feats):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        return [self.forward_single(feats[1][0], feats[0], feats[2][0])]
