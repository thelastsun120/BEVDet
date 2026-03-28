from mmdet.models import DETECTORS

from .dal import DAL


@DETECTORS.register_module()
class DALOcc(DAL):
    """DAL detector with occupancy branch.

    This wrapper only changes loss wiring so occupancy supervision can be
    passed into the DALOccHead while keeping DAL training flow unchanged.
    """

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          voxel_semantics=None,
                          mask_camera=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs,
            gt_bboxes_ignore=gt_bboxes_ignore,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera)
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = self.img_view_transformer(
            img_feats + img_inputs[1:7], depth_from_lidar=kwargs['gt_depth'])

        losses = dict()
        losses_pts = self.forward_pts_train(
            [img_feats, pts_feats, img_feats_bev],
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera)
        losses.update(losses_pts)
        losses_img_auxiliary = self.forward_img_auxiliary_train(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_bboxes_ignore,
            **kwargs)
        losses.update(losses_img_auxiliary)
        return losses
