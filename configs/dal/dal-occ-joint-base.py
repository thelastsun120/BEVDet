_base_ = ['./dal-occ-base.py']

# Joint DAL + sparse occupancy training config.
# This file keeps DAL detection pipeline and adds occupancy supervision keys.

dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        opencv_pp=True,
        data_config=_base_.data_config),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(type='ObjectSample', db_sampler=_base_.db_sampler),
    dict(type='VelocityAug'),
    dict(
        type='BEVAug',
        bda_aug_conf=_base_.bda_aug_conf,
        classes=_base_.class_names),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=_base_.grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=_base_.point_cloud_range),
    dict(type='ObjectNameFilter', classes=_base_.class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=_base_.class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'img_inputs', 'gt_depth', 'gt_bboxes_ignore',
            'voxel_semantics', 'mask_camera', 'mask_lidar'
        ])
]

# Keep test pipeline aligned with DAL detection evaluation first.
test_pipeline = _base_.test_pipeline

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=_base_.class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            img_info_prototype='bevdet',
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=_base_.class_names,
        modality=input_modality,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
        img_info_prototype='bevdet',
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=_base_.class_names,
        modality=input_modality,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
        img_info_prototype='bevdet',
        box_type_3d='LiDAR'))

