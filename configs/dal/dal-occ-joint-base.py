_base_ = ['./dal-occ-base.py']

# Joint DAL + sparse occupancy training config.
# This file keeps DAL detection pipeline and adds occupancy supervision keys.

dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# Keep these values local in this config to avoid Python-level `_base_` access
# during config import.
point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 5,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.44),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'random_crop_height': True,
    'vflip': True,
    'resize_test': 0.04,
    'pmd': dict(
        brightness_delta=32,
        contrast_lower=0.5,
        contrast_upper=1.5,
        saturation_lower=0.5,
        saturation_upper=1.5,
        hue_delta=18,
        rate=0.5)
}

grid_config = {
    'x': [-54.0, 54.0, 0.6],
    'y': [-54.0, 54.0, 0.6],
    'z': [-3, 5, 8],
    'depth': [1.0, 60.0, 0.5],
}

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5])

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'bevdetv3-nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

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
        data_config=data_config),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='VelocityAug'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='PointToMultiViewDepthFusion', downsample=1, grid_config=grid_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d',
            'img_inputs', 'gt_depth', 'gt_bboxes_ignore',
            'voxel_semantics', 'mask_camera', 'mask_lidar'
        ])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False,
        opencv_pp=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='ToEgo'),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='PointToMultiViewDepthFusion',
        downsample=1,
        grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'])
        ])
]

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
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            img_info_prototype='bevdet',
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
        img_info_prototype='bevdet',
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
        img_info_prototype='bevdet',
        box_type_3d='LiDAR'))
