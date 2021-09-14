from create_gt_database import create_groundtruth_database
point_cloud_range = [-80, -80, -5, 80, 80, 3]   # x y z x y z

# dataset settings
dataset_type = 'DeeprouteDataset'
data_root = '../../data/deeproute_competition/'

class_names=[
            'CAR',
            'VAN',
            'TRUCK',
            'BIG_TRUCK',
            'BUS',
            'PEDESTRIAN',
            'CYCLIST',
            'TRICYCLE',
            'CONE']

input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05]),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointShuffle'),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'sample_idx', 'ann_info']),
]

dataset_cfg = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_train.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False)

create_groundtruth_database(dataset_cfg)
