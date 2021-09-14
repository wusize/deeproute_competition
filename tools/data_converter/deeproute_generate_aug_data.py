import os

import mmcv
import numpy as np
import pickle
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from os import path as osp
from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import torch
import argparse


def generate_augmented_database(dataset_cfg,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(dataset_cfg['data_root'], dataset_cfg['split'], 'augmented_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(dataset_cfg['data_root'], dataset_cfg['split'], 'augmented_infos.pkl')
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = []

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        if input_dict is None:
            print(f'no annotations: {j}', flush=True)
            continue
        gt_info = {}
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].data.numpy()
        pts_filename = os.path.basename(input_dict['pts_filename']).replace('.bin', '.npy')
        # save points
        pts_save_file = os.path.join(database_save_path, pts_filename)
        with open(pts_save_file, 'wb') as f:
            np.save(f, points)

        gt_info['gt_bboxes_3d'] = example['gt_bboxes_3d'].data.tensor.numpy()
        gt_info['gt_labels_3d'] = example['gt_labels_3d'].data.numpy()
        gt_info['gt_names'] = example['ann_info']['gt_names']
        gt_info['is_aug'] = example['is_aug']

        all_db_infos.append(gt_info)


    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def generate_augmented_data_in_anno_format(dataset,
                                        database_save_path,
                                        start_idx, end_idx,
                                used_classes=None,
                                db_info_save_path=None):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """

    mmcv.mkdir_or_exist(database_save_path)
    print(start_idx, end_idx, flush=True)
    group_counter = 0
    gt_dets = []
    if start_idx == 0:
        for j in track_iter_progress(list(range(start_idx, end_idx))):
            input_dict = dataset.get_data_info(j)
            if input_dict is None:
                print(f'no annotations: {j}', flush=True)
                continue
            gt_info = {}
            dataset.pre_pipeline(input_dict)
            example = dataset.pipeline(input_dict)
            annos = example['ann_info']
            image_idx = example['sample_idx']
            points = example['points'].data.numpy()
            pts_filename = os.path.basename(input_dict['pts_filename'])  # .replace('.bin', '.npy')
            # save points
            pts_save_file = os.path.join(database_save_path, pts_filename)
            # with open(pts_save_file, 'wb') as f:
            #     np.save(f, points)
            with open(pts_save_file, 'w') as f:
                points.tofile(f)

            gt_info['boxes_3d'] = example['gt_bboxes_3d'].data.tensor
            gt_info['scores_3d'] = torch.ones(gt_info['boxes_3d'].shape[0])
            gt_info['labels_3d'] = example['gt_labels_3d'].data
            for idx in range(len(example['is_aug'])):
                if example['is_aug'][idx] > 0:
                    gt_info['scores_3d'][idx] = 0.99

            gt_dets.append(gt_info)
    else:
        for j in range(start_idx, end_idx):
            input_dict = dataset.get_data_info(j)
            if input_dict is None:
                # print(f'no annotations: {j}', flush=True)
                continue
            gt_info = {}
            dataset.pre_pipeline(input_dict)
            example = dataset.pipeline(input_dict)
            annos = example['ann_info']
            image_idx = example['sample_idx']
            points = example['points'].data.numpy()
            pts_filename = os.path.basename(input_dict['pts_filename'])  # .replace('.bin', '.npy')
            # save points
            pts_save_file = os.path.join(database_save_path, pts_filename)
            # with open(pts_save_file, 'wb') as f:
            #     np.save(f, points)
            with open(pts_save_file, 'w') as f:
                points.tofile(f)

            gt_info['boxes_3d'] = example['gt_bboxes_3d'].data.tensor
            gt_info['scores_3d'] = torch.ones(gt_info['boxes_3d'].shape[0])
            gt_info['labels_3d'] = example['gt_labels_3d'].data
            for idx in range(len(example['is_aug'])):
                if example['is_aug'][idx] > 0:
                    gt_info['scores_3d'][idx] = 0.99

            gt_dets.append(gt_info)

    dataset.format_results(gt_dets, 'aug_groundtruth')


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

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'training/instance_infos.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            CAR=5,
            VAN=5,
            TRUCK=5,
            BIG_TRUCK=5,
            BUS=5,
            PEDESTRIAN=5,
            CYCLIST=5,
            TRICYCLE=5,
            CONE=5,
        )),
    classes=class_names,
    sample_groups=dict(
        CAR=20,
        VAN=20,
        TRUCK=20,
        BIG_TRUCK=20,
        BUS=20,
        PEDESTRIAN=20,
        CYCLIST=20,
        TRICYCLE=20,
        CONE=20,
    ))

input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05]),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'sample_idx', 'ann_info', 'is_aug']),
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--format',
        action='store_true',
        help='Format the output results')
    parser.add_argument(
        '--num_parts', help='save dir',
        default=5,
        type=int)

    args, _ = parser.parse_known_args()

    return args


def single_thread_func(start_id, num_parts):
    dataset = build_dataset(dataset_cfg)
    num_frames = len(dataset)
    length = num_frames // num_parts
    if start_id == num_parts - 1:
        _range = [start_id * length, num_frames]
    else:
        _range = [start_id * length, start_id * length + length]
    database_save_path = osp.join(dataset_cfg['data_root'], dataset_cfg['split'], 'aug_pointcloud')
    generate_augmented_data_in_anno_format(dataset, database_save_path, _range[0], _range[1])


if __name__ == '__main__':
    args = parse_args()
    if args.format:
        import threading
        threads = []
        for i in range(args.num_parts):
            # print(f'i={i}', flush=True)
            t = threading.Thread(target=single_thread_func, name='single_thread_func', args={i, args.num_parts})
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        print('end', flush=True)
    else:
        generate_augmented_database(dataset_cfg)









