data='cityperson'    # 'caltech'    # 'cityperson'
if data=='cityperson':
    # model settings
    model = dict(
        type='FasterRCNN',
        pretrained='open-mmlab://msra/hrnetv2_w18',
        backbone=dict(
            type='DynamicHRNet',
            num_backbones=2,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(4,),
                    num_channels=(64,)),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=(4, 4),
                    num_channels=(18, 36)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(18, 36, 72)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(18, 36, 72, 144)))),
        # neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256),
        neck=[
            dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=0, num_outs=2),
            # dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=1, num_outs=1),
            dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=2, num_outs=3),  # neck copy 3 times
            # dict(type='HRFPN'DynamicHRFPN, in_channels=[18, 36, 72, 144], out_channels=256, outlayer=1, num_outs=1),
            # dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=2, num_outs=2)
        ],
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            # anchor_scales=[[8],[8],[10],[20]],          #[8],
            # anchor_ratios=[2.44],
            # anchor_strides=[4,8,16,16],                  #[4, 8, 16, 32, 64],
            anchor_scales=[8,12],          #[8], [6,8,10]
            anchor_ratios=[2.44],
            anchor_strides=[4,8,16,32,64],
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4,8,16,32]),    #[4, 8, 16, 32]),   # , 32
        bbox_head=dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
    # model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.005, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    )
    # dataset settings
    dataset_type = 'CocoDataset'
    data_root = '../datasets/cityperson/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MinIoURandomCrop'),
        dict(type='Resize', img_scale=(1280,640), keep_ratio=True),  # (1280,640)(2048, 1024)
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),  # [(1640,820), (2662,1331)],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/cityperson_train_all.json',
            img_prefix=data_root + 'images/train/mix/',
            # ann_file=data_root + 'annotations/instances_train_crop_modify.json',
            # img_prefix=data_root + 'images/train_cropImgs_modify/',
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/val_gt.json',
            img_prefix=data_root + 'images/val/mix/',
            pipeline=test_pipeline))
    # optimizer
    optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    # learning policy
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[45,60])
    checkpoint_config = dict(interval=1)
    # yapf:disable
    log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            # dict(type='TensorboardLoggerHook')
        ])
    # yapf:enable
    # runtime settings
    total_epochs = 70
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    work_dir = './work_dirs/DHRNet/hrnetw18_MinIoURandomCrop_2backbone_dynamicFromS4_lr0002_sig22_fanin'
    load_from = None
    resume_from = None
    workflow = [('train', 1)]

if data=='caltech':
    # model settings
    model = dict(
        type='FasterRCNN',
        pretrained='open-mmlab://msra/hrnetv2_w18',
        backbone=dict(
            type='DynamicHRNet',
            num_backbones=1,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(4,),
                    num_channels=(64,)),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=(4, 4),
                    num_channels=(18, 36)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(18, 36, 72)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(18, 36, 72, 144)))),
        neck=[
            dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=0, num_outs=4),
            # dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=2, num_outs=2),
            # dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=2, num_outs=2),
            # neck copy 3 times
            # dict(type='HRFPN'DynamicHRFPN, in_channels=[18, 36, 72, 144], out_channels=256, outlayer=1, num_outs=1),
            # dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256, outlayer=2, num_outs=2)
        ],
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            # anchor_scales=[[8],[8],[10],[20]],          #[8],
            # anchor_ratios=[2.44],
            # anchor_strides=[4,8,16,16],                  #[4, 8, 16, 32, 64],
            anchor_scales=[8,12],  # [8],
            anchor_ratios=[2.44],
            anchor_strides=[4, 8, 16, 32],
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),  # [4, 8, 16, 32]),   # , 32
        bbox_head=dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
    # model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,  ####################
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,  # [25600, 6400, 1600, 400, 100]
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    )
    # dataset settings
    dataset_type = 'CocoDataset'
    data_root = '../../datasets/caltech/'
    # dataset_type = 'ChestXrayDataset'
    # data_root = '../../datasets/chestXray/'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(640, 480), keep_ratio=True),  # (640, 480)
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 480),  # (1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    data = dict(
        imgs_per_gpu=6,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            # ann_file=data_root + 'annotation/train_chest_addIscrowd.json',
            # img_prefix=data_root + 'train_data/train/',
            ann_file=data_root + 'annotations/instances_train.json',
            # img_prefix=data_root + 'images/train/mix/',
            # ann_file=data_root + 'annotations/train.json',
            # _crop_modify'annotations/instances_train_crop.json',
            img_prefix=data_root + 'train2014/',
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            pipeline=test_pipeline),
        test=dict(
            type=dataset_type,
            # ann_file=data_root + 'annotation/val_chest_addIscrowd.json',
            # img_prefix=data_root + 'train_data/train/',
            ann_file=data_root + 'annotations/test.json',
            img_prefix=data_root + 'minival2015/',
            # ann_file=data_root + 'annotations/instances_train.json',
            # img_prefix=data_root + 'images/train/mix/',
            # ann_file=data_root + 'annotations/instances_train_crop_modify_subset.json',
            # img_prefix=data_root + 'images/train_cropImgs_modify/',
            pipeline=test_pipeline))
    # optimizer
    optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    # learning policy
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[8,11])
    checkpoint_config = dict(interval=1)
    # yapf:disable
    log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            # dict(type='TensorboardLoggerHook')
        ])
    # yapf:enable
    # runtime settings
    total_epochs = 12
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    work_dir = './work_dirs/caltech_faster_rcnn_hrnetv2p_w18_dynamicFromS4_1branch_-1_anchors812-12e_pretrain'
    load_from = '../../mmdetv1.1.0/mmdetection-1.1.0/work_dirs/faster_rcnn_hrnetv2p_w18_1x_1backbone_dynamicFromS4_cropImgs_lr0002_sig+22_channel1x_5fp_2anchors_-1/epoch_5.pth'
    # load_from = '../../mmdetv1.1.0/mmdetection-1.1.0/work_dirs/faster_rcnn_hrnetv2p_w18_1x_2backbone_dynamicFromS4_cropImgs_lr0002_sigmoid_channel1x_5fp_2anchors_-1/epoch_5.pth'
    resume_from = None  # '../../mmdetv1.1.0/mmdetection-1.1.0/work_dirs/faster_rcnn_hrnetv2p_w18_1x_2backbone_dynamicFromS4_cropImgs_lr0002_sigmoid_channel1x_5fp_2anchors_-1/epoch_5.pth'
    workflow = [('train', 1)]