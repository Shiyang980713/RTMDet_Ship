_base_ = [
    './_base_/default_runtime.py', './_base_/schedule_3x.py',
    './_base_/ship_rr.py'
]
checkpoint = '/remote-home/syfeng/MyProject/ship_det/RTMDet_Ship/settings/pretrain/lsk_s_backbone-e9d2e551.pth'  # noqa
val_cfg = None
angle_version = 'le90'
model = dict(
    type='mmdet.RTMDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        # embed_dims=[128, 256, 512, 1024],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2,2,4,2],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=133,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=4, num_workers=4)
