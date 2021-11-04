model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    )
)
data = dict(
    batch_size=8,
    train=dict(
        type='PAN_PP_BENCHMARK',
        split=('pretrain', 'train'),
        is_train=True,
        is_transform=True,
        img_size=736,
        min_sizes=(600, 672, 704, 736, 768, 800, 832, 864, 896),
        max_sizes=(1600, ),
        kernel_scale=0.5,
        read_type='pil',
        with_rec=True
    ),
    test=dict(
        type='PAN_PP_BENCHMARK',
        split=('val'),
        is_train=False,
        is_transform=True,
        img_size=736,
        min_sizes=(1000, ),
        max_sizes=(1600, ),
        kernel_scale=0.5,
        read_type='pil',
        with_rec=True
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=10,
    optimizer='Adam',
    use_ex=False,
)
test_cfg = dict(
    min_score=0.85,
    min_area=260,
    min_kernel_area=2.6,
    scale=4,
    bbox_type='poly',
    result_path='outputs/submit_ic15.zip',
)
