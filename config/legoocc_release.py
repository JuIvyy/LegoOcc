optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)}
    ),
)
grad_max_norm = 1.0
amp = False
seed = 1
print_freq = 50
eval_freq = 10
max_epochs = 10
target_size = 518
load_from = None
find_unused_parameters = False
track_running_stats = True
flag_depthanything_as_gt = False

ignore_label = 0
empty_idx = 12   # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
scale_range = [0.01, 0.08]
image_size = [480, 640]
resize_lim = [1.0, 1.0] 
num_frames = 1
offset = 0
grad_frames = None

_dim_ = 96
num_cams = 1
num_heads = 3
num_levels = 4
num_anchor = 16200
num_anchor_init = 8100
num_cross_layer = 3
num_self_layer = 3
num_decoder_fillhead = 2
semantics_activation = 'identity'
use_camera_embed = False

model = dict(
    type='VGGTGaussianSegmentor',
    min_max_temperature=[0.001, 1],
    with_opacity=True,
    semantic_dim=0,
    target_size=target_size,
    patch_size=14,
    frozen_backbone=False,
    freeze_blocks=0, # total 24
    flag_depthbranch=False,
    flag_depthanything_as_gt=flag_depthanything_as_gt,
    use_depthanything=True,
    num_bins=8,
    opacities_threshold=0.01,

    cuda_kwargs=dict(
        scale_multiplier=3,
        H=60, W=60, D=36,
        pc_min=[-51.2, -51.2, -5.0],
        grid_size=0.08),

)

depth_loss = dict(
    type='DepthLoss',
    input_dict=dict(
        depth_preds='depth_pred',
        depth_labels='depth_gt',
    ),
    weight=2.)

occ_loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='FocalLoss',
            activated=True,
            weight=100.0, 
            gamma=2.0,
            alpha=0.25,
            cls_freq=[5080655412, sum([722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393, 30949122, 9871618]), 3196722886],
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='LovaszLoss',
            activated=True,
            weight=1.0,
            ignore_label=ignore_label,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Sem_Scal_Loss',
            activated=True,
            weight=1.0,
            ignore_label=ignore_label,
            sem_cls_range=[1, 2],
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
        dict(
            type='Geo_Scal_Loss',
            activated=True,
            weight=1.0,
            empty_idx=2,
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label',
                'fov_mask': 'fov_mask'}),
    ]
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='FeatureAlignmentLoss',
            weight=2.0),
    ]
)

data_path = './data/occscannet' # path/to/your/data/occscannet
export_path = './data/scannet/export'
covis_threshold = 0.6
sam_mask_cache_root = './data/scannet/export/_sam_instance_cache'

train_dataset_config = dict(
    type='Scannet_Scene_OpenOccupancy_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx = empty_idx,
    phase='train',
    num_pts=num_anchor_init,
    data_tg='base', # 'mini' for mini-set
    vggt_image_preprocess=True,
    load_neighbor=5,
    load_neighbor_random_all=10,
    target_size=target_size,
)

val_dataset_config = dict(
    type='Scannet_Scene_OpenOccupancy_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx=empty_idx,
    phase='test',
    num_pts=num_anchor_init,
    data_tg='base', # 'mini' for mini-set
    vggt_image_preprocess=True,
    target_size=target_size,
)

train_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper_VGGT',
    final_dim = [480, 640], 
    resize_lim = resize_lim,
    phase='train', 
)

val_wrapper_config = dict(
    type='Scannet_Scene_Occ_DatasetWrapper_VGGT',
    final_dim = [480, 640],
    resize_lim = resize_lim,
    phase='test', 
)

train_loader_config = dict(
    batch_size = 2,
    shuffle = True,
    num_workers = 6,
)

val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
)
