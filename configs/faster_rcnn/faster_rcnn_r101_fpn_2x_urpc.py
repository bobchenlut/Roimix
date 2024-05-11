_base_ = './faster_rcnn_r50_fpn_2x_urpc.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=15,norm_type=2))
