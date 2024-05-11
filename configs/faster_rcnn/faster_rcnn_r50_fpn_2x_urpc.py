_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/urpc_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35,norm_type=2))