_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py', '../../_base_/datasets/voc0712.py',
    '../../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=20))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12

load_from = "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth"

data = dict(
    samples_per_gpu=8,  # Batch size of a single GPU
    workers_per_gpu=3,
)

# nncf config
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './output'
workflow = [('train', 1)]

find_unused_parameters = True
nncf_load_from = None

nncf_config = {
    "input_info": {
        "sample_size": [1, 3, 1000, 600]
    },
    "compression": [
        {
            "algorithm": "quantization",
            "initializer": {
                "range": {
                    "num_init_steps": 10
                },
                "batchnorm_adaptation": {
                    "num_bn_adaptation_steps": 30,
                }

            }
        },
        {
            "algorithm": "filter_pruning",
            "params": {
                "schedule": "baseline",
                "pruning_target": 0.3
            }
        }
    ],
    "log_dir": work_dir
}
