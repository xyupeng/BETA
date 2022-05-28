# model
num_classes = 31
src_model = dict(type='ResNet', depth=50, num_classes=num_classes)
tgt_model = dict(type='ResNet', depth=50, num_classes=num_classes, pretrained=True)
tgt_head = dict(type='BottleNeckMLP', feature_dim=2048, bottleneck_dim=256, num_classes=num_classes,
                type1='bn', type2='wn')

# DivideMix hyper-parameters
lam_u, lam_p, lam_t = 0, 1.0, 0.
T_sharpen = 0.5
alpha = 1.0
tau_p = 0.8

# DINE hyper-parameters
ema = 0.6
lam_mix = 1.0
topk = 1

loss = dict(
    train=dict(type='SemiLoss') if lam_u > 0 else dict(type='SmoothCE'),
    test=dict(type='CrossEntropyLoss'),
)

# data
src, tgt = 'a', 'd'  # a: amazon; d: dslr; w: webcam
info_path = f'./data/office31_infos/{tgt}_list.txt'
batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data = dict(
    warmup=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='warmup'
        ),
        trans_dict=dict(
            type='OHMultiView', views='w',
            mean=mean, std=std,
        )
    ),
    eval_train=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='eval_train'
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std
        )
    ),
    label=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='label'
        ),
        trans_dict=dict(
            type='OHMultiView', views='ws', 
            mean=mean, std=std
        )
    ),
    unlabel=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='unlabel'
        ),
        trans_dict=dict(
            type='OHMultiView', views='ws',
            mean=mean, std=std
        )
    ),
    test=dict(
        ds_dict=dict(
            type='SubOfficeHome',
            info_path=info_path,
            mode='test'
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std
        )
    ),
)

# training optimizer & scheduler
epochs = 50
warmup_epochs = 0
rampup_epochs = 5
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 50
test_interval = 1
pred_interval = epochs // 10
work_dir = None
resume = None
load = f'./checkpoints/office31/src_{src}/train_src_{src}/best_val.pth'
port = 10001
