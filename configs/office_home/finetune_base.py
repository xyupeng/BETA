# model
num_classes = 65
tgt_model = dict(type='ResNet', depth=50, num_classes=num_classes)
tgt_head = dict(type='BottleNeckMLP', feature_dim=2048, bottleneck_dim=256, num_classes=num_classes,
                type1='bn', type2='wn')
loss = dict(
    test=dict(type='CrossEntropyLoss'),
)

# data
src, tgt = 'A', 'C'
info_path = f'./data/office_home_infos/{tgt}_list.txt'
batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data = dict(
    train=dict(
        ds_dict=dict(
            type='OfficeHome',
            info_path=info_path,
        ),
        trans_dict=dict(
            type='office_home_train',
            mean=mean, std=std,
        )
    ),
    test=dict(
        ds_dict=dict(
            type='OfficeHome',
            info_path=info_path,
        ),
        trans_dict=dict(
            type='office_home_test',
            mean=mean, std=std,
        )
    ),
)


# training optimizer & scheduler
epochs = 30
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
work_dir = None
resume = None
load = f'./checkpoints/office_home/src_{src}/BETA_{tgt}/last.pth'
port = 10001
