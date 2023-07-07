# model
num_classes = 345
src_model = dict(type='ResNet', depth=50, num_classes=num_classes)
tgt_model = dict(type='ResNet', depth=50, num_classes=num_classes, pretrained=True)
tgt_head = dict(type='BottleNeckMLP', feature_dim=2048, bottleneck_dim=256, num_classes=num_classes,
                type1='bn', type2='wn')
loss = dict(
    test=dict(type='CrossEntropyLoss'),
)

# DINE hyper-parameters
ema = 0.6
lam_mix = 1.0
topk = 1

# domain
domains = {'C': 'clipart', 'I': 'infograph', 'P': 'painting', 'Q': 'quickdraw', 'R': 'real', 'S': 'sketch'}
src, tgt = None, None  # TODO

# data
dataset_type = 'DomainNet'
root = './data/DomainNet'
info_file_train = None  # TODO: f'{domains[tgt]}_train.txt'
info_file_test = None  # TODO: f'{domains[tgt]}_test.txt'
batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
data = dict(
    train=dict(
        ds_dict=dict(
            type=dataset_type,
            root=root,
            info_file=info_file_train,
            return_idx=True,
        ),
        trans_dict=dict(
            type='OHMultiView', views='w',
            mean=mean, std=std,
        )
    ),
    eval_train=dict(
        ds_dict=dict(
            type=dataset_type,
            root=root,
            info_file=info_file_train,
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std,
        )
    ),
    test=dict(
        ds_dict=dict(
            type=dataset_type,
            root=root,
            info_file=info_file_test,
        ),
        trans_dict=dict(
            type='OHMultiView', views='t',
            mean=mean, std=std,
        )
    ),
)

# training optimizer & scheduler
epochs = 10
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
test_interval = 1
pred_interval = 1  # epochs // 10
work_dir = None  # rewritten by args
resume = None
load = None  # TODO: f'./checkpoints/DomainNet/src_{src}/train_src_{src}/best_val.pth'
port = 10001
