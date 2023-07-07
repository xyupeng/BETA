# model
num_classes = 345
model = dict(type='ResNet', depth=50, num_classes=num_classes, pretrained=True)
loss = dict(
    train=dict(type='SmoothCE'),
    val=dict(type='CrossEntropyLoss'),
)

domains = {'C': 'clipart', 'I': 'infograph', 'P': 'painting', 'Q': 'quickdraw', 'R': 'real', 'S': 'sketch'}
src = None  # TODO

# data
root = './data/DomainNet/'
batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
eps = 0.1  # label smoothing
data = dict(
    train=dict(
        ds_dict=dict(
            type='DomainNet',
            root=root,
            info_file=None,  # TODO
        ),
        trans_dict=dict(
            type='office_home_train',
            mean=mean, std=std,
        )
    ),
    val=dict(
        ds_dict=dict(
            type='DomainNet',
            root=root,
            info_file=None,  # TODO
        ),
        trans_dict=dict(
            type='office_home_test',
            mean=mean, std=std,
        )
    ),
)

# training optimizer & scheduler
epochs = 50
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
