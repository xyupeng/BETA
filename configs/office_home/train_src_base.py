# train_src_v1.py: build_office_home_loaders

# model
num_classes = 65
model = dict(type='ResNet', depth=50, num_classes=num_classes, pretrained=True)
loss = dict(
    train=dict(type='SmoothCE'),
    val=dict(type='CrossEntropyLoss'),
)

# data
src = 'A'  # A: Art, C: Clipart, P: Product, R: Real_World
info_path = f'./data/office_home_infos/{src}_list.txt'

batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
eps = 0.1  # label smoothing

# training optimizer & scheduler
epochs = 100
lr = 0.01
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

# log & save
log_interval = 100
work_dir = None
resume = None
load = None
port = 10001
