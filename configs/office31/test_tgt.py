# model
num_classes = 31
model = dict(type='ResNet', depth=50, num_classes=num_classes)
loss = dict(
    test=dict(type='CrossEntropyLoss'),
)

# data
sample_list_dir = f'./data/office31_list'
test_domains = ['a', 'd', 'w']  # a: amazon; d: dslr; w: webcam

batch_size = 64
num_workers = 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# log & save
log_interval = 100
work_dir = None  # rewritten by args
resume = None
load = './checkpoints/office31/train_src_A/best_val.pth'
port = 10001
