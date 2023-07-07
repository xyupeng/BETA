_base_ = [
    '../BETA_base.py'
]

domains = {'C': 'clipart', 'I': 'infograph', 'P': 'painting', 'Q': 'quickdraw', 'R': 'real', 'S': 'sketch'}
src, tgt = 'R', 'I'

# data
info_file_train = f'{domains[tgt]}_train.txt'
info_file_test = f'{domains[tgt]}_test.txt'
data = dict(
    warmup=dict(
        ds_dict=dict(
            info_file=info_file_train,
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            info_file=info_file_train,
        ),
    ),
    label=dict(
        ds_dict=dict(
            info_file=info_file_train,
        ),
    ),
    unlabel=dict(
        ds_dict=dict(
            info_file=info_file_train,
        ),
    ),
    test=dict(
        ds_dict=dict(
            info_file=info_file_test,
        ),
    ),
)

load = f'./checkpoints/DomainNet/src_{src}/train_src_{src}/best_val.pth'
