_base_ = [
    '../BETA_base.py'
]

# data
src, tgt = 'a', 'd'
info_path = f'./data/office31_infos/{tgt}_list.txt'
data = dict(
    warmup=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    label=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    unlabel=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    test=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
)

load = f'./checkpoints/office31/src_{src}/train_src_{src}/best_val.pth'
