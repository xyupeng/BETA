_base_ = [
    '../finetune_base.py'
]

# data
src, tgt = 'a', 'w'
info_path = f'./data/office31_infos/{tgt}_list.txt'
data = dict(
    train=dict(
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

load = f'./checkpoints/office31/src_{src}/BETA_{tgt}/last.pth'
