_base_ = [
    '../finetune_base.py'
]

# data
src, tgt = 'R', 'C'
info_path = f'./data/office_home_infos/{tgt}_list.txt'
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

load = f'./checkpoints/office_home/src_{src}/BETA_{tgt}/last.pth'
