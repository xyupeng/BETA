_base_ = [
    '../train_src_base.py'
]

domains = {'C': 'clipart', 'I': 'infograph', 'P': 'painting', 'Q': 'quickdraw', 'R': 'real', 'S': 'sketch'}
src = 'R'

# data
data = dict(
    train=dict(
        ds_dict=dict(
            info_file=f'{domains[src]}_train.txt',
        ),
    ),
    val=dict(
        ds_dict=dict(
            info_file=f'{domains[src]}_test.txt',
        ),
    ),
)
