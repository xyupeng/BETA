import os


def broadcast_config(cfg_dir='DomainNet', task='DINE1_divm', domains=None):
    base_cfg = f'./configs/{cfg_dir}/src_{domains[0]}/{task}_{domains[1]}.py'
    base_lines = open(base_cfg).readlines()
    line_idx = -1
    for i, line in enumerate(base_lines):
        if line.startswith('src, tgt'):
            line_idx = i
            break
    assert line_idx != -1

    for src in domains:
        for tgt in domains:
            if tgt == src:
                continue
            cfg_path = f'./configs/{cfg_dir}/src_{src}/{task}_{tgt}.py'
            if cfg_path == base_cfg:
                continue

            new_line = f"src, tgt = '{src}', '{tgt}'\n"
            base_lines[line_idx] = new_line
            with open(cfg_path, 'w') as f:
                f.writelines(base_lines)


if __name__ == '__main__':
    # broadcast_config(cfg_dir='office31', task='BETA', domains=['a', 'd', 'w'])
    broadcast_config(cfg_dir='office31', task='finetune', domains=['a', 'd', 'w'])

    # broadcast_config(cfg_dir='office_home', task='BETA', domains=['A', 'C', 'P', 'R'])
    broadcast_config(cfg_dir='office_home', task='finetune', domains=['A', 'C', 'P', 'R'])
