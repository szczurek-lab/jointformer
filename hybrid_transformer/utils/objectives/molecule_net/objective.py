import torch

DTYPE_OBJECTIVE = torch.float32

MOLECULE_NET_REGRESSION_TASKS = [
    'esol',
    'freesolv',
    'lipo'
]

MOLECULE_NET_CLASSIFICATION_TASKS = [
    # 'bace',
    'bbbp',
    # 'hiv',
    # 'tox21',
    # 'toxcast',
    # 'sider',
    # 'muv',
    # 'clintox'
]

MOLECULE_NET_TASKS = MOLECULE_NET_CLASSIFICATION_TASKS + MOLECULE_NET_REGRESSION_TASKS
