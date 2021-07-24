import os
import os.path as osp

USER = 'open-mmlab'
DEFAULT_URL = f'https://github.com/{USER}'

WHEEL_URL = {
    'mmcv-full':
    'https://download.openmmlab.com/mmcv/dist/{cuda_version}/'
    '{torch_version}/index.html',
}
RAW_GITHUB_URL = 'https://raw.githubusercontent.com/{owner}/{repo}/{branch}'
PKG2PROJECT = {
    'mmcv-full': 'mmcv',
    'mmcls': 'mmclassification',
    'mmdet': 'mmdetection',
    'mmdet3d': 'mmdetection3d',
    'mmsegmentation': 'mmsegmentation',
    'mmaction2': 'mmaction2',
    'mmtrack': 'mmtracking',
    'mmpose': 'mmpose',
    'mmedit': 'mmediting',
    'mmocr': 'mmocr',
    'mmgen': 'mmgeneration',
}
# TODO: Should directly infer MODULE name from PKG info
PKG2MODULE = {
    'mmcv-full': 'mmcv',
    'mmaction2': 'mmaction',
    'mmsegmentation': 'mmseg',
}
MODULE2PKG = {
    'mmaction': 'mmaction2',
    'mmseg': 'mmsegmentation',
}

HOME = osp.expanduser('~')
DEFAULT_CACHE_DIR = osp.join(HOME, '.cache', 'mim')
if not osp.exists(DEFAULT_CACHE_DIR):
    os.makedirs(DEFAULT_CACHE_DIR)
