import os
import site

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

DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/.cache/mim'
if not os.path.exists(DEFAULT_CACHE_DIR):
    os.makedirs(DEFAULT_CACHE_DIR)

MMPACKAGE_PATH = os.path.join(site.getsitepackages()[0], 'mmpackage.txt')
