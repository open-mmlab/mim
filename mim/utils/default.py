import os
import pkg_resources

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

# Although site.getsitepackage() can also get the directory of site-packages,
# it will faild in virtualenv. It is an issue with virtualenv, which copies
# the bundled version of site.py to the venv when it is created.
MMPACKAGE_PATH = os.path.join(
    pkg_resources.get_distribution('click').location, 'mmpackage.txt')
