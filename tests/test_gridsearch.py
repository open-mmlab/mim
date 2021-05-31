import os.path as osp
import shutil
import time

from click.testing import CliRunner

from mim.commands.gridsearch import cli as gridsearch
from mim.commands.install import cli as install
from mim.utils import download_from_file, extract_tar, is_installed

dataset_url = 'https://download.openmmlab.com/mim/dataset.tar'
cfg_url = 'https://download.openmmlab.com/mim/resnet18_b16x8_custom.py'


def setup_module():
    runner = CliRunner()

    if not is_installed('mmcls'):
        result = runner.invoke(install, ['mmcls', '--yes'])
        assert result.exit_code == 0


def test_gridsearch():
    runner = CliRunner()
    if not osp.exists('/tmp/dataset'):
        download_from_file(dataset_url, '/tmp/dataset.tar')
        extract_tar('/tmp/dataset.tar', '/tmp/')

    if not osp.exists('/tmp/config.py'):
        download_from_file(cfg_url, '/tmp/config.py')

    # wait for the download task to complete
    time.sleep(5)

    args1 = [
        'mmcls', '/tmp/config.py', '--gpus=0', '--work-dir=tmp',
        '--search-args', '--optimizer.lr 1e-3 1e-4'
    ]
    args2 = [
        'mmcls', '/tmp/config.py', '--gpus=0', '--work-dir=tmp',
        '--search-args', '--optimizer.weight_decay 1e-3 1e-4'
    ]
    args3 = [
        'mmcls', '/tmp/xxx.py', '--gpus=0', '--work-dir=tmp', '--search-args',
        '--optimizer.lr 1e-3 1e-4'
    ]
    args4 = [
        'mmcls', '/tmp/config.py', '--gpus=0', '--work-dir=tmp',
        '--search-args'
    ]

    result = runner.invoke(gridsearch, args1)
    assert result.exit_code == 0

    result = runner.invoke(gridsearch, args2)
    assert result.exit_code == 0

    result = runner.invoke(gridsearch, args3)
    assert result.exit_code != 0

    result = runner.invoke(gridsearch, args4)
    assert result.exit_code != 0

    shutil.rmtree('tmp_search_optimizer.lr_0.001')
    shutil.rmtree('tmp_search_optimizer.lr_0.0001')
    shutil.rmtree('tmp_search_optimizer.weight_decay_0.001')
    shutil.rmtree('tmp_search_optimizer.weight_decay_0.0001')
