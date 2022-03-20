# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import time

from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.run import cli as run
from mim.utils import download_from_file, extract_tar, is_installed

dataset_url = 'https://download.openmmlab.com/mim/dataset.tar'
cfg_url = 'https://download.openmmlab.com/mim/resnet18_b16x8_custom.py'
ckpt_url = 'https://download.openmmlab.com/mim/epoch_3.pth'


def setup_module():
    runner = CliRunner()

    if not is_installed('mmcls'):
        result = runner.invoke(install, ['mmcls', '--yes'])
        assert result.exit_code == 0


def test_run():
    runner = CliRunner()

    if not osp.exists('/tmp/dataset'):
        download_from_file(dataset_url, '/tmp/dataset.tar')
        extract_tar('/tmp/dataset.tar', '/tmp/')

    if not osp.exists('/tmp/config.py'):
        download_from_file(cfg_url, '/tmp/config.py')

    if not osp.exists('/tmp/ckpt.pth'):
        download_from_file(ckpt_url, '/tmp/ckpt.pth')

    # wait for the download task to complete
    time.sleep(5)

    result = runner.invoke(
        run,
        ['mmcls', 'train', '/tmp/config.py', '--device=cpu', '--work-dir=tmp'])
    assert result.exit_code == 0
    result = runner.invoke(run, [
        'mmcls', 'test', '/tmp/config.py', '/tmp/ckpt.pth',
        '--metrics=accuracy'
    ])
    assert result.exit_code == 0
    result = runner.invoke(run, [
        'mmcls', 'xxx', '/tmp/config.py', '/tmp/ckpt.pth', '--metrics=accuracy'
    ])
    assert result.exit_code != 0
    result = runner.invoke(run, [
        'mmcls', 'test', '/tmp/xxx.py', '/tmp/ckpt.pth', '--metrics=accuracy'
    ])
    assert result.exit_code != 0

    shutil.rmtree('tmp')
