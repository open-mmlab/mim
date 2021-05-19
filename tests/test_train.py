import os.path as osp
import tempfile
import time

from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.train import cli as train
from mim.utils import download_from_file, extract_tar, is_installed

dataset_url = ('https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mim/'
               'dataset.tar')
cfg_url = ('https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mim/'
           'resnet18_b16x8_custom.py')


def setup_module():
    runner = CliRunner()

    if not is_installed('mmcls'):
        result = runner.invoke(install, ['mmcls', '--yes'])
        assert result.exit_code == 0


def test_train():
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_root:
        if not osp.exists(f'{temp_root}/dataset'):
            download_from_file(dataset_url, f'{temp_root}/dataset.tar')
            extract_tar(f'{temp_root}/dataset.tar', f'{temp_root}/')

        if not osp.exists(f'{temp_root}/config.py'):
            download_from_file(cfg_url, f'{temp_root}/config.py')

        # wait for the download task to complete
        time.sleep(5)

        result = runner.invoke(train, [
            'mmcls', f'{temp_root}/config.py', '--gpus=1',
            f'--work-dir={temp_root}'
        ])
        assert result.exit_code == 0

        result = runner.invoke(train, [
            'mmcls', f'{temp_root}/xxx.py', '--gpus=1',
            f'--work-dir={temp_root}'
        ])
        assert result.exit_code != 0
