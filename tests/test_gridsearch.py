import os.path as osp
import tempfile
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
    with tempfile.TemporaryDirectory() as temp_root:
        if not osp.exists(f'/{temp_root}/dataset'):
            download_from_file(dataset_url, f'/{temp_root}/dataset.tar')
            extract_tar(f'{temp_root}/dataset.tar', f'{temp_root}/')

        if not osp.exists(f'{temp_root}/config.py'):
            download_from_file(cfg_url, f'{temp_root}/config.py')

        # wait for the download task to complete
        time.sleep(5)

        args1 = [
            'mmcls', f'{temp_root}/config.py', '--gpus=1',
            f'--work-dir={temp_root}', '--search-args',
            '--optimizer.lr 1e-3 1e-4'
        ]
        args2 = [
            'mmcls', f'{temp_root}/config.py', '--gpus=1',
            f'--work-dir={temp_root}', '--search-args',
            '--optimizer.weight_decay 1e-3 1e-4'
        ]
        args3 = [
            'mmcls', f'{temp_root}/xxx.py', '--gpus=1',
            f'--work-dir={temp_root}', '--search-args',
            '--optimizer.lr 1e-3 1e-4'
        ]
        args4 = [
            'mmcls', f'{temp_root}/config.py', '--gpus=1',
            f'--work-dir={temp_root}', '--search-args'
        ]

        result = runner.invoke(gridsearch, args1)
        assert result.exit_code == 0

        result = runner.invoke(gridsearch, args2)
        assert result.exit_code == 0

        result = runner.invoke(gridsearch, args3)
        assert result.exit_code != 0

        result = runner.invoke(gridsearch, args4)
        assert result.exit_code != 0
