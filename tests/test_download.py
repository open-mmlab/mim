import tempfile

import pytest
from click.testing import CliRunner

from mim.commands.download import download
from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    # mim uninstall mmcls --yes
    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0


def test_download():
    runner = CliRunner()
    result = runner.invoke(install, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    with pytest.raises(ValueError):
        # verion is not allowed
        download('mmcls==0.11.0', ['resnet18_b16x8_cifar10'])

    with pytest.raises(RuntimeError):
        # mmcls is not installed
        download('mmcls', ['resnet18_b16x8_cifar10'])

    with pytest.raises(ValueError):
        # invalid config
        download('mmcls==0.11.0', ['resnet18_b16x8_cifar1'])

    runner = CliRunner()
    # mim install mmcls --yes
    result = runner.invoke(install, [
        'mmcls', '--yes', '-f',
        'https://github.com/open-mmlab/mmclassification.git'
    ])
    assert result.exit_code == 0

    # mim download mmcls --config resnet18_b16x8_cifar10
    checkpoints = download('mmcls', ['resnet18_b16x8_cifar10'])
    assert checkpoints == ['resnet18_b16x8_cifar10_20200823-f906fa4e.pth']
    checkpoints = download('mmcls', ['resnet18_b16x8_cifar10'])
    checkpoints = download('mmcls',
                           ['resnet18_b16x8_cifar10'],
                           model_info_local=False)
    assert checkpoints == ['resnet18_b16x8_cifar10_20200823-f906fa4e.pth']

    # mim download mmcls --config resnet18_b16x8_cifar10 --dest temp_root
    with tempfile.TemporaryDirectory() as temp_root:
        checkpoints = download('mmcls', ['resnet18_b16x8_cifar10'], temp_root)
        assert checkpoints == ['resnet18_b16x8_cifar10_20200823-f906fa4e.pth']
        checkpoints = download('mmcls',
                               ['resnet18_b16x8_cifar10'],
                               temp_root,
                               model_info_local=False)
        assert checkpoints == ['resnet18_b16x8_cifar10_20200823-f906fa4e.pth']
