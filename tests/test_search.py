from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.search import cli as search
from mim.utils import is_installed


def setup_module():
    runner = CliRunner()

    if not is_installed('mmcls'):
        result = runner.invoke(install, ['mmcls', '--yes'])
        assert result.exit_code == 0


def test_search():
    runner = CliRunner()
    # mim search mmcls
    result = runner.invoke(search, ['mmcls'])
    assert result.exit_code == 0

    # mim search mmcls --remote
    # search master branch
    result = runner.invoke(search, ['mmcls', '--remote'])
    assert result.exit_code == 0

    # mim search mmcls==0.11.0 --remote
    result = runner.invoke(search, ['mmcls==0.11.0', '--remote'])
    assert result.exit_code == 0

    # mim search mmcls --model res
    # invali model
    result = runner.invoke(search, ['mmcls', '--model', 'res'])
    assert result.exit_code == 1
    # mim search mmcls --model resnet
    result = runner.invoke(search, ['mmcls', '--model', 'resnet'])
    assert result.exit_code == 0

    # mim search mmcls --valid-config
    result = runner.invoke(search, ['mmcls', '--valid-config'])
    assert result.exit_code == 0

    # mim search mmcls --config resnet18_b16x8_cifar1
    # invalid config
    result = runner.invoke(search,
                           ['mmcls', '--config', 'resnet18_b16x8_cifar1'])
    assert result.exit_code == 1
    # mim search mmcls --config resnet18_b16x8_cifar10
    result = runner.invoke(search,
                           ['mmcls', '--config', 'resnet18_b16x8_cifar10'])
    assert result.exit_code == 0

    # mim search mmcls --dataset cifar-1
    # invalid dataset
    result = runner.invoke(search, ['mmcls', '--dataset', 'cifar-1'])
    assert result.exit_code == 1

    # mim search mmcls --dataset cifar-10
    result = runner.invoke(search, ['mmcls', '--dataset', 'cifar-10'])
    assert result.exit_code == 0

    # mim search mmcls --condition 'batch_size>45,epochs>100'
    result = runner.invoke(
        search, ['mmcls', '--condition', 'batch_size>45,epochs>100'])
    assert result.exit_code == 0

    # mim search mmcls --condition 'batch_size>45 epochs>100'
    result = runner.invoke(
        search, ['mmcls', '--condition', 'batch_size>45 epochs>100'])
    assert result.exit_code == 0

    # mim search mmcls --condition '128<batch_size<=256'
    result = runner.invoke(search,
                           ['mmcls', '--condition', '128<batch_size<=256'])
    assert result.exit_code == 0

    # mim search mmcls --sort epoch
    result = runner.invoke(search, ['mmcls', '--sort', 'epoch'])
    assert result.exit_code == 0
    # mim search mmcls --sort epochs
    result = runner.invoke(search, ['mmcls', '--sort', 'epochs'])
    assert result.exit_code == 0

    # mim search mmcls --field epoch
    result = runner.invoke(search, ['mmcls', '--field', 'epoch'])
    assert result.exit_code == 0
    # mim search mmcls --field epochs
    result = runner.invoke(search, ['mmcls', '--field', 'epochs'])
    assert result.exit_code == 0
