import os
import os.path as osp
import random as rd
import subprocess
from typing import Optional, Tuple, Union

import click

from mim.click import CustomCommand, param2lowercase
from mim.utils import (
    echo_success,
    exit_with_error,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    recursively_find,
)


@click.command(
    'test',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str, callback=param2lowercase)
@click.argument('config', type=str)
@click.option(
    '-C', '--checkpoint', type=str, default=None, help='checkpoint path')
@click.option(
    '-l',
    '--launcher',
    type=click.Choice(['none', 'pytorch', 'slurm'], case_sensitive=False),
    default='none',
    help='Job launcher')
@click.option(
    '--port',
    type=int,
    default=None,
    help=('The port used for inter-process communication (only applicable to '
          'slurm / pytorch launchers). If set to None, will randomly choose '
          'a port between 20000 and 30000'))
@click.option(
    '-G',
    '--gpus',
    type=int,
    help='Number of gpus to use (only applicable to launcher == "slurm")')
@click.option(
    '-g',
    '--gpus-per-node',
    type=int,
    help=('Number of gpus per node to use '
          '(only applicable to launcher == "slurm")'))
@click.option(
    '-c',
    '--cpus-per-task',
    type=int,
    default=2,
    help='Number of cpus per task (only applicable to launcher == "slurm")')
@click.option(
    '-p',
    '--partition',
    type=str,
    help='The partition to use (only applicable to launcher == "slurm")')
@click.option(
    '--srun-args', type=str, help='Other srun arguments that might be used')
@click.option('-y', '--yes', is_flag=True, help='Don’t ask for confirmation.')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def cli(package: str,
        config: str,
        checkpoint: str,
        gpus: int,
        gpus_per_node: int,
        partition: str,
        cpus_per_task: int = 2,
        launcher: str = 'none',
        port: int = None,
        srun_args: Optional[str] = None,
        yes: bool = False,
        other_args: tuple = ()) -> None:
    """Perform Testing.

    Example:

    \b
    # Test models on a single server with 1 GPU, report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 1 --metrics accuracy
    # Test models on a single server with 1 GPU, save predictions
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 1 --out tmp.pkl
    # Test models on a single server with 4 GPUs, pytorch distributed,
    # report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 4 --launcher pytorch --metrics accuracy
    # Test models on a slurm HPC with one 8-GPU node, report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 8 --metrics accuracy --partition \
        partition_name --gpus-per-node 8 --launcher slurm
    # Print help messages of sub-command test
    > mim test -h
    # Print help messages of sub-command test and the testing script of mmcls
    > mim test mmcls -h
    """

    is_success, msg = test(
        package=package,
        config=config,
        checkpoint=checkpoint,
        gpus=gpus,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        partition=partition,
        launcher=launcher,
        port=port,
        srun_args=srun_args,
        yes=yes,
        other_args=other_args)

    if is_success:
        echo_success(msg)  # type: ignore
    else:
        exit_with_error(msg)


def test(
    package: str,
    config: str,
    checkpoint: str = None,
    gpus: int = None,
    gpus_per_node: int = None,
    cpus_per_task: int = 2,
    partition: str = None,
    launcher: str = 'none',
    port: int = None,
    srun_args: Optional[str] = None,
    yes: bool = True,
    other_args: tuple = ()
) -> Tuple[bool, Union[str, Exception]]:
    """Test a model with given config.

    Args:
        package (str): The codebase name.
        config (str): The config file path. If not exists, will search in the
            config files of the codebase.
        checkpoint (str): The path to the checkpoint file. Default to None.
        gpus (int): Number of gpus used for testing
            (only applicable to launcher == "slurm"). Defaults to None.
        gpus_per_node (int, optional): Number of gpus per node to use
            (only applicable to launcher == "slurm"). Defaults to None.
        cpus_per_task (int, optional): Number of cpus per task to use
            (only applicable to launcher == "slurm"). Defaults to None.
        partition (str, optional): The partition name
            (only applicable to launcher == "slurm"). Defaults to None.
        launcher (str, optional): The launcher used to launch jobs.
            Defaults to 'none'.
        port (int | None, optional): The port used for inter-process
            communication (only applicable to slurm / pytorch launchers).
            Default to None. If set to None, will randomly choose a port
            between 20000 and 30000.
        srun_args (str, optional): Other srun arguments that might be
            used, all arguments should be in a string. Defaults to None.
        yes (bool): Don’t ask for confirmation. Default: True.
        other_args (tuple, optional): Other arguments, will be passed to the
            codebase's training script. Defaults to ().
    """
    full_name = module_full_name(package)
    if full_name == '':
        msg = f"Can't determine a unique package given abbreviation {package}"
        raise ValueError(highlighted_error(msg))
    package = full_name

    # `checkpoint` is a compulsory argument for all mm codebases except
    # mmtracking, so that if the codebase is not mmtracking, user must specify
    # the checkpoint.
    # [TODO]: refactor the code logic and remove the hard coding
    if checkpoint is None:
        assert package == 'mmtrack', (
            'You must specify the checkpoint path '
            'unless you are testing mmtracking models')

    if port is None:
        port = rd.randint(20000, 30000)

    if launcher in ['slurm', 'pytorch']:
        click.echo(f'Using port {port} for synchronization. ')

    # If launcher == "slurm", must have following args
    if launcher == 'slurm':
        msg = ('If launcher is slurm, '
               'gpus, gpus-per-node and partition should not be None')
        flag = (gpus_per_node
                is not None) and (partition is not None) and (gpus is not None)
        assert flag, msg

    if not is_installed(package):
        msg = (f'The codebase {package} is not installed, '
               'do you want to install it? ')
        if yes or click.confirm(msg):
            click.echo(f'Installing {package}')
            cmd = ['mim', 'install', package]
            ret = subprocess.check_call(cmd)
            if ret != 0:
                msg = f'{package} is not successfully installed'
                raise RuntimeError(highlighted_error(msg))
            else:
                click.echo(f'{package} is successfully installed')
        else:
            msg = f'You can not test this model without {package} installed.'
            return False, msg

    pkg_root = get_installed_path(package)

    if not osp.exists(config):
        files = recursively_find(pkg_root, osp.basename(config))

        if len(files) == 0:
            msg = (f"The path {config} doesn't exist and we can not find "
                   f'the config file in codebase {package}.')
            raise ValueError(highlighted_error(msg))
        elif len(files) > 1:
            msg = (
                f"The path {config} doesn't exist and we find multiple "
                f'config files with same name in codebase {package}: {files}.')
            raise ValueError(highlighted_error(msg))
        click.echo(
            f"The path {config} doesn't exist but we find the config file "
            f'in codebase {package}, will use {files[0]} instead.')
        config = files[0]

    # We know that 'config' exists and is legal.
    test_script = osp.join(pkg_root, 'tools/test.py')

    common_args = ['--launcher', launcher] + list(other_args)

    if launcher == 'none':
        cmd = ['python', test_script, config]
        if checkpoint:
            cmd += [checkpoint]
        cmd += common_args
    elif launcher == 'pytorch':
        cmd = [
            'python', '-m', 'torch.distributed.launch',
            f'--nproc_per_node={gpus}', f'--master_port={port}', test_script,
            config
        ]
        if checkpoint:
            cmd += [checkpoint]
        cmd += common_args
    elif launcher == 'slurm':
        parsed_srun_args = srun_args.split() if srun_args else []
        cmd = [
            'srun', '-p', f'{partition}', f'--gres=gpu:{gpus_per_node}',
            f'--ntasks={gpus}', f'--ntasks-per-node={gpus_per_node}',
            f'--cpus-per-task={cpus_per_task}', '--kill-on-bad-exit=1'
        ]
        cmd += parsed_srun_args
        cmd += ['python', '-u', test_script, config]
        if checkpoint:
            cmd += [checkpoint]
        cmd += common_args

    cmd_text = ' '.join(cmd)
    click.echo(f'Testing command is {cmd_text}. ')
    ret = subprocess.check_call(
        cmd, env=dict(os.environ, MASTER_PORT=str(port)))
    if ret == 0:
        return True, 'Testing finished successfully.'
    else:
        return False, 'Testing not finished successfully.'
