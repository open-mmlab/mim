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
    name='train',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str, callback=param2lowercase)
@click.argument('config', type=str)
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
          'a port between 20000 and 30000. '))
@click.option(
    '-G', '--gpus', type=int, default=1, help='Number of gpus to use')
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
        gpus: int,
        gpus_per_node: int,
        partition: str,
        cpus_per_task: int = 2,
        launcher: str = 'none',
        port: int = None,
        srun_args: Optional[str] = None,
        yes: bool = False,
        other_args: tuple = ()) -> None:
    """Perform Training.

    Example:

    \b
    # Train models on a single server with CPU by setting `gpus` to 0 and
    # 'launcher' to 'none' (if applicable). The training script of the
    # corresponding codebase will fail if it doesn't support CPU training.
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 0
    # Train models on a single server with one GPU
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 1
    # Train models on a single server with 4 GPUs and pytorch distributed
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 4 \
        --launcher pytorch
    # Train models on a slurm HPC with one 8-GPU node
    > mim train mmcls resnet101_b16x8_cifar10.py --launcher slurm --gpus 8 \
        --gpus-per-node 8 --partition partition_name --work-dir tmp
    # Print help messages of sub-command train
    > mim train -h
    # Print help messages of sub-command train and the training script of mmcls
    > mim train mmcls -h
    """
    is_success, msg = train(
        package=package,
        config=config,
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


def train(
    package: str,
    config: str,
    gpus: int,
    gpus_per_node: int = None,
    cpus_per_task: int = 2,
    partition: str = None,
    launcher: str = 'none',
    port: int = None,
    srun_args: Optional[str] = None,
    yes: bool = True,
    other_args: tuple = ()
) -> Tuple[bool, Union[str, Exception]]:
    """Train a model with given config.

    Args:
        package (str): The codebase name.
        config (str): The config file path. If not exists, will search in the
            config files of the codebase.
        gpus (int): Number of gpus used for training.
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

    # If launcher == "slurm", must have following args
    if launcher == 'slurm':
        msg = ('If launcher is slurm, '
               'gpus-per-node and partition should not be None')
        flag = (gpus_per_node is not None) and (partition is not None)
        assert flag, msg

    if port is None:
        port = rd.randint(20000, 30000)

    if launcher in ['slurm', 'pytorch']:
        click.echo(f'Using port {port} for synchronization. ')

    if not is_installed(package):
        msg = (f'The codebase {package} is not installed, '
               'do you want to install the latest release? ')
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
            msg = f'You can not train this model without {package} installed.'
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

    train_script = osp.join(pkg_root, 'tools/train.py')

    common_args = ['--launcher', launcher] + list(other_args)

    if launcher == 'none':
        if gpus:
            cmd = ['python', train_script, config, '--gpus',
                   str(gpus)] + common_args
        else:
            cmd = ['python', train_script, config, '--device', 'cpu'
                   ] + common_args
    elif launcher == 'pytorch':
        cmd = [
            'python', '-m', 'torch.distributed.launch',
            f'--nproc_per_node={gpus}', f'--master_port={port}', train_script,
            config
        ] + common_args
    elif launcher == 'slurm':
        parsed_srun_args = srun_args.split() if srun_args else []
        cmd = [
            'srun', '-p', f'{partition}', f'--gres=gpu:{gpus_per_node}',
            f'--ntasks={gpus}', f'--ntasks-per-node={gpus_per_node}',
            f'--cpus-per-task={cpus_per_task}', '--kill-on-bad-exit=1'
        ] + parsed_srun_args + ['python', '-u', train_script, config
                                ] + common_args

    cmd_text = ' '.join(cmd)
    click.echo(f'Training command is {cmd_text}. ')
    ret = subprocess.check_call(
        cmd, env=dict(os.environ, MASTER_PORT=str(port)))
    if ret == 0:
        return True, 'Training finished successfully. '
    else:
        return False, 'Training not finished successfully. '
