# flake8: noqa
import os
import os.path as osp
import subprocess
from typing import Tuple, Union

import click

from mim.click import CustomCommand, param2lowercase
from mim.utils import (
    echo_success,
    echo_warning,
    exit_with_error,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    recursively_find,
)


@click.command(
    'run',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str, callback=param2lowercase)
@click.argument('command', type=str)
@click.option('-y', '--yes', is_flag=True, help='Don’t ask for confirmation.')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def cli(package: str, command: str, yes: bool, other_args: tuple = ()) -> None:
    """Run arbitrary command of a codebase.

    Note if the command you call takes config files or checkpoint paths as
    arguments, you must use paths that do exists. We do not accept URLs or
    config names (like resnet101_b16x8_cifar10.py) if that doesn't exist under
    your current directory.

    Example:

    \b
    # Get the Flops of a model
    > mim run mmcls get_flops resnet101_b16x8_cifar10.py
    # Publish a model
    > mim run mmcls publish_model input.pth output.pth
    # Train models on a slurm HPC with one GPU
    > srun -p partition --gres=gpu:1 mim run mmcls train \
        resnet101_b16x8_cifar10.py --work-dir tmp
    # Test models on a slurm HPC with one GPU, report accuracy
    > srun -p partition --gres=gpu:1 mim run mmcls test \
        resnet101_b16x8_cifar10.py tmp/epoch_3.pth --metrics accuracy
    # Print help messages of sub-command run
    > mim run -h
    # Print help messages of sub-command run, list all available scripts in
    # codebase mmcls
    > mim run mmcls -h
    # Print help messages of sub-command run, print the help message of
    # training script in mmcls
    > mim run mmcls train -h
    """
    is_success, msg = run(
        package=package, command=command, yes=yes, other_args=other_args)

    if is_success:
        echo_success(msg)  # type: ignore
    else:
        exit_with_error(msg)


def run(
    package: str,
    command: str,
    yes: bool = True,
    other_args: tuple = ()
) -> Tuple[bool, Union[str, Exception]]:
    """Run arbitrary command of a codebase.

    This command assumes the command scripts have been put into the
    ``package/tools`` directory.

    Args:
        package (str): The codebase name.
        command (str): The command name.
        yes (bool): Don’t ask for confirmation. Default: True.
        other_args (tuple, optional): Other arguments, will be passed to the
            codebase's script. Defaults to ().
    """
    full_name = module_full_name(package)
    if full_name == '':
        msg = f"Can't determine a unique package given abbreviation {package}"
        raise ValueError(highlighted_error(msg))
    package = full_name

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
            msg = (f"You can't run commands in {package} without "
                   f'{package} installed.')
            return False, msg

    pkg_root = get_installed_path(package)

    prefix = osp.join(pkg_root, 'tools/')

    command_domain = ''
    if ':' in command:
        split_command = command.split(':')
        command_domain = '/'.join(split_command[:-1])
        command = split_command[-1]

    files = recursively_find(prefix, command + '.py')

    if command_domain == '':
        suffix = f'/{command}.py'
    else:
        suffix = f'/{command_domain}/{command}.py'
    files = [f for f in files if f.endswith(suffix)]

    if len(files) == 0:
        msg = f'No script in codebase {package} has suffix {suffix}.'
        raise ValueError(highlighted_error(msg))
    elif len(files) > 1:
        echo_warning(
            f'Multiple scripts in codebase {package} have suffix {suffix}: ')
        for f in files:
            echo_warning(f)

        # Use the shortest path
        files.sort(key=lambda x: len(x.split('/')))
        echo_warning(f'We are using the script {files[0]}. ')
        echo_warning('To use other scripts, you need to use these commands: ')
        for f in files[1:]:
            cmd = f.split(prefix)[1].split('.')[0].replace('/', ':')
            echo_warning(f'Command for {f}: {cmd}')

    script = files[0]
    click.echo(f'Use the script {script} for command {command}.')

    cmd = ['python', script] + list(other_args)

    cmd_text = ' '.join(cmd)
    click.echo(f'The command to call is {cmd_text}. ')
    ret = subprocess.check_call(cmd, env=dict(os.environ))
    if ret == 0:
        return True, 'The script finished successfully.'
    else:
        return False, 'The script not finished successfully.'
