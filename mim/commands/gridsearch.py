import copy as cp
import itertools
import os
import os.path as osp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor as Executor
from typing import Optional, Tuple, Union

import click

from mim.click import CustomCommand
from mim.utils import (
    args2string,
    echo_error,
    echo_success,
    exit_with_error,
    get_config,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    recursively_find,
    set_config,
    string2args,
)


@click.command(
    name='gridsearch',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('package', type=str)
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
    default=29500,
    help=('The port used for inter-process communication '
          '(only applicable to slurm / pytorch launchers)'))
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
    '-j', '--max-jobs', type=int, help='Max parallel number', default=1)
@click.option(
    '--srun-args', type=str, help='Other srun arguments that might be used')
@click.option('-y', '--yes', is_flag=True, help='Don’t ask for confirmation.')
@click.option(
    '--search-args', type=str, help='Arguments for hyper parameters search')
@click.argument('other_args', nargs=-1, type=click.UNPROCESSED)
def cli(package: str,
        config: str,
        gpus: int,
        gpus_per_node: int,
        partition: str,
        cpus_per_task: int = 2,
        max_jobs: int = 1,
        launcher: str = 'none',
        port: int = 29500,
        srun_args: Optional[str] = None,
        search_args: str = '',
        yes: bool = False,
        other_args: tuple = ()) -> None:
    """Perform Hyper-parameter search.

    Example:

    \b
    # Parameter search on a single server with CPU by setting `gpus` to 0 and
    # 'launcher' to 'none' (if applicable). The training script of the
    # corresponding codebase will fail if it doesn't support CPU training.
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        0 --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search learning
    # rate
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search
    # weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.weight_decay 1e-3 1e-4'
    # Parameter search with on a single server with one GPU, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        1 --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay \
        1e-3 1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        8 --partition partition_name --gpus-per-node 8 --launcher slurm \
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 \
        1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay, max parallel jobs is 2
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus \
        8 --partition partition_name --gpus-per-node 8 --launcher slurm \
        --max-jobs 2 --search-args '--optimizer.lr 1e-2 1e-3 \
        --optimizer.weight_decay 1e-3 1e-4'
    # Print the help message of sub-command search
    > mim gridsearch -h
    # Print the help message of sub-command search and the help message of the
    # training script of codebase mmcls
    > mim gridsearch mmcls -h
    """

    is_success, msg = gridsearch(
        package=package,
        config=config,
        gpus=gpus,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        max_jobs=max_jobs,
        partition=partition,
        launcher=launcher,
        port=port,
        srun_args=srun_args,
        search_args=search_args,
        yes=yes,
        other_args=other_args)

    if is_success:
        echo_success(msg)  # type: ignore
    else:
        exit_with_error(msg)


def gridsearch(
    package: str,
    config: str,
    gpus: int,
    gpus_per_node: int = None,
    cpus_per_task: int = 2,
    max_jobs: int = 1,
    partition: str = None,
    launcher: str = 'none',
    port: int = 29500,
    srun_args: Optional[str] = None,
    search_args: str = '',
    yes: bool = True,
    other_args: tuple = ()
) -> Tuple[bool, Union[str, Exception]]:
    """Hyper parameter search with given config.

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
        max_jobs (int, optional): The max number of workers. Applicable only
            if launcher == 'slurm'. Default to 1.
        launcher (str, optional): The launcher used to launch jobs.
            Defaults to 'none'.
        port (int, optional): The port used for inter-process communication
            (only applicable to slurm / pytorch launchers). Default to 29500.
        srun_args (str, optional): Other srun arguments that might be
            used, all arguments should be in a string. Defaults to None.
        search_args (str, optional): Arguments for hyper parameters search, all
            arguments should be in a string. Defaults to None.
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
        if not flag:
            raise AssertionError(highlighted_error(msg))

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
            msg = f'You can not train this model without {package} installed.'
            return False, msg

    pkg_root = get_installed_path(package)

    if not osp.exists(config):
        files = recursively_find(pkg_root, osp.basename(config))

        if len(files) == 0:
            msg = (f"The path {config} doesn't exist and we can not "
                   f'find the config file in codebase {package}.')
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

    # parsing search_args
    # the search_args looks like:
    # "--optimizer.lr 0.001 0.01 0.1 --optimizer.weight_decay 1e-4 1e-3 1e-2"
    search_args_dict = string2args(search_args)
    if not len(search_args_dict):
        msg = 'Should specify at least one arg for searching'
        raise ValueError(highlighted_error(msg))

    for k in search_args_dict:
        if search_args_dict[k] is bool:
            msg = f'Should specify at least one value for arg {k}'
            raise ValueError(highlighted_error(msg))

    try:
        from mmcv import Config
    except ImportError:
        msg = 'Please install mmcv to use the gridsearch command.'
        raise ImportError(highlighted_error(msg))

    cfg = Config.fromfile(config)
    for arg in search_args_dict:
        try:
            arg_value = get_config(cfg, arg)
            if arg_value and not isinstance(arg_value, str):
                search_args_dict[arg] = [
                    eval(x) for x in search_args_dict[arg]
                ]
                for val in search_args_dict[arg]:
                    assert type(val) == type(arg_value)
        except AssertionError:
            msg = f'Arg {arg} not in the config file. '
            raise AssertionError(highlighted_error(msg))

    other_args_dict = string2args(' '.join(other_args))

    work_dir = other_args_dict.get('work-dir')

    if work_dir:
        work_dir = work_dir[0]
    else:
        work_dir = cfg.get('work_dir')

    if work_dir is None:
        msg = 'work_dir is not specified'
        raise AssertionError(highlighted_error(msg))

    cfg.pop('work_dir', None)

    # remove redundant '/' at the end of work_dir

    assert work_dir  # To pass mypy test
    while work_dir.endswith('/'):
        work_dir = work_dir[:-1]

    config_tmpl, config_suffix = osp.splitext(osp.basename(config))
    work_dir_tmpl = work_dir

    cmds = []
    exp_names = []

    arg_names = [k for k in search_args_dict]
    arg_values = [search_args_dict[k] for k in arg_names]

    for combination in itertools.product(*arg_values):
        cur_cfg = Config(cp.deepcopy(cfg))
        suffix_list = []

        for k, v in zip(arg_names, combination):
            suffix_list.extend([k, str(v)])
            set_config(cur_cfg, k, v)

        name_suffix = '_search_' + '_'.join(suffix_list)
        work_dir = work_dir_tmpl + name_suffix
        os.makedirs(work_dir, exist_ok=True)

        config_name = config_tmpl + name_suffix + config_suffix
        exp_names.append(config_tmpl + name_suffix)
        config_path = osp.join(work_dir, config_name)

        with open(config_path, 'w') as fout:
            fout.write(cur_cfg.pretty_text)

        other_args_dict_ = cp.deepcopy(other_args_dict)
        other_args_dict_['work-dir'] = [work_dir]

        other_args_str = args2string(other_args_dict_)

        common_args = ['--launcher', launcher] + other_args_str.split()

        if launcher == 'none':
            if gpus:
                cmd = [
                    'python', train_script, config_path, '--gpus',
                    str(gpus)
                ] + common_args
            else:
                cmd = ['python', train_script, config_path, '--device', 'cpu'
                       ] + common_args
        elif launcher == 'pytorch':
            cmd = [
                'python', '-m', 'torch.distributed.launch',
                f'--nproc_per_node={gpus}', f'--master_port={port}',
                train_script, config_path
            ] + common_args
        elif launcher == 'slurm':
            parsed_srun_args = srun_args.split() if srun_args else []
            cmd = [
                'srun', '-p', f'{partition}', f'--gres=gpu:{gpus_per_node}',
                f'--ntasks={gpus}', f'--ntasks-per-node={gpus_per_node}',
                f'--cpus-per-task={cpus_per_task}', '--kill-on-bad-exit=1'
            ] + parsed_srun_args + ['python', '-u', train_script, config_path
                                    ] + common_args

        cmds.append(cmd)

    time.sleep(5)
    succeed_list, fail_list = [], []
    if launcher in ['none', 'pytorch']:
        for cmd, exp_name in zip(cmds, exp_names):
            cmd_text = ' '.join(cmd)
            click.echo(f'Training command for exp {exp_name} is {cmd_text}. ')

            ret = subprocess.check_call(
                cmd, env=dict(os.environ, MASTER_PORT=str(port)))
            if ret == 0:
                click.echo(f'Exp {exp_name} finished successfully.')
                succeed_list.append(exp_name)
            else:
                echo_error('Training not finished successfully.')
                fail_list.append(exp_name)

    elif launcher == 'slurm':

        with Executor(max_workers=max_jobs) as executor:

            for exp, ret in zip(exp_names,
                                executor.map(subprocess.check_call, cmds)):
                if ret == 0:
                    click.echo(f'Exp {exp} finished successfully.')
                    succeed_list.append(exp)
                else:
                    echo_error(f'Exp {exp} not finished successfully.')
                    fail_list.append(exp)

    if len(fail_list):
        msg = ('The following experiments in hyper parameter search '
               f'failed: \n {fail_list}')
        return False, msg
    else:
        msg = ('The hyper parameter search finished successfully.'
               f'Experiment list: \n {succeed_list}')
        return True, msg
