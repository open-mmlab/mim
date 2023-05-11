# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
from typing import List, Optional

import click
import yaml

from mim.click import (
    OptionEatAll,
    argument,
    get_downstream_package,
    param2lowercase,
)
from mim.commands.search import get_model_info
from mim.utils import (
    DEFAULT_CACHE_DIR,
    call_command,
    download_from_file,
    echo_success,
    get_installed_path,
    highlighted_error,
    is_installed,
    module_full_name,
    split_package_version,
)


@click.command('download')
@argument(
    'package',
    type=str,
    autocompletion=get_downstream_package,
    callback=param2lowercase)
@click.option(
    '--config',
    'configs',
    cls=OptionEatAll,
    help='Config ids to download, such as resnet18_8xb16_cifar10',
    default=None)
@click.option(
    '--dataset',
    'dataset',
    help='dataset name to download, such as coco2017',
    default=None)
@click.option(
    '--ignore-ssl',
    'check_certificate',
    is_flag=True,
    default=True,
    help='Ignore ssl certificate check')
@click.option(
    '--dest', 'dest_root', type=str, help='Destination of saving checkpoints.')
def cli(package: str,
        configs: Optional[List[str]],
        dataset: Optional[str],
        dest_root: Optional[str] = None,
        check_certificate: bool = True) -> None:
    """Download checkpoints from url and parse configs from package.

    \b
    Example:
        > mim download mmcls --config resnet18_8xb16_cifar10
        > mim download mmcls --config resnet18_8xb16_cifar10 --dest .
    """
    download(package, configs, dataset, dest_root, check_certificate)


def download(package: str,
             configs: Optional[List[str]],
             dataset: Optional[str],
             dest_root: Optional[str] = None,
             check_certificate: bool = True) -> List[str]:
    """Download checkpoints from url and parse configs from package.

    Args:
        package (str): Name of package.
        configs (List[str]): List of config ids.
        dest_root (Optional[str]): Destination directory to save checkpoint and
            config. Default: None.
        check_certificate (bool): Whether to check the ssl certificate.
            Default: True.
    """
    full_name = module_full_name(package)
    if full_name == '':
        msg = f"Can't determine a unique package given abbreviation {package}"
        raise ValueError(highlighted_error(msg))
    package = full_name

    if dest_root is None:
        dest_root = DEFAULT_CACHE_DIR

    if configs is not None and dataset is not None:
        raise ValueError(
            'Cannot download config and dataset at the same time!')
    if configs is None and dataset is None:
        raise ValueError('Please specify config or dataset to download!')
    
    if configs is not None:
        return _download_configs(package, configs, dest_root, check_certificate)
    else:
        return _download_dataset(package, dataset, dest_root)


def _download_configs(package: str,
                      configs: Optional[List[str]],
                      dest_root: Optional[str] = None,
                      check_certificate: bool = True) -> List[str]:
    # Create the destination directory if it does not exist.
    if not osp.exists(dest_root):
        os.makedirs(dest_root)

    package, version = split_package_version(package)
    if version:
        raise ValueError(
            highlighted_error('version is not allowed, please type '
                              '"mim download -h" to show the correct way.'))

    if not is_installed(package):
        raise RuntimeError(
            highlighted_error(f'{package} is not installed. Please install it '
                              'first.'))

    checkpoints = []
    model_info = get_model_info(
        package, shown_fields=['weight', 'config'], to_dict=True)
    valid_configs = model_info.keys()
    invalid_configs = set(configs) - set(valid_configs)
    if invalid_configs:
        raise ValueError(
            highlighted_error(f'Expected configs: {valid_configs}, but got '
                              f'{invalid_configs}'))

    try:
        from mmengine import Config
    except ImportError:
        try:
            from mmcv import Config
        except ImportError:
            raise ImportError(
                'Please install mmengine to use the download command: '
                '`mim install mmengine`.')

    for config in configs:
        click.echo(f'processing {config}...')

        checkpoint_urls = model_info[config]['weight']
        for checkpoint_url in checkpoint_urls.split(','):
            filename = checkpoint_url.split('/')[-1]
            checkpoint_path = osp.join(dest_root, filename)
            if osp.exists(checkpoint_path):
                echo_success(f'{filename} exists in {dest_root}')
            else:
                # TODO: check checkpoint hash when all the models are ready.
                download_from_file(
                    checkpoint_url,
                    checkpoint_path,
                    check_certificate=check_certificate)

                echo_success(
                    f'Successfully downloaded {filename} to {dest_root}')

        config_paths = model_info[config]['config']
        for config_path in config_paths.split(','):
            installed_path = get_installed_path(package)
            # configs will be put in package/.mim in PR #68
            possible_config_paths = [
                osp.join(installed_path, '.mim', config_path),
                osp.join(installed_path, config_path)
            ]
            for config_path in possible_config_paths:
                if osp.exists(config_path):
                    config_obj = Config.fromfile(config_path)
                    saved_config_path = osp.join(dest_root, f'{config}.py')
                    config_obj.dump(saved_config_path)
                    echo_success(
                        f'Successfully dumped {config}.py to {dest_root}')
                    checkpoints.append(filename)
                    break
            else:
                raise ValueError(
                    highlighted_error(f'{config_path} is not found.'))

    return checkpoints


def _download_dataset(package: str,
                      dataset: str,
                      dest_root: Optional[str] = None) -> None:
    if not is_installed(package):
        raise RuntimeError(
            f'Please install {package} by ')
    installed_path = get_installed_path(package)
    mim_path = osp.join(installed_path, '.mim')
    dataset_index_path = osp.join(mim_path, 'dataset-index.yml')

    if not osp.exists(dataset_index_path):
        raise FileNotFoundError(
            f'Cannot find dataset-index.yaml in {dataset_index_path}, please update '
            f'{package} to the latest version! If you have already updated it '
            f'and still get this error, please report an issue to {package}')
    with open(dataset_index_path, 'r') as f:
        datasets_meta = yaml.load(f, Loader=yaml.SafeLoader)

    if dataset not in datasets_meta:
        raise KeyError(f'Cannot find {dataset} in {dataset_index_path}. '
                       'here are the available datasets: '
                       '{}'.format("\n".join(datasets_meta.keys())))
    dataset_meta = datasets_meta.get(dataset)
    
    # TODO rename
    script_path = dataset_meta.get('script_path')
    script_path = osp.join(mim_path, script_path)
    src_name = dataset_meta.get('src_name', dataset)
    data_root = dataset_meta['data_root'] if dest_root is None else dest_root
    download_root = dataset_meta['download_root']

    os.makedirs(download_root, exist_ok=True)

    try:
        import sys
        process = subprocess.Popen(
            ['odl', 'get', src_name, '-d', download_root],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr)
        process.wait()
        echo_success(f'Down load {dataset} to {download_root} successfully!')
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        if 'login' in output:
            raise RuntimeError('please login first by "odl login"')
        raise RuntimeError(output)
    
    
    if script_path:
        echo_success('Preprocess data ...')
        os.makedirs(data_root, exist_ok=True)
        # call_command(['chmod', '+x', script_path, osp.curdir])
        call_command([script_path, download_root, data_root])
        echo_success('Finished!')
