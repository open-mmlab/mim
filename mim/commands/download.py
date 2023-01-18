# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import List, Optional

import click

from mim.click import (
    OptionEatAll,
    argument,
    get_downstream_package,
    param2lowercase,
)
from mim.commands.search import get_model_info
from mim.utils import (
    DEFAULT_CACHE_DIR,
    download_from_file,
    echo_success,
    get_installed_path,
    highlighted_error,
    is_installed,
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
    required=True,
    help='Config ids to download, such as resnet18_8xb16_cifar10')
@click.option(
    '--ignore-ssl',
    'check_certificate',
    is_flag=True,
    default=True,
    help='Ignore ssl certificate check')
@click.option(
    '--dest', 'dest_root', type=str, help='Destination of saving checkpoints.')
def cli(package: str,
        configs: List[str],
        dest_root: Optional[str] = None,
        check_certificate: bool = True) -> None:
    """Download checkpoints from url and parse configs from package.

    \b
    Example:
        > mim download mmcls --config resnet18_8xb16_cifar10
        > mim download mmcls --config resnet18_8xb16_cifar10 --dest .
    """
    download(package, configs, dest_root, check_certificate)


def download(package: str,
             configs: List[str],
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
    if dest_root is None:
        dest_root = DEFAULT_CACHE_DIR

    dest_root = osp.abspath(dest_root)

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
