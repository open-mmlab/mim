import os.path as osp
from pkg_resources import resource_filename
from typing import List, Optional

import click

from mim.click import OptionEatAll, get_downstream_package, param2lowercase
from mim.commands.search import get_model_info
from mim.utils import (
    DEFAULT_CACHE_DIR,
    PKG2MODULE,
    download_from_file,
    echo_success,
    highlighted_error,
    is_installed,
    split_package_version,
)


@click.command('download')
@click.argument(
    'package',
    type=str,
    autocompletion=get_downstream_package,
    callback=param2lowercase)
@click.option(
    '--config',
    'configs',
    cls=OptionEatAll,
    required=True,
    help='Config ids to download, such as resnet18_b16x8_cifar10')
@click.option(
    '--dest', 'dest_root', type=str, help='Destination of saving checkpoints.')
def cli(package: str,
        configs: List[str],
        dest_root: Optional[str] = None) -> None:
    """Download checkpoints from url and parse configs from package.

    \b
    Example:
        > mim download mmcls --config resnet18_b16x8_cifar10
        > mim download mmcls --config resnet18_b16x8_cifar10 --dest .
    """
    download(package, configs, dest_root)


def download(package: str,
             configs: List[str],
             dest_root: Optional[str] = None) -> List[str]:
    """Download checkpoints from url and parse configs from package.

    Args:
        package (str): Name of package.
        configs (List[str]): List of config ids.
        dest_root (Optional[str]): Destination directory to save checkpoint and
            config. Default: None.
    """
    if dest_root is None:
        dest_root = DEFAULT_CACHE_DIR

    dest_root = osp.abspath(dest_root)

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

    from mmcv import Config

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
                download_from_file(checkpoint_url, checkpoint_path)

                echo_success(
                    f'Successfully downloaded {filename} to {dest_root}')

        config_paths = model_info[config]['config']
        for config_path in config_paths.split(','):
            module_name = PKG2MODULE.get(package, package)
            config_path = resource_filename(module_name, config_path)
            if not osp.exists(config_path):
                raise ValueError(
                    highlighted_error(f'{config_path} is not found.'))

            config_obj = Config.fromfile(config_path)
            saved_config_path = osp.join(dest_root, f'{config}.py')
            config_obj.dump(saved_config_path)
            echo_success(f'Successfully dumped {config}.py to {dest_root}')

            checkpoints.append(filename)

    return checkpoints
