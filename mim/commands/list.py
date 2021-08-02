import importlib
import os.path as osp
import pkg_resources
from email.parser import FeedParser
from typing import List, Tuple

import click
from tabulate import tabulate


@click.command('list')
@click.option(
    '--all',
    is_flag=True,
    help='List packages of OpenMMLab projects or all the packages in the '
    'python environment.')
def cli(all: bool = True) -> None:
    """List packages.

    \b
    Example:
        > mim list
        > mim list --all
    """
    table_header = ['Package', 'Version', 'Source']
    table_data = list_package(all=all)
    click.echo(tabulate(table_data, headers=table_header, tablefmt='simple'))


def list_package(all: bool = False) -> List[Tuple[str, ...]]:
    """List packages.

    List packages of OpenMMLab projects or all the packages in the python
    environment.

    Args:
        all (bool): List all installed packages. If all is False, it just lists
            the packages installed by mim. Default: False.
    """
    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)

    pkgs_info: List[Tuple[str, ...]] = []
    for pkg in pkg_resources.working_set:
        pkg_name = pkg.project_name
        if all:
            pkgs_info.append((pkg_name, pkg.version))
        else:
            installed_path = osp.join(pkg.location, pkg_name)
            if not osp.exists(installed_path):
                module_name = None
                if pkg.has_metadata('top_level.txt'):
                    module_name = pkg.get_metadata('top_level.txt').split(
                        '\n')[0]
                if module_name:
                    installed_path = osp.join(pkg.location, module_name)
                else:
                    continue

            home_page = pkg.location
            if pkg.has_metadata('METADATA'):
                metadata = pkg.get_metadata('METADATA')
                feed_parser = FeedParser()
                feed_parser.feed(metadata)
                home_page = feed_parser.close().get('home-page')

            if pkg_name.startswith('mmcv'):
                pkgs_info.append((pkg_name, pkg.version, home_page))
                continue

            possible_metadata_paths = [
                osp.join(installed_path, '.mim', 'model-index.yml'),
                osp.join(installed_path, 'model-index.yml'),
                osp.join(installed_path, '.mim', 'model_zoo.yml'),
                osp.join(installed_path, 'model_zoo.yml')
            ]
            for path in possible_metadata_paths:
                if osp.exists(path):
                    pkgs_info.append((pkg_name, pkg.version, home_page))
                    break

    pkgs_info.sort(key=lambda pkg_info: pkg_info[0])
    return pkgs_info
