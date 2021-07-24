import importlib
import os.path as osp
import pkg_resources
from typing import List, Tuple

import click
from tabulate import tabulate

from mim.utils import parse_home_page


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
            if pkg.has_metadata('top_level.txt'):
                module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
                if not module_name:
                    continue

                home_page = parse_home_page(pkg_name)
                if not home_page:
                    home_page = pkg.location

                # rename the model_zoo.yml to model-index.yml but support both
                # of them for backward compatibility
                possible_metadata_paths = [
                    osp.join(pkg.location, module_name, 'model-index.yml'),
                    osp.join(pkg.location, module_name, 'model_zoo.yml')
                ]
                if pkg_name.startswith('mmcv') or any(
                        map(osp.exists, possible_metadata_paths)):
                    pkgs_info.append((pkg_name, pkg.version, home_page))
    return pkgs_info
