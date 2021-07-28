import importlib
import pkg_resources
from typing import List, Tuple

import click
from importlib_metadata import files, metadata
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
            home_page = metadata(pkg_name)['Home-page']
            if not home_page:
                home_page = pkg.location

            if pkg_name.startswith('mmcv'):
                pkgs_info.append((pkg_name, pkg.version, home_page))
                continue

            for file in files(pkg_name):
                # rename the model_zoo.yml to model-index.yml but support both
                # of them for backward compatibility.
                filename = file.locate().name
                if filename in ['model-index.yml', 'model_zoo.yml']:
                    pkgs_info.append((pkg_name, pkg.version, home_page))
                    break

    pkgs_info.sort(key=lambda pkg_info: pkg_info[0])
    return pkgs_info
