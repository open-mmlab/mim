# Copyright (c) OpenMMLab. All rights reserved.
import os
import tarfile
import tempfile
import typing
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional, Tuple

import click
from pip._internal.commands import create_command

from mim.utils import (
    PKG2PROJECT,
    WHEEL_URL,
    echo_warning,
    get_torch_cuda_version,
)


@click.command(
    'install',
    context_settings=dict(ignore_unknown_options=True),
)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@click.option(
    '-i',
    '--index-url',
    '--pypi-url',
    'index_url',
    help='Base URL of the Python Package Index (default %default). '
    'This should point to a repository compliant with PEP 503 '
    '(the simple repository API) or a local directory laid out '
    'in the same format.')
def cli(args: Tuple[str], index_url: Optional[str] = None) -> None:
    """Install packages.

    You can use `mim install` in the same way you use `pip install`!

    And `mim install` will **install the 'mim' extra requirements**
    for OpenMMLab package if needed.

    \b
    Example:
        > mim install mmdet mmcls
        > mim install git+https://github.com/open-mmlab/mmdetection.git
        > mim install -r requirements.txt
        > mim install -e <path>
        > mim install mmdet -i <url> -f <url>
        > mim install mmdet --extra-index-url <url> --trusted-host <hostname>
    """
    install(list(args), index_url=index_url)


def install(install_args: List[str], index_url: Optional[str] = None) -> Any:
    """Install the package via pip and add 'mim' extra requirements for
    OpenMMLab package during pip install process.

    Args:
        install_args (list): List of arguments passed to `pip install`.
        index_url (str, optional): The pypi index url.
    """
    # add mmcv-full find links by default
    install_args += ['-f', get_mmcv_full_find_link()]

    if index_url is not None and '-i' not in install_args:
        install_args += ['-i', index_url]

    patch_mm_distribution: Callable = patch_pkg_resources_distribution
    try:
        # pip>=22.1 have two distribution backend: pkg_resources and importlib.
        from pip._internal.metadata import _should_use_importlib_metadata  # type: ignore # isort: skip # noqa: E501
        if _should_use_importlib_metadata():
            patch_mm_distribution = patch_importlib_distribution
    except ImportError:
        pass

    with patch_mm_distribution(index_url):
        # We can use `create_command` method since pip>=19.3 (2019/10/14).
        return create_command('install').main(install_args)


def get_mmcv_full_find_link() -> str:
    """Get the mmcv-full find link corresponding to the current environment."""
    torch_v, cuda_v = get_torch_cuda_version()
    major, minor, *_ = torch_v.split('.')
    torch_v = '.'.join([major, minor, '0'])

    if cuda_v.isdigit():
        cuda_v = f'cu{cuda_v}'
    find_link = WHEEL_URL['mmcv-full'].format(
        cuda_version=cuda_v, torch_version=f'torch{torch_v}')
    return find_link


@contextmanager
def patch_pkg_resources_distribution(
        index_url: Optional[str] = None) -> Generator:
    """A patch for `pip._vendor.pkg_resources.Distribution`.

    Since the old version of the OpenMMLab package did not add the 'mim' extra
    requirements to the release distribution, we need to hack the Distribution
    and manually fetch the 'mim' requirements from `mminstall.txt`.

    This patch works with 'pip<22.1' and 'pip>=22.1,python<3.11'.

    Args:
        index_url (str, optional): The pypi index url that pass to
            `get_mmdeps_from_mmpkg`.
    """
    from pip._vendor.pkg_resources import Distribution, parse_requirements

    origin_requires = Distribution.requires

    def patched_requires(self, extras=()):
        deps = origin_requires(self, extras)
        if self.project_name not in PKG2PROJECT or self.project_name == 'mmcv-full':  # noqa: E501
            return deps

        if 'mim' in self.extras:
            mim_extra_requires = origin_requires(self, ('mim', ))
            deps += mim_extra_requires
        else:
            if not hasattr(self, '_mm_deps'):
                assert self.version is not None
                mmpkg = f'{self.project_name}=={self.version}'
                mmdeps_text = get_mmdeps_from_mmpkg(mmpkg, index_url)
                self._mm_deps = list(parse_requirements(mmdeps_text))
                echo_warning(
                    "Get 'mim' extra requirements from `mminstall.txt` "
                    f'for {self}: {[str(i) for i in self._mm_deps]}.')
            deps += self._mm_deps
        return deps

    Distribution.requires = patched_requires  # type: ignore
    yield

    Distribution.requires = origin_requires  # type: ignore
    return


@contextmanager
def patch_importlib_distribution(index_url: Optional[str] = None) -> Generator:
    """A patch for `pip._internal.metadata.importlib.Distribution`.

    Since the old version of the OpenMMLab package did not add the 'mim' extra
    requirements to the release distribution, we need to hack the Distribution
    and manually fetch the 'mim' requirements from `mminstall.txt`.

    This patch works with 'pip>=22.1,python>=3.11'.

    Args:
        index_url (str, optional): The pypi index url that pass to
            `get_mmdeps_from_mmpkg`.
    """
    from pip._internal.metadata.importlib import Distribution
    from pip._internal.metadata.importlib._dists import Requirement

    origin_iter_dependencies = Distribution.iter_dependencies

    def patched_iter_dependencies(self, extras=()):
        deps = list(origin_iter_dependencies(self, extras))
        if self.canonical_name not in PKG2PROJECT or self.canonical_name == 'mmcv-full':  # noqa: E501
            return deps

        if 'mim' in self.iter_provided_extras:
            mim_extra_requires = list(
                origin_iter_dependencies(self, ('mim', )))
            deps += mim_extra_requires
        else:
            if not hasattr(self, '_mm_deps'):
                assert self.version is not None
                mmpkg = f'{self.canonical_name}=={self.version}'
                mmdeps_text = get_mmdeps_from_mmpkg(mmpkg, index_url)
                self._mm_deps = [
                    Requirement(req) for req in mmdeps_text.splitlines()
                ]
            deps += self._mm_deps
        return deps

    Distribution.iter_dependencies = patched_iter_dependencies
    yield

    Distribution.iter_dependencies = origin_iter_dependencies
    return


@typing.no_type_check
def get_mmdeps_from_mmpkg(mmpkg: str, index_url: Optional[str] = None) -> str:
    """Get 'mim' extra requirements for a given OpenMMLab package from
    `mminstall.txt`.

    Args:
        mmpkg (str): The OpenMMLab package name, optionally with a version
            specifier. e.g. 'mmdet', 'mmdet==2.25.0'.
        index_url (str, optional): The pypi index url, if given, will be used
            in `pip download`.

    Returns:
        (str): The text content read from `mminstall.txt`, returns an empty
            string if anything goes wrong.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        download_args = [
            mmpkg, '-d', temp_dir, '--no-deps', '--no-binary', ':all:'
        ]
        if index_url is not None:
            download_args += ['-i', index_url]
        status_code = create_command('download').main(download_args)
        if status_code != 0:
            echo_warning(
                f'pip download failed with arguments: {download_args}')
            return ''
        mmpkg_tar_fpath = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        with tarfile.open(mmpkg_tar_fpath) as tarf:
            mmdeps_fpath = tarf.members[0].name + '/requirements/mminstall.txt'
            mmdeps_member = tarf._getmember(name=mmdeps_fpath)
            if mmdeps_member is None:
                echo_warning(f'{mmdeps_fpath} not found in {mmpkg_tar_fpath}')
                return ''
            tarf.fileobj.seek(mmdeps_member.offset_data)
            mmdeps_content = tarf.fileobj.read(mmdeps_member.size).decode()
    return mmdeps_content
