# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import tarfile
import tempfile
import typing
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import click
import pip._vendor.pkg_resources
from pip._internal.commands import create_command

from mim.utils import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MMCV_BASE_URL,
    PKG2PROJECT,
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
@click.option(
    '-y',
    '--yes',
    'is_yes',
    is_flag=True,
    help="Don't ask for confirmation of uninstall deletions."
    'Deprecated, will have no effect.')
def cli(
    args: Tuple[str],
    index_url: Optional[str] = None,
    is_yes: bool = False,
) -> None:
    """Install packages.

    You can use `mim install` in the same way you use `pip install`!

    And `mim install` will **install the 'mim' extra requirements**
    for OpenMMLab packages if needed.

    \b
    Example:
        > mim install mmdet mmcls
        > mim install git+https://github.com/open-mmlab/mmdetection.git
        > mim install -r requirements.txt
        > mim install -e <path>
        > mim install mmdet -i <url> -f <url>
        > mim install mmdet --extra-index-url <url> --trusted-host <hostname>

    Here we list several commonly used options.
    For more options, please refer to `pip install --help`.
    """
    if is_yes:
        echo_warning(
            'The `--yes` option has been deprecated, will have no effect.')
    exit_code = install(list(args), index_url=index_url, is_yes=is_yes)
    exit(exit_code)


def install(
    install_args: List[str],
    index_url: Optional[str] = None,
    is_yes: bool = False,
) -> Any:
    """Install packages via pip and add 'mim' extra requirements for OpenMMLab
    packages during pip install process.

    Args:
        install_args (list): List of arguments passed to `pip install`.
        index_url (str, optional): The pypi index url.
        is_yes (bool, optional): Deprecated, will have no effect. Reserved for
            interface compatibility only.
    """

    # Reload `pip._vendor.pkg_resources` so that pip can refresh to get the
    # latest working set in the same process.
    # In some cases, when a package is uninstalled and then installed, the
    # working set is not updated in time, leading to the mistaken belief that
    # the package is already installed.
    #
    # NOTE: Some unpredictable bugs could occurs with `importlib.reload`.
    # A known issues in pip < 22.0: `METADATA not found in /tmp/xxx/xxx.whel`
    # >>> import pip._vendor.pkg_resources
    # >>> import importlib
    # >>> a = pip._vendor.pkg_resources.DistInfoDistribution()
    # >>> type(a) is pip._vendor.pkg_resources.DistInfoDistribution
    # True
    # >>> importlib.reload(pip._vendor.pkg_resources)
    # <module 'pip._vendor.pkg_resources' from '...'>
    # >>> type(a) is pip._vendor.pkg_resources.DistInfoDistribution
    # False  # This will cause some problems!!!
    importlib.reload(pip._vendor.pkg_resources)

    # Get mmcv_base_url from environment variable if exists.
    mmcv_base_url = os.environ.get('MMCV_BASE_URL', None)
    if mmcv_base_url is not None:
        echo_warning('Using the mmcv find base URL from environment variable '
                     f'`MMCV_BASE_URL`: {mmcv_base_url}')
    else:
        mmcv_base_url = DEFAULT_MMCV_BASE_URL

    # Check if `mmcv_base_url` match the pattern: <scheme>://<netloc>
    parse_result = urlparse(mmcv_base_url)
    assert parse_result.scheme, 'Missing URL scheme (http / https). A valid ' \
        f'`MMCV_BASE_URL` example: {DEFAULT_MMCV_BASE_URL}'

    # Mark mmcv find host as trusted if URL scheme is http.
    if parse_result.scheme == 'http':
        install_args += ['--trusted-host', parse_result.netloc]

    # Add mmcv-full find links by default.
    install_args += ['-f', get_mmcv_full_find_link(mmcv_base_url)]

    index_url_opt_names = ['-i', '--index-url', '--pypi-url']
    if any([opt_name in install_args for opt_name in index_url_opt_names]):
        echo_warning(
            'The index url should be passed in via the index_url parameter, '
            'not specified in install_args via -i/--index-url/--pypi-url.')
    if index_url is not None:
        install_args += ['-i', index_url]

    patch_mm_distribution: Callable = patch_pkg_resources_distribution
    try:
        # pip>=22.1 have two distribution backends: pkg_resources and importlib.  # noqa: E501
        from pip._internal.metadata import _should_use_importlib_metadata  # type: ignore # isort: skip # noqa: E501
        if _should_use_importlib_metadata():
            patch_mm_distribution = patch_importlib_distribution
    except ImportError:
        pass

    with patch_mm_distribution(index_url):
        # We can use `create_command` method since pip>=19.3 (2019/10/14).
        status_code = create_command('install').main(install_args)

    check_mim_resources()
    return status_code


def get_mmcv_full_find_link(mmcv_base_url: str) -> str:
    """Get the mmcv-full find link corresponding to the current environment.

    Args:
        mmcv_base_url (str): The base URL of mmcv find link.

    Returns:
        str: The mmcv find links corresponding to the current torch version and
        cuda version.
    """
    torch_v, cuda_v = get_torch_cuda_version()
    major, minor, *_ = torch_v.split('.')
    torch_v = '.'.join([major, minor, '0'])

    if cuda_v.isdigit():
        cuda_v = f'cu{cuda_v}'

    find_link = f'{mmcv_base_url}/mmcv/dist/{cuda_v}/torch{torch_v}/index.html'  # noqa: E501
    return find_link


@contextmanager
def patch_pkg_resources_distribution(
        index_url: Optional[str] = None) -> Generator:
    """A patch for `pip._vendor.pkg_resources.Distribution`.

    Since the old version of the OpenMMLab packages did not add the 'mim' extra
    requirements to the release distribution, we need to hack the Distribution
    and manually fetch the 'mim' requirements from `mminstall.txt`.

    This patch works with 'pip<22.1' and 'pip>=22.1,python<3.11'.

    Args:
        index_url (str, optional): The pypi index url that pass to
            `get_mmdeps_from_mmpkg_pypi`.
    """
    from pip._vendor.pkg_resources import Distribution, parse_requirements

    origin_requires = Distribution.requires

    def patched_requires(self, extras=()):
        deps = origin_requires(self, extras)
        if self.project_name not in PKG2PROJECT or self.project_name == 'mmcv-full':  # noqa: E501
            return deps

        if 'mim' in self.extras:
            mim_extra_requires = origin_requires(self, ('mim', ))
            filter_invalid_marker(mim_extra_requires)
            deps += mim_extra_requires
        else:
            if not hasattr(self, '_mm_deps'):
                assert self.version is not None
                mmdeps_text = get_mmdeps_from_mmpkg(self.project_name,
                                                    self.version, index_url)
                self._mm_deps = list(parse_requirements(mmdeps_text))
                echo_warning(
                    "Get 'mim' extra requirements from `mminstall.txt` "
                    f'for {self}: {[str(i) for i in self._mm_deps]}.')
            deps += self._mm_deps
        return deps

    Distribution.requires = patched_requires  # type: ignore
    try:
        yield
    finally:
        Distribution.requires = origin_requires  # type: ignore


@contextmanager
def patch_importlib_distribution(index_url: Optional[str] = None) -> Generator:
    """A patch for `pip._internal.metadata.importlib.Distribution`.

    Since the old version of the OpenMMLab packages did not add the 'mim' extra
    requirements to the release distribution, we need to hack the Distribution
    and manually fetch the 'mim' requirements from `mminstall.txt`.

    This patch works with 'pip>=22.1,python>=3.11'.

    Args:
        index_url (str, optional): The pypi index url that pass to
            `get_mminstall_from_pypi`.
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
            filter_invalid_marker(mim_extra_requires)
            deps += mim_extra_requires
        else:
            if not hasattr(self, '_mm_deps'):
                assert self.version is not None
                mmdeps_text = get_mmdeps_from_mmpkg(self.canonical_name,
                                                    self.version, index_url)
                self._mm_deps = [
                    Requirement(req) for req in mmdeps_text.splitlines()
                ]
            deps += self._mm_deps
        return deps

    Distribution.iter_dependencies = patched_iter_dependencies  # type: ignore
    try:
        yield
    finally:
        Distribution.iter_dependencies = origin_iter_dependencies  # type: ignore  # noqa: E501


def filter_invalid_marker(extra_requires: List) -> None:
    """Filter out invalid marker in requirements parsed from METADATA.

    More detail can be found at: https://github.com/pypa/pip/issues/11191

    Args:
        extra_requires (list): A list of Requirement parsed from distribution
            METADATA.
    """
    for req in extra_requires:
        if req.marker is None:
            continue
        try:
            req.marker.evaluate()
        except:  # noqa: E722
            req.marker = None


def get_mmdeps_from_mmpkg(mmpkg_name: str,
                          mmpkg_version: str,
                          index_url: Optional[str] = None) -> str:
    """Get 'mim' extra requirements for a given OpenMMLab package from
    `mminstall.txt`.

    If there is a cached `mminstall.txt`, use the cache, otherwise download the
    source distribution package from pypi and extract `mminstall.txt` content.

    Args:
        mmpkg_name (str): The OpenMMLab package name.
        mmpkg_version (str): The OpenMMLab package version.
        index_url (str, optional): The pypi index url that pass to
            `get_mminstall_from_pypi`.

    Returns:
        str: The text content read from `mminstall.txt`, returns an empty
        string if anything goes wrong.
    """
    mmpkg = f'{mmpkg_name}=={mmpkg_version}'
    cache_mminstall_dir = os.path.join(DEFAULT_CACHE_DIR, 'mminstall')
    if not os.path.exists(cache_mminstall_dir):
        os.mkdir(cache_mminstall_dir)
    cache_mminstall_fpath = os.path.join(cache_mminstall_dir, f'{mmpkg}.txt')
    if os.path.exists(cache_mminstall_fpath):
        # use cached `mminstall.txt`
        with open(cache_mminstall_fpath) as f:
            mminstall_content = f.read()
        echo_warning(
            f'Using cached `mminstall.txt` for {mmpkg}: {cache_mminstall_fpath}'  # noqa: E501
        )
    else:
        # fetch `mminstall.txt` content from pypi
        mminstall_content = get_mminstall_from_pypi(mmpkg, index_url=index_url)
        with open(cache_mminstall_fpath, 'w') as f:
            f.write(mminstall_content)
    return mminstall_content


@typing.no_type_check
def get_mminstall_from_pypi(mmpkg: str,
                            index_url: Optional[str] = None) -> str:
    """Get the `mminstall.txt` content for a given OpenMMLab package from PyPi.

    Args:
        mmpkg (str): The OpenMMLab package name, optionally with a version
            specifier. e.g. 'mmdet', 'mmdet==2.25.0'.
        index_url (str, optional): The pypi index url, if given, will be used
            in `pip download`.

    Returns:
        str: The text content read from `mminstall.txt`, returns an empty
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
            echo_warning(f'pip download failed with args: {download_args}')
            exit(status_code)
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


def check_mim_resources() -> None:
    """Check if the mim resource directory exists.

    Newer versions of the OpenMMLab packages have packaged the mim resource
    files into the distribution package, while earlier versions do not.

    If the mim resources file (aka `.mim`) do not exists, log a warning that a
    new version needs to be installed.
    """
    importlib.reload(pip._vendor.pkg_resources)
    for pkg in pip._vendor.pkg_resources.working_set:  # type: ignore
        pkg_name = pkg.project_name
        if pkg_name not in PKG2PROJECT or pkg_name == 'mmcv-full':
            continue
        if pkg.has_metadata('top_level.txt'):
            module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
            installed_path = os.path.join(pkg.location, module_name)
        else:
            installed_path = os.path.join(pkg.location, pkg_name)
        mim_resources_path = os.path.join(installed_path, '.mim')
        if not os.path.exists(mim_resources_path):
            echo_warning(f'mim resources not found: {mim_resources_path}, '
                         f'you can try to install the latest {pkg_name}.')
