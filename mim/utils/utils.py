import functools
import hashlib
import importlib
import os
import os.path as osp
import pkg_resources
import re
import subprocess
import tarfile
import typing
from collections import defaultdict
from distutils.version import LooseVersion
from email.parser import FeedParser
from pkg_resources import get_distribution, parse_version
from typing import Any, List, Optional, Tuple, Union

import click
import requests
from requests.exceptions import InvalidURL, RequestException, Timeout
from requests.models import Response

from .default import PKG2PROJECT


def parse_url(url: str) -> Tuple[str, str]:
    """Parse username and repo from url.

    Args:
        url (str): Url for parsing username and repo name.

    Example:
        >>> parse_url('https://github.com/open-mmlab/mmcv.git')
        'open-mmlab', 'mmcv'
        >>> parse_ulr('git@github.com:open-mmlab/mmcv.git')
        'open-mmlab', 'mmcv'
    """
    if url.startswith('git@'):
        res = url.split(':')[-1].split('/')
    elif 'git' in url:
        res = url.split('/')[-2:]
    else:
        raise ValueError(highlighted_error(f'{url} is invalid.'))

    username = res[0]
    repo = res[1].split('.')[0]
    return username, repo


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def ensure_installation(func):
    """A decorator to make sure a package has been installed.

    Before invoking those functions which depend on installed package, the
    decorator makes sure the package has been installed.
    """

    @functools.wraps(func)
    def wrapper(package):
        if not is_installed(package):
            raise RuntimeError(
                highlighted_error(f'{package} is not installed.'))
        return func(package)

    return wrapper


@ensure_installation
def parse_home_page(package: str) -> Optional[str]:
    """Parse home page from package metadata.

    Args:
        package (str): Package to parse home page.
    """
    home_page = None
    pkg = get_distribution(package)
    if pkg.has_metadata('METADATA'):
        metadata = pkg.get_metadata('METADATA')
        feed_parser = FeedParser()
        feed_parser.feed(metadata)
        home_page = feed_parser.close().get('home-page')
    return home_page


def get_github_url(package: str) -> str:
    """Get github url.

    Args:
        package (str): Name of package, like mmcls.

    Example:
        >>> get_github_url('mmcls')
        'https://github.com/open-mmlab/mmclassification.git'
    """
    home_page = None
    if is_installed(package):
        home_page = parse_home_page(package)

    if not home_page:
        try:
            pkg_info = get_package_info_from_pypi(package)
            home_page = pkg_info['info'].get('home_page')
        except Exception:
            pass

    if home_page:
        if home_page.endswith('.git'):
            github_url = home_page
        elif home_page.endswith('.com'):
            github_url = home_page.replace('.com', '.git')
        else:
            github_url = home_page + '.git'
        return github_url
    else:
        raise ValueError(
            highlighted_error(f'Failed to get github url of {package}.'))


def get_content_from_url(url: str,
                         timeout: int = 15,
                         stream: bool = False) -> Response:
    """Get content from url.

    Args:
        url (str): Url for getting content.
        timeout (int): Set the socket timeout. Default: 15.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=stream)
    except InvalidURL as err:
        raise highlighted_error(err)  # type: ignore
    except Timeout as err:
        raise highlighted_error(err)  # type: ignore
    except RequestException as err:
        raise highlighted_error(err)  # type: ignore
    except Exception as err:
        raise highlighted_error(err)  # type: ignore
    return response


@typing.no_type_check
def download_from_file(url: str,
                       dest_path: str,
                       hash_prefix: Optional[str] = None) -> None:
    """Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download.
        dest_path (str): Path where object will be saved.
        hash_prefix (string, optional): If not None, the SHA256 downloaded
            file should start with `hash_prefix`. Default: None.
    """
    if hash_prefix is not None:
        sha256 = hashlib.sha256()

    response = get_content_from_url(url, stream=True)
    size = int(response.headers.get('content-length'))
    with open(dest_path, 'wb') as fw:
        content_iter = response.iter_content(chunk_size=1024)
        with click.progressbar(content_iter, length=size / 1024) as chunks:
            for chunk in chunks:
                if chunk:
                    fw.write(chunk)
                    fw.flush()
                    if hash_prefix is not None:
                        sha256.update(chunk)

    if hash_prefix is not None:
        digest = sha256.hexdigest()
        if digest[:len(hash_prefix)] != hash_prefix:
            raise RuntimeError(
                highlighted_error(
                    f'invalid hash value, expected "{hash_prefix}", but got '
                    f'"{digest}"'))


def split_package_version(package: str) -> Tuple[str, ...]:
    """Split the package which maybe contains version info.

    Args:
        package (str): Name of package to split.

    Example:
        >>> split_package_version('mmcls')
        'mmcls', ''
        >>> split_package_version('mmcls=0.11.0')
        'mmcls', '0.11.0'
        >>> split_package_version('mmcls==0.11.0')
        'mmcls', '0.11.0'
    """
    if '=' in package:
        return tuple(re.split(r'=+', package))
    else:
        return package, ''


def get_package_version(repo_root: str) -> Tuple[str, str]:
    """Get package and version from local repo.

    Args:
        repo_root (str): Directory of repo.
    """
    for file_name in os.listdir(repo_root):
        version_path = osp.join(repo_root, file_name, 'version.py')
        if osp.exists(version_path):
            with open(version_path, 'r', encoding='utf-8') as f:
                exec(compile(f.read(), version_path, 'exec'))
            return file_name, locals()['__version__']

    return '', ''


@ensure_installation
def get_installed_version(package: str) -> str:
    """Get the version of package from local environment.

    Args:
        package (str): Name of package.
    """
    return get_distribution(package).version


def get_package_info_from_pypi(package: str, timeout: int = 15) -> dict:
    """Get packege information from pypi.

    Args:
        package (str): Package to get information.
        timeout (int): Set the socket timeout. Default: 15.
    """
    pkg_url = f'https://pypi.org/pypi/{package}/json'
    response = get_content_from_url(pkg_url, timeout)
    return response.json()


def get_release_version(package: str, timeout: int = 15) -> List[str]:
    """Get release version from pypi.

    The return list of versions is sorted by ascending order.

    Args:
        package (str): Package to get version.
        timeout (int): Set the socket timeout. Default: 15.
    """
    pkg_info = get_package_info_from_pypi(package, timeout)
    releases = pkg_info['releases']
    return sorted(releases, key=parse_version)


def get_latest_version(package: str, timeout: int = 15) -> str:
    """Get latest version of package.

    Args:
        package (str): Package to get latest version.
        timeout (int): Set the socket timeout. Default: 15.

    Example:
        >>> get_latest_version('mmcv-full')
            '0.11.0'
    """
    release_version = get_release_version(package, timeout)
    return release_version[-1]


def is_version_equal(version1: str, version2: str) -> bool:
    return LooseVersion(version1) == LooseVersion(version2)


@ensure_installation
def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
        return module_name
    else:
        raise ValueError(
            highlighted_error(f'can not infer the module name of {package}'))


@ensure_installation
def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))


def get_torch_cuda_version() -> Tuple[str, str]:
    """Get PyTorch version and CUDA version if it is available.

    Example:
        >>> get_torch_cuda_version()
        '1.8.0', '102'
    """
    try:
        import torch
    except ImportError as err:
        raise err

    torch_v = torch.__version__
    if '+' in torch_v:  # 1.8.1+cu111 -> 1.8.1
        torch_v = torch_v.split('+')[0]

    if torch.cuda.is_available():
        # torch.version.cuda like 10.2 -> 102
        cuda_v = ''.join(torch.version.cuda.split('.'))
    else:
        cuda_v = 'cpu'
    return torch_v, cuda_v


def cast2lowercase(input: Union[list, tuple, str]) -> Any:
    """Cast input into lowercase.

    Example:
        >>> cast2lowercase('Hello World')
        'hello world'
        >>> cast2lowercase(['Hello', 'World'])
        ['hello', 'world']
    """
    inputs = []
    outputs = []
    if isinstance(input, str):
        inputs = [input]
    else:
        inputs = input  # type: ignore

    for _input in inputs:
        outputs.append(_input.lower())

    if isinstance(input, str):
        return outputs[0]
    elif isinstance(input, tuple):
        return tuple(outputs)
    else:
        return outputs


def recursively_find(root: str, base_name: str, followlinks=False) -> list:
    """Recursive list a directory, return all files with a given base_name.

    Args:
        root (str): The root directory to list.
        base_name (str): The base_name.
        followlinks (bool): Follow symbolic links. Defaults to False.

    Return:
        Files with given base_name.
    """
    files = []
    for _root, _, _files in os.walk(root, followlinks=followlinks):
        if base_name in _files:
            files.append(osp.join(_root, base_name))

    return files


def highlighted_error(msg: Union[str, Exception]) -> str:
    return click.style(msg, fg='red', bold=True)  # type: ignore


def color_echo(msg: str, color: str) -> None:
    click.echo(click.style(msg, fg=color))  # type: ignore


def echo_error(msg: Union[str, Exception]) -> None:
    color_echo(msg=msg, color='red')  # type: ignore


def echo_warning(msg: Union[str, Exception]) -> None:
    color_echo(msg=msg, color='yellow')  # type: ignore


def echo_success(msg: str) -> None:
    color_echo(msg=msg, color='green')


def exit_with_error(msg: Union[str, Exception]) -> None:
    echo_error(msg)
    exit(1)


def call_command(cmd: list) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        raise highlighted_error(e)  # type: ignore


def string2args(text: str) -> dict:
    """Parse string to arguments.

    Args:
        text (str): The string to be parsed, which should be of the format:
            "--arg1 value1 value2 --arg2 value1 ... --argn value1".
            Using '=' is also OK, like "--argn=value1". It also support flag
            args like "--arg1".

    Return:
        A dictionary that contains parsed args. Note that the type of values
        will all be strings.

    Example:
        >>> text = '--arg1 value1 value2 --arg2 value3 --arg3 value4'
        >>> string2args(text)
        args = {
            'arg1': [value1, value2],
            'arg2': [value3],
            'arg3': [value4]
        }
    """

    ret: dict = defaultdict(list)
    name = None
    items = text.split()
    for item in items:
        if name is None:
            assert item.startswith('--')
        if item.startswith('--'):
            if name is not None and ret[name] == []:
                ret[name] = bool
            if '=' in item:
                name, value = item[2:].split('=')
                ret[name] = [value]
                name = None
            else:
                name = item[2:]
        else:
            ret[name].append(item)
    if name is not None and ret[name] == []:
        ret[name] = bool
    return ret


def args2string(args: dict) -> str:
    """Convert args dictionary to a string.

    Args:
        args (dict): A dictionary that contains parsed args.

    Return:
        A converted string.

    Example:
        >>> args = {
            'arg1': [value1, value2],
            'arg2': [value3],
            'arg3': [value4]
        }
        >>> args2string(args)
        '--arg1 value1 value2 --arg2 value3 --arg3 value4'
    """
    text = []
    for k in args:
        text.append(f'--{k}')
        if args[k] is not bool:
            text.extend([str(x) for x in args[k]])
    return ' '.join(text)


def get_config(cfg, name):
    """Given the argument name, read the value from the config file.

    The name can be multi-level, like 'optimizer.lr'
    """

    name = name.split('.')
    suffix = ''
    for item in name:
        assert item in cfg, f'attribute {item} not cfg{suffix}'
        cfg = cfg[item]
        suffix += f'.{item}'
    return cfg


def set_config(cfg, name, value):
    """Given the argument name and value, set the value of the config file.

    The name can be multi-level, like 'optimizer.lr'
    """

    name = name.split('.')
    suffix = ''
    for item in name[:-1]:
        assert item in cfg, f'attribute {item} not cfg{suffix}'
        cfg = cfg[item]
        suffix += f'.{item}'

    assert name[-1] in cfg, f'attribute {item} not cfg{suffix}'
    cfg[name[-1]] = value


def extract_tar(tar_path: str, dst: str) -> None:
    """Extract file from tar.

    Args:
        tar_path (str): Path for extracting.
        dst (str): Destination to save file.
    """
    assert tarfile.is_tarfile(tar_path), f'{tar_path} is an invalid path.'

    with tarfile.open(tar_path, 'r') as tar_file:
        tar_file.extractall(dst)


def module_full_name(abbr: str) -> str:
    """Get the full name of the module given abbreviation.

    Args:
        abbr (str): The abbreviation, should be the sub-string of one
            (and only one) supported module.

    Return:
        str: The full name of the corresponding module. If abbr is the
            sub-string of zero / multiple module names, return empty string.
    """
    names = [x for x in PKG2PROJECT if abbr in x]
    if len(names) == 1:
        return names[0]
    elif abbr in names or is_installed(abbr):
        return abbr
    return ''
