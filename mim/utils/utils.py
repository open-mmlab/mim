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
from pkg_resources import parse_version
from typing import Any, List, Optional, Tuple, Union

import click
import requests
from requests.exceptions import InvalidURL, RequestException, Timeout
from requests.models import Response

from .default import DEFAULT_URL, MMPACKAGE_PATH, PKG2MODULE, PKG2PROJECT


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


def get_github_url(package: str) -> str:
    """Get github url.

    Args:
        package (str): Name of package, like mmcls.

    Example:
        >>> get_github_url('mmcls')
        'https://github.com/open-mmlab/mmclassification.git'
    """
    for _package, _, _url in read_installation_records():
        if _package == package and _url != 'local':
            github_url = _url
            break
    else:
        if package not in PKG2PROJECT:
            raise ValueError(
                highlighted_error(f'Failed to get url of {package}.'))

        github_url = f'{DEFAULT_URL}/{PKG2PROJECT[package]}.git'

    return github_url


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


def is_installed(package: str) -> Any:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    module_name = PKG2MODULE.get(package, package)
    return importlib.util.find_spec(module_name)  # type: ignore


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


def get_installed_version(package: str) -> str:
    """Get the version of package from local environment.

    Args:
        package (str): Name of package.
    """
    module_name = PKG2MODULE.get(package, package)

    if not is_installed(module_name):
        raise RuntimeError(highlighted_error(f'{package} is not installed.'))

    module = importlib.import_module(module_name)
    return module.__version__  # type: ignore


def get_release_version(package: str, timeout: int = 15) -> List[str]:
    """Get release version from pypi.

    The return list of versions is sorted by ascending order.

    Args:
        package (str): Package to get version.
        timeout (int): Set the socket timeout. Default: 15.
    """
    pkg_url = f'https://pypi.org/pypi/{package}/json'
    response = get_content_from_url(pkg_url, timeout)
    content = response.json()
    releases = content['releases']
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


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    module_name = PKG2MODULE.get(package, package)
    module = importlib.import_module(module_name)
    return module.__path__[0]  # type: ignore


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


def read_installation_records() -> list:
    """Read installed packages from mmpackage.txt."""
    if not osp.isfile(MMPACKAGE_PATH):
        return []

    seen = set()
    pkgs_info = []
    with open(MMPACKAGE_PATH, 'r') as fr:
        for line in fr:
            line = line.strip()
            package, version, source = line.split(',')
            if not is_installed(package):
                continue

            pkgs_info.append((package, version, source))
            seen.add(package)

    # handle two cases
    # 1. install mmrepos by other ways not mim, such as pip install mmcls
    # 2. existed mmrepos
    for pkg in pkg_resources.working_set:
        pkg_name = pkg.project_name
        if pkg_name not in seen and (pkg_name in PKG2PROJECT
                                     or pkg_name in PKG2MODULE):
            pkgs_info.append((pkg_name, pkg.version, ''))

    return pkgs_info


def write_installation_records(package: str,
                               version: str,
                               source: str = '') -> None:
    """Write installed package to mmpackage.txt."""
    pkgs_info = read_installation_records()
    with open(MMPACKAGE_PATH, 'w') as fw:
        if pkgs_info:
            for _package, _version, _source in pkgs_info:
                if _package != package:
                    fw.write(f'{_package},{_version},{_source}\n')
        fw.write(f'{package},{version},{source}\n')


def remove_installation_records(package: str) -> None:
    """Remove package from mmpackage.txt."""
    pkgs_info = read_installation_records()
    if not pkgs_info:
        with open(MMPACKAGE_PATH, 'w') as fw:
            for _package, _version, _source in pkgs_info:
                if _package != package:
                    fw.write(f'{_package},{_version},{_source}\n')


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


def recursively_find(root: str, base_name: str) -> list:
    """Recursive list a directory, return all files with a given base_name.

    Args:
        root (str): The root directory to list.
        base_name (str): The base_name.

    Return:
        Files with given base_name.
    """
    results = list(os.walk(root))
    files = []
    for tup in results:
        root = tup[0]
        if base_name in tup[2]:
            files.append(osp.join(root, base_name))

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
    supported_pkgs = [
        PKG2MODULE[k] if k in PKG2MODULE else k for k in PKG2PROJECT
    ]
    supported_pkgs = list(set(supported_pkgs))
    names = [x for x in supported_pkgs if abbr in x]
    if len(names) == 1:
        return names[0]
    else:
        return ''
