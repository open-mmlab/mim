import os
import os.path as osp
import tempfile
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
from distutils.version import LooseVersion
from pkg_resources import parse_requirements
from typing import List

import click

from mim.click import get_official_package, param2lowercase
from mim.commands.uninstall import uninstall
from mim.utils import (
    DEFAULT_URL,
    MODULE2PKG,
    PKG2MODULE,
    PKG2PROJECT,
    WHEEL_URL,
    call_command,
    echo_success,
    echo_warning,
    get_installed_version,
    get_latest_version,
    get_package_version,
    get_release_version,
    get_torch_cuda_version,
    highlighted_error,
    is_installed,
    is_version_equal,
    parse_url,
    split_package_version,
    write_installation_records,
)


@click.command('install')
@click.argument(
    'package',
    type=str,
    autocompletion=get_official_package,
    callback=param2lowercase)
@click.option(
    '-f', '--find', 'find_url', type=str, help='Url for finding package.')
@click.option(
    '--default-timeout',
    'timeout',
    type=int,
    default=45,
    help='Set the socket timeout (default 15 seconds).')
@click.option(
    '-y',
    '--yes',
    'is_yes',
    is_flag=True,
    help='Don’t ask for confirmation of uninstall deletions.')
@click.option(
    '--user',
    'is_user_dir',
    is_flag=True,
    help='Install to the Python user install directory')
def cli(
    package: str,
    find_url: str = '',
    timeout: int = 30,
    is_yes: bool = False,
    is_user_dir: bool = False,
) -> None:
    """Install package.

    Example:

    \b
    # install latest version of mmcv-full
    > mim install mmcv-full  # wheel
    # install 1.3.1
    > mim install mmcv-full==1.3.1
    # install master branch
    > mim install mmcv-full -f https://github.com/open-mmlab/mmcv.git

    # install latest version of mmcls
    > mim install mmcls
    # install 0.11.0
    > mim install mmcls==0.11.0  # v0.11.0
    # install master branch
    > mim install mmcls -f https://github.com/open-mmlab/mmclassification.git
    # install local repo
    > git clone https://github.com/open-mmlab/mmclassification.git
    > cd mmclassification
    > mim install .

    # install extension based on OpenMMLab
    > mim install mmcls-project -f https://github.com/xxx/mmcls-project.git
    """
    install(package, find_url, timeout, is_yes=is_yes, is_user_dir=is_user_dir)


def install(package: str,
            find_url: str = '',
            timeout: int = 15,
            is_yes: bool = False,
            is_user_dir: bool = False) -> None:
    """Install a package by wheel or from github.

    Args:
        package (str): The name of installed package, such as mmcls.
        find_url (str): Url for finding package. If finding is not provided,
            program will infer the find_url as much as possible. Default: ''.
        timeout (int): The socket timeout. Default: 15.
        is_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: False.
        is_usr_dir (bool): Install to the Python user install directory for
            environment variables and user configuration. Default: False.
    """
    target_pkg, target_version = split_package_version(package)

    # whether install from local repo
    is_install_local_repo = osp.isdir(osp.abspath(target_pkg)) and not find_url

    # whether install master branch from github
    is_install_master = bool(not target_version and find_url)

    # get target version
    if target_pkg in PKG2PROJECT:
        latest_version = get_latest_version(target_pkg, timeout)
        if target_version:
            if LooseVersion(target_version) > LooseVersion(latest_version):
                error_msg = (f'target_version=={target_version} should not be'
                             f' greater than latest_version=={latest_version}')
                raise ValueError(highlighted_error(error_msg))
        else:
            target_version = latest_version

    # check local environment whether package existed
    if is_install_master or is_install_local_repo:
        pass
    elif is_installed(target_pkg) and target_version:
        existed_version = get_installed_version(target_pkg)
        if is_version_equal(existed_version, target_version):
            echo_warning(f'{target_pkg}=={existed_version} existed.')
            return None
        else:
            if is_yes:
                uninstall(target_pkg, is_yes)
            else:
                confirm_msg = (f'{target_pkg}=={existed_version} has been '
                               f'installed, but want to install {target_pkg}=='
                               f'{target_version}, do you want to uninstall '
                               f'{target_pkg}=={existed_version} and '
                               f'install {target_pkg}=={target_version}? ')
                if click.confirm(confirm_msg):
                    uninstall(target_pkg, True)
                else:
                    echo_warning(f'skip {target_pkg}')
                    return None

    # try to infer find_url if possible
    if not find_url:
        find_url = infer_find_url(target_pkg)

    # whether to write installation records to mmpackage.txt
    is_record = False

    if is_install_local_repo:
        is_record = True
        repo_root = osp.abspath(target_pkg)
        module_name, target_version = get_package_version(repo_root)
        if not module_name:
            raise FileNotFoundError(
                highlighted_error(f'version.py is missed in {repo_root}'))

        target_pkg = MODULE2PKG.get(module_name, module_name)
        if target_pkg == 'mmcv' and os.getenv('MMCV_WITH_OPS', '0') == '1':
            target_pkg = 'mmcv-full'

        echo_success(f'installing {target_pkg} from local repo.')

        install_from_repo(
            repo_root,
            package=target_pkg,
            timeout=timeout,
            is_yes=is_yes,
            is_user_dir=is_user_dir)

    elif find_url and find_url.find('git') >= 0 or is_install_master:
        is_record = True
        install_from_github(target_pkg, target_version, find_url, timeout,
                            is_yes, is_user_dir, is_install_master)
    else:
        # if installing from wheel failed, it will try to install package by
        # building from source if possible.
        is_record = bool(target_pkg in PKG2MODULE)
        try:
            install_from_wheel(target_pkg, target_version, find_url, timeout,
                               is_user_dir)
        except RuntimeError as error:
            if target_pkg in PKG2PROJECT:
                find_url = f'{DEFAULT_URL}/{PKG2PROJECT[target_pkg]}.git'
                if target_version:
                    target_pkg = f'{target_pkg}=={target_version}'
                if is_yes:
                    install(target_pkg, find_url, timeout, is_yes, is_user_dir)
                else:
                    confirm_msg = (f'install {target_pkg} from wheel, but it '
                                   'failed. Do you want to build it from '
                                   'source if possible?')
                    if click.confirm(confirm_msg):
                        install(target_pkg, find_url, timeout, is_yes,
                                is_user_dir)
                    else:
                        raise RuntimeError(
                            highlighted_error(
                                f'Failed to install {target_pkg}.'))
            else:
                raise RuntimeError(highlighted_error(error))

    if is_record:
        if not target_version:
            target_version = get_installed_version(target_pkg)
        if is_install_local_repo:
            find_url = 'local'

        write_installation_records(target_pkg, target_version, find_url)

    echo_success(f'Successfully installed {target_pkg}.')


def infer_find_url(package: str) -> str:
    """Try to infer find_url if possible.

    If package is the official package, the find_url can be inferred.

    Args:
        package (str): The name of package, such as mmcls.
    """
    find_url = ''
    if package in WHEEL_URL:
        torch_v, cuda_v = get_torch_cuda_version()

        # In order to avoid builiding mmcv-full from source, we ignore the
        # difference among micro version because there are usually no big
        # changes among micro version. For example, the mmcv-full built in
        # pytorch 1.8.0 also works on 1.8.1 or other versions.
        major, minor, *_ = torch_v.split('.')
        torch_v = '.'.join([major, minor, '0'])

        if cuda_v.isdigit():
            cuda_v = f'cu{cuda_v}'
        find_url = WHEEL_URL[package].format(
            cuda_version=cuda_v, torch_version=f'torch{torch_v}')
    elif package in PKG2PROJECT:
        find_url = (f'{DEFAULT_URL}/{PKG2PROJECT[package]}.git')

    return find_url


def parse_dependencies(path: str) -> list:
    """Parse dependencies from repo/requirements/mminstall.txt.

    Args:
        path (str): Path of mminstall.txt.
    """

    def _get_proper_version(package, version, op):
        releases = get_release_version(package)
        if op == '>':
            for r_v in releases:
                if LooseVersion(r_v) > LooseVersion(version):
                    return r_v
            else:
                raise ValueError(
                    highlighted_error(f'invalid min version of {package}'))
        elif op == '<':
            for r_v in releases[::-1]:
                if LooseVersion(r_v) < LooseVersion(version):
                    return r_v
            else:
                raise ValueError(
                    highlighted_error(f'invalid max version of {package}'))

    dependencies = []
    with open(path, 'r') as fr:
        for requirement in parse_requirements(fr):
            pkg_name = requirement.project_name
            min_version = ''
            max_version = ''
            for op, version in requirement.specs:
                if op == '==':
                    min_version = max_version = version
                    break
                elif op == '>=':
                    min_version = version
                elif op == '>':
                    min_version = _get_proper_version(pkg_name, version, '>')
                elif op == '<=':
                    max_version = version
                elif op == '<':
                    max_version = _get_proper_version(pkg_name, version, '<')

            dependencies.append([pkg_name, min_version, max_version])

    return dependencies


def install_dependencies(dependencies: List[List[str]],
                         timeout: int = 15,
                         is_yes: bool = False,
                         is_user_dir: bool = False) -> None:
    """Install dependencies, such as mmcls depends on mmcv.

    Args:
        dependencies (list): The list of dependency.
        timeout (int): The socket timeout. Default: 15.
        is_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: False.
        is_usr_dir (bool): Install to the Python user install directory for
            environment variables and user configuration. Default: False.
    """
    for target_pkg, min_v, max_v in dependencies:
        target_version = max_v
        latest_version = get_latest_version(target_pkg, timeout)
        if not target_version or LooseVersion(target_version) > LooseVersion(
                latest_version):
            target_version = latest_version

        if is_installed(target_pkg):
            existed_version = get_installed_version(target_pkg)
            if (LooseVersion(min_v) <= LooseVersion(existed_version) <=
                    LooseVersion(target_version)):
                continue

        echo_success(f'installing dependency: {target_pkg}')

        target_pkg = f'{target_pkg}=={target_version}'

        install(
            target_pkg,
            timeout=timeout,
            is_yes=is_yes,
            is_user_dir=is_user_dir)

    echo_success('Successfully installed dependencies.')


def install_from_repo(repo_root: str,
                      *,
                      package: str = '',
                      timeout: int = 15,
                      is_yes: bool = False,
                      is_user_dir: bool = False):
    """Install package from local repo.

    Args:
        repo_root (str): The root of repo.
        package (str): The name of installed package. Default: ''.
        timeout (int): The socket timeout. Default: 15.
        is_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: False.
        is_usr_dir (bool): Install to the Python user install directory for
            environment variables and user configuration. Default: False.
    """
    # install dependencies. For example,
    # install mmcls should install mmcv first if it is not installed or
    # its(mmcv) verison does not match.
    mminstall_path = osp.join(repo_root, 'requirements', 'mminstall.txt')
    if osp.exists(mminstall_path):
        dependencies = parse_dependencies(mminstall_path)
        if dependencies:
            install_dependencies(dependencies, timeout, is_yes, is_user_dir)

    module_name = PKG2MODULE.get(package, package)
    pkg_root = osp.join(repo_root, module_name)
    src_tool_root = osp.join(repo_root, 'tools')
    dst_tool_root = osp.join(pkg_root, 'tools')
    src_config_root = osp.join(repo_root, 'configs')
    dst_config_root = osp.join(pkg_root, 'configs')
    src_model_zoo_path = osp.join(repo_root, 'model_zoo.yml')
    dst_model_zoo_path = osp.join(pkg_root, 'model_zoo.yml')
    if osp.exists(src_tool_root):
        copy_tree(src_tool_root, dst_tool_root)
    if osp.exists(src_config_root):
        copy_tree(src_config_root, dst_config_root)
    if osp.exists(src_model_zoo_path):
        copy_file(src_model_zoo_path, dst_model_zoo_path)

    third_dependencies = osp.join(repo_root, 'requirements', '/build.txt')
    if osp.exists(third_dependencies):
        dep_cmd = [
            'python', '-m', 'pip', 'install', '-r', third_dependencies,
            '--default-timeout', f'{timeout}'
        ]
        if is_user_dir:
            dep_cmd.append('--user')

        call_command(dep_cmd)

    install_cmd = ['python', '-m', 'pip', 'install', repo_root]
    if is_user_dir:
        install_cmd.append('--user')

    # The issue is caused by the import order of numpy and torch
    # Please refer to github.com/pytorch/pytorch/issue/37377
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    if package in WHEEL_URL:
        echo_success(f'compiling {package} with "MMCV_WITH_OPS=1"')
        os.environ['MMCV_WITH_OPS'] = '1'

    call_command(install_cmd)


def install_from_github(package: str,
                        version: str = '',
                        find_url: str = '',
                        timeout: int = 15,
                        is_yes: bool = False,
                        is_user_dir: bool = False,
                        is_install_master: bool = False) -> None:
    """Install package from github.

    Args:
        package (str): The name of installed package, such as mmcls.
        version (str): Version of package. Default: ''.
        find_url (str): Url for finding package. If finding is not provided,
            program will infer the find_url as much as possible. Default: ''.
        timeout (int): The socket timeout. Default: 15.
        is_yes (bool): Don’t ask for confirmation of uninstall deletions.
            Default: False.
        is_usr_dir (bool): Install to the Python user install directory for
            environment variables and user configuration. Default: False.
        is_install_master (bool): Whether install master branch. If it is True,
            process will install master branch. If it is False, process will
            install the specified version. Default: False.
    """
    click.echo(f'installing {package} from {find_url}.')

    _, repo = parse_url(find_url)
    clone_cmd = ['git', 'clone', find_url]
    if not is_install_master:
        clone_cmd.extend(['-b', f'v{version}'])

    with tempfile.TemporaryDirectory() as temp_root:
        repo_root = osp.join(temp_root, repo)
        clone_cmd.append(repo_root)
        call_command(clone_cmd)

        install_from_repo(
            repo_root,
            package=package,
            timeout=timeout,
            is_yes=is_yes,
            is_user_dir=is_user_dir)


def install_from_wheel(package: str,
                       version: str = '',
                       find_url: str = '',
                       timeout: int = 15,
                       is_user_dir: bool = False) -> None:
    """Install wheel from find_url.

    Args:
        package (str): The name of installed package, such as mmcls.
        version (str): Version of package. Default: ''.
        find_url (str): Url for finding package. If finding is not provided,
            program will infer the find_url as much as possible. Default: ''.
        timeout (int): The socket timeout. Default: 15.
        is_usr_dir (bool): Install to the Python user install directory for
            environment variables and user configuration. Default: False.
    """
    click.echo(f'installing {package} from wheel.')

    install_cmd = [
        'python', '-m', 'pip', '--default-timeout', f'{timeout}', 'install'
    ]
    if version:
        install_cmd.append(f'{package}=={version}')
    else:
        install_cmd.append(package)
    if find_url:
        install_cmd.extend(['-f', find_url])
    if is_user_dir:
        install_cmd.append('--user')

    call_command(install_cmd)
