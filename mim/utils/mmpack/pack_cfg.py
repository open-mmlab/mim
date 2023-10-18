# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import logging
import os
import os.path as osp
import shutil
import signal
import sys
import traceback
from datetime import datetime
from typing import Optional, Union

from mmengine.config import Config, ConfigDict
from mmengine.hub import get_config
from mmengine import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import get_installed_path, mkdir_or_exist

from .common import _import_pack_str, _init_str
from .utils import *  # noqa: F401,F403
from .utils import (
    _get_all_files,
    _postprocess_importfrom_module_to_pack,
    _postprocess_registry_locations,
    _replace_config_scope_to_pack,
    _wrapper_all_registries_build_func,
)


def export_from_cfg(cfg: Union[str, ConfigDict],
                    export_root_dir: Optional[str] = None,
                    fast_test: Optional[bool] = False,
                    model_only: Optional[bool] = False,
                    keep_log: Optional[bool] = False):
    """A function to pack the minimum available package according to config
    file.

    Args:
        cfg  (:obj:`ConfigDict` or str): Config file for packing the
            minimum package.
        export_root_dir (str, optional): The pack directory to save the packed package.
        fast_test (bool, optional): Trun to fast testing mode. Defaults to "False".
    """
    # delete the uncomplete export package when keyboard interupt
    signal.signal(signal.SIGINT, lambda sig, frame: keyboardinterupt_handler(
        sig, frame, export_root_dir))  # type: ignore[arg-type]

    # get config
    if isinstance(cfg, str):
        if '::' in cfg:
            cfg = get_config(cfg)
        else:
            cfg = Config.fromfile(cfg)

    origin_cfg = copy.deepcopy(cfg)
    default_scope = cfg.get('default_scope', 'mmengine')

    # automatically generate ``export_root_dir`` and ``export_module_dir``
    if export_root_dir is None:
        export_root_dir = f"pack_from_{default_scope}_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}"    
    
    if osp.sep in export_root_dir:
        export_path = osp.dirname(export_root_dir)
    else:
        export_path = os.getcwd()
    
    export_log_dir = osp.join(export_path, 'export_log')
    mkdir_or_exist(export_log_dir)
    
    export_logger = MMLogger.get_instance(
        "export",
        log_file=osp.join(export_log_dir, 'export.log'),
    )

    export_module_dir = osp.join(export_root_dir, 'pack')
    if osp.exists(export_module_dir):
        shutil.rmtree(export_module_dir)

    # export config
    if '.mim/' in cfg.filename:
        cfg_path = osp.join(export_module_dir, cfg.filename.split('.mim/')[-1])
    else:
        cfg_path = osp.join(
            osp.join(export_module_dir, 'configs'),
            cfg.filename.split('/')[-1])
    mkdir_or_exist(osp.dirname(cfg_path))

    # NOTE: use less data for faster testing
    fast_test_mode(cfg, fast_test)

    # transform to default_scope
    init_default_scope(default_scope)

    # wrap ``Registry.build()`` for exporting modules
    _wrapper_all_registries_build_func(
        export_module_dir=export_module_dir, scope=default_scope)

    print_log(
        f"[ Export Package Name ]: {export_root_dir}\n"
        f"    package from config: {cfg_path}\n",
        logger='export',
        level=logging.INFO
    )

    cfg['work_dir'] = export_log_dir  # creat temp work_dirs for export
    # use runner to export all needed modules
    runner = Runner.from_cfg(cfg)

    
    if model_only:
        _replace_config_scope_to_pack(cfg)
        cfg.dump(cfg_path)
        print_log(
            f"[ Export Package Name ]: {osp.join(os.getcwd(), export_root_dir)}\n",
            logger='export',
            level=logging.INFO
        )
        return 0
    
    try:
        runner.build_train_loop(cfg.train_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path),
                          'train_dataloader')

    try:
        if 'val_cfg' in cfg and cfg.val_cfg is not None:
            runner.build_val_loop(cfg.val_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path),
                          'val_dataloader')

    try:
        if 'test_cfg' in cfg and cfg.test_cfg is not None:
            runner.build_test_loop(cfg.test_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path),
                          'test_dataloader')

    if 'optim_wrapper' in cfg and cfg.optim_wrapper is not None:
        runner.optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
    if 'param_scheduler' in cfg and cfg.param_scheduler is not None:
        runner.build_param_scheduler(cfg.param_scheduler)

    # add ``__init__.py`` to all dirs, for transferring directories
    # to be modules
    for directory, _, _ in os.walk(export_module_dir):
        if not osp.exists(osp.join(directory, '__init__.py')) \
                and 'configs' not in directory \
                and not directory.endswith(f'{export_module_dir}/'):
            with open(osp.join(directory, '__init__.py'), 'w') as f:
                f.write(_init_str)

    # postprocess for ``pack/registry.py``
    _postprocess_registry_locations(export_root_dir)

    # postprocess for ImportFrom Node, turn to import from export path
    all_export_files = _get_all_files(export_module_dir)
    for file in all_export_files:
        _postprocess_importfrom_module_to_pack(file)

    # get tools from web
    tools_dir = osp.join(export_root_dir, 'tools')
    mkdir_or_exist(tools_dir)

    pack_tools(
        'train.py',
        scope=default_scope,
        path=osp.join(export_root_dir, 'tools/train.py'),
        auto_import=True)
    pack_tools(
        'test.py',
        scope=default_scope,
        path=osp.join(export_root_dir, 'tools/test.py'),
        auto_import=True)

    # TODO: get demo.py

    if not keep_log:
        shutil.rmtree(cfg['work_dir'])

    _replace_config_scope_to_pack(cfg)
    cfg.dump(cfg_path)
    print_log(
        f"[ Export Package Name ]: {osp.join(os.getcwd(), export_root_dir)}\n",
        logger='export',
        level=logging.INFO
    )
    return 0


def keyboardinterupt_handler(sig: int, frame, export_root_dir: str):
    """Clear uncompleted exported package by interrupting with keyboard."""
    if osp.exists(export_root_dir):
        shutil.rmtree(export_root_dir)

    sys.exit(-1)


def error_postprocess(export_root_dir: str, export_log_dir: str,
                      cfg_name: str, error_key: str):
    """Print Debug message when package can't successfully export for missing
    datasets.

    Args:
        export_root_dir (str): _description_
        absolute_cfg_path (str): _description_
        origin_cfg (ConfigDict): _description_
        error_key (str): _description_
        logger (_type_): _description_
    """
    if osp.exists(export_root_dir):
        shutil.rmtree(export_root_dir)

    traceback.print_exc()

    # print(f"[\033[94m Debug \033[0m] The data root of '{error_key}' "
    #         f"is not found. Please modify the 'data_root' in "
    #         f"duplicate config '\033[1m{osp.basename(absolute_cfg_path)}\033[0m'.")

    # TODO: modified to print_log
    # Examples:
    # >>> 09/29 16:45:56 - mmengine - ERROR - /home/panguoping/mm/only_for_test/mmengine/mmengine/logging/logger.py - print_log - 350 - The data root of 'train_dataloader'is not found. Please modify the 'data_root' in duplicate config 'gl_8xb12_celeba-256x256.py'.
    print_log(
        f"The data root of '{error_key}'"
        f" is not found. Please modify the 'data_root' in "
        f"duplicate config '{export_log_dir}/{cfg_name}' or use"
        f" '--model_only' to export model only.",
        logger='export',
        level=logging.ERROR)

    sys.exit(-1)


def pack_tools(tool_name: str,
               scope: str,
               path: str,
               auto_import: Optional[bool] = False):
    """pack tools from installed repo.

    Args:
        tool_name (str): Tool name in repos' tool dir.
        scope (str): The scope of repo.
        path (str): Path to save tool.
        auto_import (bool, optional): Automatically add "import pack" to the
            tool file. Defaults to "False"
    """
    pkg_root = get_installed_path(scope)

    if os.path.exists(path):
        os.remove(path)

    # tools will be put in package/.mim in PR #68
    tool_script = osp.join(pkg_root, '.mim', 'tools', tool_name)
    if not osp.exists(tool_script):
        tool_script = osp.join(pkg_root, 'tools', tool_name)

    shutil.copy(tool_script, path)

    # automatically import the pack modules
    if auto_import:
        with open(path, 'r+') as f:
            lines = f.readlines()
            code = ''.join(lines[:1] + [_import_pack_str] + lines[1:])
            f.seek(0)
            f.write(code)
            f.truncate()


def fast_test_mode(cfg, fast_test: bool = False):
    """Use less data for faster testing.

    Args:
        cfg (ConfigDict): Config of export package.
        fast_test (bool, optional): Fast testing mode. Defaults to False.
    """
    if fast_test:
        # for batch_norm using at least 2 data
        if 'dataset' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset.dataset.indices = [0, 1]
        else:
            cfg.train_dataloader.dataset.indices = [0, 1]
        cfg.train_dataloader.batch_size = 2

        if cfg.get('test_dataloader') is not None:
            cfg.test_dataloader.dataset.indices = [0, 1]
            cfg.test_dataloader.batch_size = 2

        if cfg.get('val_dataloader') is not None:
            cfg.val_dataloader.dataset.indices = [0, 1]
            cfg.val_dataloader.batch_size = 2

        if (cfg.train_cfg.get('type') == 'IterBasedTrainLoop') \
                or (cfg.train_cfg.get('by_epoch') is None
                    and cfg.train_cfg.get('type') != 'EpochBasedTrainLoop'):
            cfg.train_cfg.max_iters = 2
        else:
            cfg.train_cfg.max_epochs = 2

        cfg.train_cfg.val_interval = 1
        cfg.default_hooks.logger.interval = 1

        if 'param_scheduler' in cfg and cfg.param_scheduler is not None:
            if isinstance(cfg.param_scheduler, list):
                for lr_sc in cfg.param_scheduler:
                    lr_sc.begin = 0
                    lr_sc.end = 2
            else:
                cfg.param_scheduler.begin = 0
                cfg.param_scheduler.end = 2
