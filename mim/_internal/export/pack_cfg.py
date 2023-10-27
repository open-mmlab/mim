# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
import shutil
import signal
import sys
import traceback
from datetime import datetime
from typing import Optional, Union

from mmengine import MMLogger
from mmengine.config import Config, ConfigDict
from mmengine.hub import get_config
from mmengine.logging import print_log
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
                    export_root_dir: str,
                    model_only: Optional[bool] = False,
                    save_log: Optional[bool] = False):
    """A function to pack the minimum available package according to config
    file.

    Args:
        cfg  (:obj:`ConfigDict` or str): Config file for packing the
            minimum package.
        export_root_dir (str, optional): The pack directory to save the
            packed package.
        fast_test (bool, optional): Turn to fast testing mode.
            Defaults to "False".
    """
    # delete the incomplete export package when keyboard interrupt
    signal.signal(signal.SIGINT, lambda sig, frame: keyboardinterupt_handler(
        sig, frame, export_root_dir))  # type: ignore[arg-type]

    # get config
    if isinstance(cfg, str):
        if '::' in cfg:
            cfg = get_config(cfg)
        else:
            cfg = Config.fromfile(cfg)

    default_scope = cfg.get('default_scope', 'mmengine')

    # automatically generate ``export_root_dir`` and ``export_module_dir``
    if export_root_dir == 'auto-dir':
        export_root_dir = f'pack_from_{default_scope}_' + \
            f"{datetime.now().strftime(r'%Y%m%d_%H%M%S')}"

    if osp.sep in export_root_dir:
        export_path = osp.dirname(export_root_dir)
    else:
        export_path = os.getcwd()

    export_log_dir = osp.join(export_path, 'export_log')
    mkdir_or_exist(export_log_dir)

    export_logger = MMLogger.get_instance(  # noqa: F841
        'export',
        log_file=osp.join(export_log_dir, 'export.log'))

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

    # transform to default_scope
    init_default_scope(default_scope)

    # wrap ``Registry.build()`` for exporting modules
    _wrapper_all_registries_build_func(
        export_module_dir=export_module_dir, scope=default_scope)

    print_log(
        f'[ Export Package Name ]: {export_root_dir}\n'
        f'    package from config: {cfg.filename}\n'
        f"    from downstream package: '{default_scope}'\n",
        logger='export',
        level=logging.INFO)

    cfg['work_dir'] = export_log_dir  # creat temp work_dirs for export
    # use runner to export all needed modules
    runner = Runner.from_cfg(cfg)

    # HARD CODE: In order to deal with some module will build in
    # ``before_run`` or ``after_run``, we can call them without need
    # to call "runner.train())".

    # Example:
    #   >>> @HOOKS.register_module()
    #   >>> class EMAHook(Hook):
    #   >>>    ...
    #   >>>   def before_run(self, runner) -> None:
    #   >>>   """Create an ema copy of the model.
    #   >>>   Args:
    #   >>>       runner (Runner): The runner of the training process.
    #   >>>   """
    #   >>>   model = runner.model
    #   >>>   if is_model_wrapper(model):
    #   >>>       model = model.module
    #   >>>   self.src_model = model
    #   >>>   self.ema_model = MODELS.build(
    #   >>>       self.ema_cfg, default_args=dict(model=self.src_model))

    # It need to build ``self.ema_model`` in ``before_run``.

    for hook in runner.hooks:
        hook.before_run(runner)
        hook.after_run(runner)

    def dump():
        cfg['work_dir'] = 'work_dirs'  # recover to default.
        _replace_config_scope_to_pack(cfg)
        cfg.dump(cfg_path)
        print_log(
            f'[ Export Package Name ]: '
            f'{osp.join(os.getcwd(), export_root_dir)}\n',
            logger='export',
            level=logging.INFO)

    if model_only:
        dump()
        return 0

    try:
        runner.build_train_loop(cfg.train_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path), 'train_dataloader')

    try:
        if 'val_cfg' in cfg and cfg.val_cfg is not None:
            runner.build_val_loop(cfg.val_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path), 'val_dataloader')

    try:
        if 'test_cfg' in cfg and cfg.test_cfg is not None:
            runner.build_test_loop(cfg.test_cfg)
    except FileNotFoundError:
        error_postprocess(export_root_dir, export_log_dir,
                          osp.basename(cfg_path), 'test_dataloader')

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

    for tool_name in [
            'train.py', 'test.py', 'dist_train.sh', 'dist_test.sh',
            'slurm_train.sh', 'slurm_test.sh'
    ]:
        pack_tools(
            tool_name=tool_name,
            scope=default_scope,
            tool_dir=tools_dir,
            auto_import=True)

    # TODO: get demo.py

    if not save_log:
        shutil.rmtree(cfg['work_dir'])

    dump()
    return 0


def keyboardinterupt_handler(sig: int, frame, export_root_dir: str):
    """Clear uncompleted exported package by interrupting with keyboard."""
    if osp.exists(export_root_dir):
        shutil.rmtree(export_root_dir)

    sys.exit(-1)


def error_postprocess(export_root_dir: str, export_log_dir: str, cfg_name: str,
                      error_key: str):
    """Print Debug message when package can't successfully export for missing
    datasets.

    Args:
        export_root_dir (str): _description_
        absolute_cfg_path (str): _description_
        origin_cfg (ConfigDict): _description_
        error_key (str): _description_
        logger (_type_): _description_
    """
    from mim.utils import echo_error
    if osp.exists(export_root_dir):
        shutil.rmtree(export_root_dir)

    traceback.print_exc()

    error_msg = f"{'=' * 20} Debug Message {'=' * 20}"\
        f"\nThe data root of '{error_key}' is not found. You can"\
        ' use the below two method to deal with.\n\n'\
        "    >>> Method 1: Please modify the 'data_root' in"\
        f" duplicate config '{export_log_dir}/{cfg_name}'.\n"\
        "    >>> Method 2: Use '--model_only' to export model only.\n\n"\
        "After finishing one of the above steps, you can use 'mim export"\
        f" {export_log_dir}/{cfg_name} [--model-only]' to export again."

    echo_error(error_msg)

    sys.exit(-1)


def pack_tools(tool_name: str,
               scope: str,
               tool_dir: str,
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
    path = osp.join(tool_dir, tool_name)

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
