# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Optional

import click
from mmengine.config import Config
from mmengine.hub import get_config

from mim.click import CustomCommand
from mim.utils.mmpack.pack_cfg import export_from_cfg

PYTHON = sys.executable


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


@click.command(
    name='export',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('config', type=str)
@click.option(
    '-p',
    '--export_root_dir',
    type=str,
    help='The root directory name of export packge')
@click.option(
    '-ft',
    '--fast_test',
    is_flag=True,
    help='The fast_test mode using few data for testing')
@click.option(
    '--keep_log',
    is_flag=True,
    help='The fast_test mode using few data for testing')
def cli(config: str,
        export_root_dir: Optional[str] = None,
        fast_test: bool = False,
        keep_log: bool = False) -> None:

    # get config
    if isinstance(config, str):
        if '::' in config:
            config = get_config(config)
        else:
            config = Config.fromfile(config)

    fast_test_mode(config, fast_test)

    export_from_cfg(config, export_root_dir=export_root_dir, keep_log=keep_log)
