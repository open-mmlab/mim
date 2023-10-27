# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Optional

import click
from mmengine.config import Config
from mmengine.hub import get_config

from mim._internal.export.pack_cfg import export_from_cfg
from mim.click import CustomCommand

PYTHON = sys.executable


@click.command(
    name='export',
    context_settings=dict(ignore_unknown_options=True),
    cls=CustomCommand)
@click.argument('config', type=str)
@click.argument('export_dir', type=str, default=None)
@click.option(
    '-ft',
    '--fast-test',
    is_flag=True,
    help='The fast_test mode. In order to quickly test if'
    ' there is any error in the export package,'
    ' it only use the first two data of your datasets'
    ' which only be used to train 2 iters/epoches.')
@click.option(
    '--save-log',
    is_flag=True,
    help='The flag to keep the export log of the process. The log of export'
    " process will be save to directory 'export_log'. Default will"
    ' automatically delete the log after export.')
def cli(config: str,
        export_dir: Optional[str] = None,
        fast_test: bool = False,
        save_log: bool = False) -> None:
    """Export package from config file.

    Example:

    \b
    >>> # Export package from downstream config file.
    >>> mim export mmdet::dab_detr/dab-detr_r50_8xb2-50e_coco.py \\
    ... dab_detr
    >>>
    >>> # Export package from specified config file.
    >>> mim export mmdetection/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py dab_detr # noqa: E501
    >>>
    >>> # Use auto-dir. It can auto generate a export dir
    >>> # like：'pack_from_mmdet_20231026_052704.
    >>> mim export mmdet::dab_detr/dab-detr_r50_8xb2-50e_coco.py \\
    ... auto-dir
    >>>
    >>> # Only export the model of config file.
    >>> mim export mmdet::dab_detr/dab-detr_r50_8xb2-50e_coco.py \\
    ... mask_rcnn_package --model-only
    >>>
    >>> # Keep the export log of the process.
    >>> mim export mmdet::dab_detr/dab-detr_r50_8xb2-50e_coco.py \\
    ... mask_rcnn_package --save-log
    >>>
    >>> # Print the help information of export command.
    >>> mim export -h
    """

    # get config
    if isinstance(config, str):
        if '::' in config:
            config = get_config(config)
        else:
            config = Config.fromfile(config)

    fast_test_mode(config, fast_test)

    export_from_cfg(config, export_root_dir=export_dir, save_log=save_log)


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
