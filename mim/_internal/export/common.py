# Copyright (c) OpenMMLab. All rights reserved.

# initialization template for __init__.py
_init_str = """
import os
import os.path as osp

all_files = os.listdir(osp.dirname(__file__))

for file in all_files:
    if (file.endswith('.py') and file != '__init__.py') or '.' not in file:
        exec(f'from .{osp.splitext(file)[0]} import *')
"""

# import pack path for tools
_import_pack_str = """
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
import pack

"""

OBJECTS_TO_BE_PATCHED = {
    'MODELS': [
        'BACKBONES',
        'NECKS',
        'HEADS',
        'LOSSES',
        'SEGMENTORS',
        'build_backbone',
        'build_neck',
        'build_head',
        'build_loss',
        'build_segmentor',
        'CLASSIFIERS',
        'RETRIEVER',
        'build_classifier',
        'build_retriever',
        'POSE_ESTIMATORS',
        'build_pose_estimator',
        'build_posenet',
    ],
    'TASK_UTILS': [
        'PIXEL_SAMPLERS',
        'build_pixel_sampler',
    ]
}

REGISTRY_TYPES = {
    'runner': 'RUNNERS',
    'runner constructor': 'RUNNER_CONSTRUCTORS',
    'hook': 'HOOKS',
    'strategy': 'STRATEGIES',
    'dataset': 'DATASETS',
    'data sampler': 'DATA_SAMPLERS',
    'transform': 'TRANSFORMS',
    'model': 'MODELS',
    'model wrapper': 'MODEL_WRAPPERS',
    'weight initializer': 'WEIGHT_INITIALIZERS',
    'optimizer': 'OPTIMIZERS',
    'optimizer wrapper': 'OPTIM_WRAPPERS',
    'optimizer wrapper constructor': 'OPTIM_WRAPPER_CONSTRUCTORS',
    'parameter scheduler': 'PARAM_SCHEDULERS',
    'param scheduler': 'PARAM_SCHEDULERS',
    'metric': 'METRICS',
    'evaluator': 'EVALUATOR',  # TODO EVALUATORS in mmagic
    'task utils': 'TASK_UTILS',
    'loop': 'LOOPS',
    'visualizer': 'VISUALIZERS',
    'vis_backend': 'VISBACKENDS',
    'log processor': 'LOG_PROCESSORS',
    'inferencer': 'INFERENCERS',
    'function': 'FUNCTIONS',
}
