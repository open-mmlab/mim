# Copyright (c) OpenMMLab. All rights reserved.

# initialization template for __init__.py
__init__str = """
import os
import os.path as osp

all_files = os.listdir(osp.dirname(__file__))

for file in all_files:
    exec(f'from .{osp.splitext(file)[0]} import *')
"""

# import pack path for tools
_import_pack_str = """
import os
import sys
sys.path.append(os.path.dirname(__file__)+'/../')
import pack

"""

BUILDER_TRANS = {
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

REGISTRY_TYPE = {
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

# module package names transfer to github's repo names
MODULE2GitPACKAGE = {
    # 'mmcls': 'mmcls',
    'mmdet': 'mmdetection',
    'mmdet3d': 'mmdetection3d',
    'mmseg': 'mmsegmentation',
    'mmaction': 'mmaction2',
    'mmtrack': 'mmtracking',
    'mmpose': 'mmpose',
    'mmedit': 'mmedit',
    'mmocr': 'mmocr',
    'mmgen': 'mmgeneration',
    'mmfewshot': 'mmfewshot',
    'mmrazor': 'mmrazor',
    'mmflow': 'mmflow',
    'mmhuman3d': 'mmhuman3d',
    'mmrotate': 'mmrotate',
    'mmselfsup': 'mmselfsup',
    'mmyolo': 'mmyolo',
    'mmpretrain': 'mmpretrain',
    'mmagic': 'mmagic',
}
