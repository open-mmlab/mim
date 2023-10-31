# Patch Utils

## Problem

This patch is mainly to solve the problem that the module cannot be properly registered due to the renaming of the registry in the downstream repo, such as an example of the `mmsegmentation`:

```python
# "mmsegmentation/mmseg/structures/sampler/builder.py"

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmseg.registry import TASK_UTILS

PIXEL_SAMPLERS = TASK_UTILS


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    warnings.warn(
        '``build_pixel_sampler`` would be deprecated soon, please use '
        '``mmseg.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
```

Some modules may use the renamed registry, which makes it difficult for `mim export` to find the original name of the renamed modules.

```python
# "mmsegmentation/mmseg/structures/sampler/ohem_pixel_sampler.py"

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pixel_sampler import BasePixelSampler
from .builder import PIXEL_SAMPLERS


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
            ...
```

## Solution

Therefore, we have currently migrated the necessary modules in `mmpose/mmdetection/mmseg/mmpretrain` listed below, directly to `patch_utils.patch_model` and `patch_utils.patch_task`. In order to build a patch containing renamed registry and special module constructor functions.

```python
"mmdetection/mmdet/models/task_modules/builder.py"
"mmdetection/build/lib/mmdet/models/task_modules/builder.py"

"mmsegmentation/mmseg/models/builder.py"
"mmsegmentation/mmseg/structures/sampler/builder.py"
"mmsegmentation/build/lib/mmseg/models/builder.py"
"mmsegmentation/build/lib/mmseg/structures/sampler/builder.py"

"mmpretrain/mmpretrain/datasets/builder.py"
"mmpretrain/mmpretrain/models/builder.py"
"mmpretrain/build/lib/mmpretrain/datasets/builder.py"
"mmpretrain/build/lib/mmpretrain/models/builder.py"

"mmpose/mmpose/datasets/builder.py"
"mmpose/mmpose/models/builder.py"
"mmpose/build/lib/mmpose/datasets/builder.py"
"mmpose/build/lib/mmpose/models/builder.py"

"mmengine/mmengine/evaluator/builder.py"
"mmengine/mmengine/model/builder.py"
"mmengine/mmengine/optim/optimizer/builder.py"
"mmengine/mmengine/visualization/builder.py"
"mmengine/build/lib/mmengine/evaluator/builder.py"
"mmengine/build/lib/mmengine/model/builder.py"
"mmengine/build/lib/mmengine/optim/optimizer/builder.py"
```
