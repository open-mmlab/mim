# Patch Utils

### 问题

该补丁主要是为了解决下游 repo 中存在对注册器进行重命名导致模块无法被正确注册的问题，如 `mmsegmentation` 中的一个例子：

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

然后在某些模块中可能会使用改名后的注册器，这对于导出后处理很难找到重命名模块原来的名字

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

### 解决方案

因此我们目前已经将如下 `mmpose / mmdetection / mmseg / mmpretrain` 中必要的模块直接迁移到 `patch_utils.patch_model` 和 `patch_utils.patch_task` 中构建一个包含注册器重命名和特殊模块构造函数的补丁。

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
