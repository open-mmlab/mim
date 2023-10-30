# Export (experimental)

`mim export` 是 `mim` 里面一个新的功能，可以实现通过 config 文件，就能够导出一个最小可训练测试的模型包。

最小模型将 `model、datasets、engine`等组件紧凑地组合在一起，不用再根据 config 文件单独寻找每一个模块，对于非源码安装的用户也能够获取到一目了然的模型文件。

此外，对于模型中冗长的继承关系，如 `mmdetection` 中的 `CondInstBboxHead -> FCOSHead -> AnchorFreeHead -> BaseDenseHead -> BaseModule`，将会被直接展平为 `CondInstBboxHead -> BaseModule`，即无需再在多个模型文件之间跳转比较了，所有继承父类的函数一览无遗。

### 使用方法

```bash
mim export config_path

# config_path 有以下两种可选类型：
# 下游 repo 的 config：通过 repo::configs/xxx.py 来完成调用。例如：
mim export mmdet::configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py

# 在某文件夹下的 config, 例如：
mim export config_dir/mask-rcnn_r101_fpn_1x_coco.py
```

### 最小模型包目录结构

```
minimun_package(Named as pack_from_{repo}_20231212_121212)
|- pack
|   |- configs # 配置文件夹
|   |  |- model_name
|   |      |- xxx.py # 配置文件
|   |
|   |- models # 模型文件夹
|   |  |- model_file.py
|   |  |- ...
|   |
|   |- data # 数据文件夹
|   |
|   |- demo # demo文件夹
|   |
|   |- datasets # 数据集类定义
|   |  |- transforms
|   |
|   |- registry.py # 注册器
|
|- tools
|  |- train.py # 训练
|  |- test.py # 测试
|
```

### 限制

`mim export` 目前只支持 `mmpose`、`mmdetection`、`mmagic` 和 `mmsegmentation` 的部分 config 配置文件，并且对下游算法库有一些约束。

#### 针对下游库

1. config 命名最好**不要有特殊符号**，否则无法通过 `mmengine.hub.get_config()` 进行解析，如：

   - gn+ws/faster-rcnn_r101_fpn_gn-ws-all_1x_coco.py
   - legacy_1.x/cascade-mask-rcnn_r50_fpn_1x_coco_v1.py

2. 针对 `mmsegmentation`, 在使用 `mim.export` 导出 `mmseg` 的 config 之前, 首先需要去掉对于 `registry.py` 的外层文件夹封装， 即修改 `mmseg/registry/registry.py -> mmseg/registry.py`。

3. 建议下游继承于 mmengine 的 Registry 名字不要改动，如 mmagic 中就将 `EVALUATOR` 重新命名为了 `EVALUATORS`

   ```python
   from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR

   # Evaluators to define the evaluation process.
   EVALUATORS = Registry(
       'evaluator',
       parent=MMENGINE_EVALUATOR,
       locations=['mmagic.evaluation'],
   )
   ```

4. 另外，如果添加了 mmengine 中没有的注册器，如 mmagic 中的 `DIFFUSION_SCHEDULERS`，需要在 `mim/_internal/export/common.py` 的 `REGISTRY_TYPE` 中添加键值对，用于注册 `torch` 模块到 `DIFFUSION_SCHEDULERS`

   ```python
   # "mmagic/mmagic/registry.py"
   # modules for diffusion models that support adding noise and denoising
   DIFFUSION_SCHEDULERS = Registry(
       'diffusion scheduler',
       locations=['mmagic.models.diffusion_schedulers'],
   )

   # "mim/utils/mmpack/common.py"
   REGISTRY_TYPE = {
       ...
       'diffusion scheduler': 'DIFFUSION_SCHEDULERS',
       ...
   }
   ```

#### 对于 `mim.export` 功能需要改进的地方

1. 目前还不支持双父类的继承关系展开，后续看需求进行改进

2. 对于用到 `isinstance()` 时，如果父类只是继承链中某个类，可能展开后判断就会为 False，因为并不会保留原有的继承关系

3. 当 config 文件中含有当前文件夹没法被访问到的`数据集路径`，导出可能会失败。目前的临时解决方法是：将原来的 config 文件保存到当前文件夹下，然后需要用户手动修改`数据集路径`为当前路径下的可访问路径。如：`data/ADEChallengeData2016/ -> your_data_dir/ADEChallengeData2016/`
