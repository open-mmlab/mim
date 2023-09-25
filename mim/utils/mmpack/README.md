# Export (experimental)

`mim export` is a new function in `mim`, which can export a minimum trainable and testable model package through the config file.

The minimal model compactly combines `model, datasets, engine` and other components. There is no need to search for each module separately based on the config file. Users who do not install from source can also get the model file at a glance.

In addition, lengthy inheritance relationships, such as `CondInstBboxHead -> FCOSHead -> AnchorFreeHead -> BaseDenseHead -> BaseModule` in `mmdetection`, will be directly flattened into `CondInstBboxHead -> BaseModule`. There is no need to open and compare multiple model files when all functions inherited from the parent class are clearly visible.

### Instructions Usage

```bash
mim export config_path

# config_path has the following three optional types:
#Config of downstream repo: Complete the call through repo::configs/xxx.py. For example:
mim export mmdet::configs/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py

# config in a certain folder, for example:
mim export config_dir/mask-rcnn_r101_fpn_1x_coco.py
```

### Minimum model package directory structure

```
minimun_package(Named as pack_from_{repo}_20231212_121212)
|- pack
| |- configs # Configuration folder
| | |- model_name
| | |- xxx.py # Configuration file
| |
| |- models # model folder
| | |- model_file.py
| | |- ...
| |
| |- data # data folder
| |
| |- demo # demo folder
| |
| |-datasets #Dataset class definition
| | |- transforms
| |
| |- registry.py # Registrar
|
|- tools
| |- train.py # training
| |- test.py # test
|
```

### limit

The implementation of `mim export` function depends on `mim/utils/mmpack`. Currently, only some config files of `mmpose/mmdetection/mmagic/mmsegmentation` are supported. And there are some constraints on the downstream repo.

#### For downstream repos

1. It is best to name the config without special symbols, otherwise it cannot be parsed through `mmengine.hub.get_config()`, such as:

   - gn+ws/faster-rcnn_r101_fpn_gn-ws-all_1x_coco.py
   - legacy_1.x/cascade-mask-rcnn_r50_fpn_1x_coco_v1.py

2. For `mmsegmentation`, before using `mim.export` for config in `mmseg`, you should firstly modify it like `mmseg/registry/registry.py -> mmseg/registry.py`, without a directory to wrap `registry.py`

3. It is recommended that the downstream Registry name inherited from mmengine should not be changed. For example, mmagic renamed `EVALUATOR` to `EVALUATORS`

   ```python
   from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR

   # Evaluators to define the evaluation process.
   EVALUATORS = Registry(
       'evaluator',
       parent=MMENGINE_EVALUATOR,
       locations=['mmagic.evaluation'],
   )
   ```

4. In addition, if you add a register that is not in mmengine, such as `DIFFUSION_SCHEDULERS` in mmagic, you need to add a key-value pair in `REGISTRY_TYPE` in `mim/utils/mmpack/common.py` for registering `torch `Module to `DIFFUSION_SCHEDULERS`

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

#### Need to be improve

1. Currently, the inheritance relationship expansion of double parent classes is not supported. Improvements will be made in the future depending on the needs.
2. The config file containing `teacher_config` cannot be executed correctly because `teacher_config` cannot be found in the current path. You can avoid export errors by manually modifying the path pointed by `teacher_config`.
3. When isinstance() is used, if the parent class is just a class in the inheritance chain, the judgment may be False after expansion, because the original inheritance relationship will not be retained.
