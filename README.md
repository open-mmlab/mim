# MIM: MIM Installs OpenMMLab Packages

MIM provides a unified interface for launching and installing OpenMMLab projects and their extensions, and managing the OpenMMLab model zoo.

## Major Features

- **Package Management**

  You can use MIM to manage OpenMMLab codebases, install or uninstall them conveniently.

- **Model Management**

  You can use MIM to manage OpenMMLab model zoo, e.g., download checkpoints by name, search checkpoints that meet specific criteria.

- **Unified Entrypoint for Scripts**

  You can execute any script provided by all OpenMMLab codebases with unified commands. Train, test and inference become easier than ever. Besides, you can use `gridsearch` command for vanilla hyper-parameter search.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.1.1 was released in 13/6/2021.

## Customization

You can use `.mimrc` for customization. Now we support customize default values of each sub-command. Please refer to [customization.md](docs/en/customization.md) for details.

## Build custom projects with MIM

We provide some examples of how to build custom projects based on OpenMMLAB codebases and MIM in [MIM-Example](https://github.com/open-mmlab/mim-example).
Without worrying about copying codes and scripts from existing codebases, users can focus on developing new components and MIM helps integrate and run the new project.

## Installation

Please refer to [installation.md](docs/en/installation.md) for installation.

## Command

<details>
<summary>1. install</summary>

[![asciicast](https://asciinema.org/a/416945.svg)](https://asciinema.org/a/416945)

- command

    ```bash
    # install latest version of mmcv-full
    > mim install mmcv-full  # wheel
    # install 1.3.1
    > mim install mmcv-full==1.3.1
    # install master branch
    > mim install mmcv-full -f https://github.com/open-mmlab/mmcv.git

    # install latest version of mmcls
    > mim install mmcls
    # install 0.11.0
    > mim install mmcls==0.11.0  # v0.11.0
    # install master branch
    > mim install mmcls -f https://github.com/open-mmlab/mmclassification.git
    # install local repo
    > git clone https://github.com/open-mmlab/mmclassification.git
    > cd mmclassification
    > mim install .

    # install extension based on OpenMMLab
    mim install mmcls-project -f https://github.com/xxx/mmcls-project.git
    ```

- api

    ```python
    from mim import install

    # install mmcv
    install('mmcv-full')

    # install mmcls
    # install mmcls will automatically install mmcv if it is not installed
    install('mmcv-full', find_url='https://github.com/open-mmlab/mmcv.git')
    install('mmcv-full==1.3.1', find_url='https://github.com/open-mmlab/mmcv.git')

    # install extension based on OpenMMLab
    install('mmcls-project', find_url='https://github.com/xxx/mmcls-project.git')
    ```

</details>

<details>
<summary>2. uninstall</summary>

[![asciicast](https://asciinema.org/a/416948.svg)](https://asciinema.org/a/416948)

- command

    ```bash
    # uninstall mmcv
    > mim uninstall mmcv-full

    # uninstall mmcls
    > mim uninstall mmcls
    ```

- api

    ```python
    from mim import uninstall

    # uninstall mmcv
    uninstall('mmcv-full')

    # uninstall mmcls
    uninstall('mmcls)
    ```

</details>

<details>
<summary>3. list</summary>

[![asciicast](https://asciinema.org/a/416949.svg)](https://asciinema.org/a/416949)

- command

    ```bash
    > mim list
    > mim list --all
    ```

- api

    ```python
    from mim import list_package

    list_package()
    list_package(True)
    ```

</details>

<details>
<summary>4. search</summary>

[![asciicast](https://asciinema.org/a/416950.svg)](https://asciinema.org/a/416950)

- command

    ```bash
    > mim search mmcls
    > mim search mmcls==0.11.0 --remote
    > mim search mmcls --config resnet18_b16x8_cifar10
    > mim search mmcls --model resnet
    > mim search mmcls --dataset cifar-10
    > mim search mmcls --valid-field
    > mim search mmcls --condition 'batch_size>45,epochs>100'
    > mim search mmcls --condition 'batch_size>45 epochs>100'
    > mim search mmcls --condition '128<batch_size<=256'
    > mim search mmcls --sort batch_size epochs
    > mim search mmcls --field epochs batch_size weight
    > mim search mmcls --exclude-field weight paper
    ```

- api

    ```python
    from mim import get_model_info

    get_model_info('mmcls')
    get_model_info('mmcls==0.11.0', local=False)
    get_model_info('mmcls', models=['resnet'])
    get_model_info('mmcls', training_datasets=['cifar-10'])
    get_model_info('mmcls', filter_conditions='batch_size>45,epochs>100')
    get_model_info('mmcls', filter_conditions='batch_size>45 epochs>100')
    get_model_info('mmcls', filter_conditions='128<batch_size<=256')
    get_model_info('mmcls', sorted_fields=['batch_size', 'epochs'])
    get_model_info('mmcls', shown_fields=['epochs', 'batch_size', 'weight'])
    ```

</details>

<details>
<summary>5. download</summary>

[![asciicast](https://asciinema.org/a/416951.svg)](https://asciinema.org/a/416951)

- command

    ```bash
    > mim download mmcls --config resnet18_b16x8_cifar10
    > mim download mmcls --config resnet18_b16x8_cifar10 --dest .
    ```

- api

    ```python
    from mim import download

    download('mmcls', ['resnet18_b16x8_cifar10'])
    download('mmcls', ['resnet18_b16x8_cifar10'], dest_dir='.')
    ```

</details>

<details>
<summary>6. train</summary>

[![asciicast](https://asciinema.org/a/416953.svg)](https://asciinema.org/a/416953)

- command

    ```bash
    # Train models on a single server with CPU by setting `gpus` to 0 and
    # 'launcher' to 'none' (if applicable). The training script of the
    # corresponding codebase will fail if it doesn't support CPU training.
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 0
    # Train models on a single server with one GPU
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 1
    # Train models on a single server with 4 GPUs and pytorch distributed
    > mim train mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 4 \
        --launcher pytorch
    # Train models on a slurm HPC with one 8-GPU node
    > mim train mmcls resnet101_b16x8_cifar10.py --launcher slurm --gpus 8 \
        --gpus-per-node 8 --partition partition_name --work-dir tmp
    # Print help messages of sub-command train
    > mim train -h
    # Print help messages of sub-command train and the training script of mmcls
    > mim train mmcls -h
    ```

- api

    ```python
    from mim import train

    train(repo='mmcls', config='resnet18_b16x8_cifar10.py', gpus=0,
          other_args='--work-dir tmp')
    train(repo='mmcls', config='resnet18_b16x8_cifar10.py', gpus=1,
          other_args='--work-dir tmp')
    train(repo='mmcls', config='resnet18_b16x8_cifar10.py', gpus=4,
          launcher='pytorch', other_args='--work-dir tmp')
    train(repo='mmcls', config='resnet18_b16x8_cifar10.py', gpus=8,
          launcher='slurm', gpus_per_node=8, partition='partition_name',
          other_args='--work-dir tmp')
    ```

</details>

<details>
<summary>7. test</summary>

[![asciicast](https://asciinema.org/a/416955.svg)](https://asciinema.org/a/416955)

- command

    ```bash
    # Test models on a single server with 1 GPU, report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 1 --metrics accuracy
    # Test models on a single server with 1 GPU, save predictions
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 1 --out tmp.pkl
    # Test models on a single server with 4 GPUs, pytorch distributed,
    # report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 4 --launcher pytorch --metrics accuracy
    # Test models on a slurm HPC with one 8-GPU node, report accuracy
    > mim test mmcls resnet101_b16x8_cifar10.py --checkpoint \
        tmp/epoch_3.pth --gpus 8 --metrics accuracy --partition \
        partition_name --gpus-per-node 8 --launcher slurm
    # Print help messages of sub-command test
    > mim test -h
    # Print help messages of sub-command test and the testing script of mmcls
    > mim test mmcls -h
    ```

- api

    ```python
    from mim import test
    test(repo='mmcls', config='resnet101_b16x8_cifar10.py',
         checkpoint='tmp/epoch_3.pth', gpus=1, other_args='--metrics accuracy')
    test(repo='mmcls', config='resnet101_b16x8_cifar10.py',
         checkpoint='tmp/epoch_3.pth', gpus=1, other_args='--out tmp.pkl')
    test(repo='mmcls', config='resnet101_b16x8_cifar10.py',
         checkpoint='tmp/epoch_3.pth', gpus=4, launcher='pytorch',
         other_args='--metrics accuracy')
    test(repo='mmcls', config='resnet101_b16x8_cifar10.py',
         checkpoint='tmp/epoch_3.pth', gpus=8, partition='partition_name',
         launcher='slurm', gpus_per_node=8, other_args='--metrics accuracy')
    ```

</details>

<details>
<summary>8. run</summary>

[![asciicast](https://asciinema.org/a/416956.svg)](https://asciinema.org/a/416956)

- command

    ```bash
    # Get the Flops of a model
    > mim run mmcls get_flops resnet101_b16x8_cifar10.py
    # Publish a model
    > mim run mmcls publish_model input.pth output.pth
    # Train models on a slurm HPC with one GPU
    > srun -p partition --gres=gpu:1 mim run mmcls train \
        resnet101_b16x8_cifar10.py --work-dir tmp
    # Test models on a slurm HPC with one GPU, report accuracy
    > srun -p partition --gres=gpu:1 mim run mmcls test \
        resnet101_b16x8_cifar10.py tmp/epoch_3.pth --metrics accuracy
    # Print help messages of sub-command run
    > mim run -h
    # Print help messages of sub-command run, list all available scripts in
    # codebase mmcls
    > mim run mmcls -h
    # Print help messages of sub-command run, print the help message of
    # training script in mmcls
    > mim run mmcls train -h
    ```

- api

    ``` python
    from mim import run

    run(repo='mmcls', command='get_flops',
        other_args='resnet101_b16x8_cifar10.py')
    run(repo='mmcls', command='publish_model',
        other_args='input.pth output.pth')
    run(repo='mmcls', command='train',
        other_args='resnet101_b16x8_cifar10.py --work-dir tmp')
    run(repo='mmcls', command='test',
        other_args='resnet101_b16x8_cifar10.py tmp/epoch_3.pth --metrics accuracy')
    ```

</details>

<details>
<summary>9. gridsearch</summary>

[![asciicast](https://asciinema.org/a/416958.svg)](https://asciinema.org/a/416958)

- command

    ```bash
    # Parameter search on a single server with CPU by setting `gpus` to 0 and
    # 'launcher' to 'none' (if applicable). The training script of the
    # corresponding codebase will fail if it doesn't support CPU training.
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 0 \
        --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search learning
    # rate
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 1 \
        --search-args '--optimizer.lr 1e-2 1e-3'
    # Parameter search with on a single server with one GPU, search
    # weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 1 \
        --search-args '--optimizer.weight_decay 1e-3 1e-4'
    # Parameter search with on a single server with one GPU, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 1 \
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 \
        1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 8 \
        --partition partition_name --gpus-per-node 8 --launcher slurm \
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 \
        1e-4'
    # Parameter search on a slurm HPC with one 8-GPU node, search learning
    # rate and weight_decay, max parallel jobs is 2
    > mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 8 \
        --partition partition_name --gpus-per-node 8 --launcher slurm \
        --max-jobs 2 --search-args '--optimizer.lr 1e-2 1e-3 \
        --optimizer.weight_decay 1e-3 1e-4'
    # Print the help message of sub-command search
    > mim gridsearch -h
    # Print the help message of sub-command search and the help message of the
    # training script of codebase mmcls
    > mim gridsearch mmcls -h
    ```

- api

    ```python
    from mim import gridsearch

    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=0,
               search_args='--optimizer.lr 1e-2 1e-3',
               other_args='--work-dir tmp')
    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=1,
               search_args='--optimizer.lr 1e-2 1e-3',
               other_args='--work-dir tmp')
    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=1,
               search_args='--optimizer.weight_decay 1e-3 1e-4',
               other_args='--work-dir tmp')
    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=1,
               search_args='--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay'
                           '1e-3 1e-4',
               other_args='--work-dir tmp')
    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=8,
               partition='partition_name', gpus_per_node=8, launcher='slurm',
               search_args='--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay'
                           ' 1e-3 1e-4',
               other_args='--work-dir tmp')
    gridsearch(repo='mmcls', config='resnet101_b16x8_cifar10.py', gpus=8,
               partition='partition_name', gpus_per_node=8, launcher='slurm',
               max_workers=2,
               search_args='--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay'
                           ' 1e-3 1e-4',
               other_args='--work-dir tmp')
    ```

</details>


## Contributing

We appreciate all contributions to improve mim. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A comprehensive toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
