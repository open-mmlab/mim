# MIM: MIM Installs OpenMMLab Packages

MIM provides a unified API for launching and installing OpenMMLab projects and their extensions, and managing the OpenMMLab model zoo.

## Installation

1. Create a conda virtual environment and activate it.

    ```bash
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```bash
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

3. Install MIM

    + from pypi

        ```bash
        python -m pip install mim
        ```

    + from source

        ```bash
        git clone https://github.com/open-mmlab/mim.git
        cd mim
        pip install -e .
        # python setup.py develop or python setup.py install
        ```

4. Auto completion (Optional)

    In order to activate shell completion, you need to inform your shell that completion is available for your script.

    + For Bash, add this to ~/.bashrc:

        ```bash
        eval "$(_MIM_COMPLETE=source mim)"
        ```

    + For Zsh, add this to ~/.zshrc:

        ```bash
        eval "$(_MIM_COMPLETE=source_zsh mim)"
        ```

    + For Fish, add this to ~/.config/fish/completions/mim.fish:

        ```bash
        eval (env _MIM_COMPLETE=source_fish mim)
        ```

    Open a new shell to enable completion. Or run the eval command directly in your current shell to enable it temporarily.

    The above eval command will invoke your application every time a shell is started. This may slow down shell startup time significantly.

    Alternatively, you can activate the script. Please refer to [activation-script](https://click.palletsprojects.com/en/7.x/bashcomplete/#activation-script)

## Command

<details>
<summary>1. install</summary>

[![asciicast](https://asciinema.org/a/5fd2bzqlxz9g0a2H3tV3avJO6.svg)](https://asciinema.org/a/5fd2bzqlxz9g0a2H3tV3avJO6)

+ command

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

+ api

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

[![asciicast](https://asciinema.org/a/7Wps2UeQ0MeqhNAFRIRpry4k1.svg)](https://asciinema.org/a/7Wps2UeQ0MeqhNAFRIRpry4k1)

+ command

    ```bash
    # uninstall mmcv
    > mim uninstall mmcv-full

    # uninstall mmcls
    > mim uninstall mmcls
    ```

+ api

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

[![asciicast](https://asciinema.org/a/m1EYNM1hrOy8yvjXpS4R62FRm.svg)](https://asciinema.org/a/m1EYNM1hrOy8yvjXpS4R62FRm)

+ command

    ```bash
    > mim list
    > mim list --all
    ```

+ api

    ```python
    from mim import list_package

    list_package()
    list_package(True)
    ```

</details>

<details>
<summary>4. search</summary>

[![asciicast](https://asciinema.org/a/wVYllCMEUOw8PrW68g2IE5fpk.svg)](https://asciinema.org/a/wVYllCMEUOw8PrW68g2IE5fpk)

+ command

    ```bash
    > mim search mmcls
    > mim search mmcls==0.11.0 --remote
    > mim search mmcls --config resnet18_b16x8_cifar10
    > mim search mmcls --model resnet
    > mim search mmcls --dataset cifar-10
    > mim search mmcls --valid-field
    > mim search mmcls --condition 'bs>45,epoch>100'
    > mim search mmcls --condition 'bs>45 epoch>100'
    > mim search mmcls --condition '128<bs<=256'
    > mim search mmcls --sort bs epoch
    > mim search mmcls --field epoch bs weight
    > mim search mmcls --exclude-field weight paper
    ```

+ api

    ```python
    from mim import get_model_info

    get_model_info('mmcls')
    get_model_info('mmcls==0.11.0', local=False)
    get_model_info('mmcls', models=['resnet'])
    get_model_info('mmcls', training_datasets=['cifar-10'])
    get_model_info('mmcls', filter_conditions='bs>45,epoch>100')
    get_model_info('mmcls', filter_conditions='bs>45 epoch>100')
    get_model_info('mmcls', filter_conditions='128<bs<=256')
    get_model_info('mmcls', sorted_fields=['bs', 'epoch'])
    get_model_info('mmcls', shown_fields=['epoch', 'bs', 'weight'])
    ```

</details>

<details>
<summary>5. download</summary>

[![asciicast](https://asciinema.org/a/Srg7AF7y07qx1on7i6Jmym8ay.svg)](https://asciinema.org/a/Srg7AF7y07qx1on7i6Jmym8ay)

+ command

    ```bash
    > mim download mmcls --config resnet18_b16x8_cifar10
    > mim download mmcls --config resnet18_b16x8_cifar10 --dest .
    ```

+ api

    ```python
    from mim import download

    download('mmcls', ['resnet18_b16x8_cifar10'])
    download('mmcls', ['resnet18_b16x8_cifar10'], dest_dir='.')
    ```

</details>

<details>
<summary>6. train</summary>

[![asciicast](https://asciinema.org/a/6PnfF3Vg8ja6RpN3gxxqqA4n8.svg)](https://asciinema.org/a/6PnfF3Vg8ja6RpN3gxxqqA4n8)

+ command

    ```bash
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

+ api

    ```python
    from mim import train

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

[![asciicast](https://asciinema.org/a/0GZWW9b9dfR6L4PNqOlzykSkp.svg)](https://asciinema.org/a/0GZWW9b9dfR6L4PNqOlzykSkp)

+ command

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

+ api

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

[![asciicast](https://asciinema.org/a/6Jh4CABs4F5kEZUGYyxcBZai1.svg)](https://asciinema.org/a/6Jh4CABs4F5kEZUGYyxcBZai1)

+ command

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

+ api

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

[![asciicast](https://asciinema.org/a/2D0rkhvaT0oM9sDvHfspdrgpM.svg)](https://asciinema.org/a/2D0rkhvaT0oM9sDvHfspdrgpM)

+ command

    ```bash
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
        --max-workers 2 --search-args '--optimizer.lr 1e-2 1e-3 \
        --optimizer.weight_decay 1e-3 1e-4'
    # Print the help message of sub-command search
    > mim gridsearch -h
    # Print the help message of sub-command search and the help message of the
    # training script of codebase mmcls
    > mim gridsearch mmcls -h
    ```

+ api

    ```python
    from mim import gridsearch

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


## Build custom projects with mim

We provide an example about how to build custom projects based on MM codebases
and mim. In `examples/custom_backbone`, we define a custom backbone and there
is a classification config file that uses the backbone. To train this model,
you can use the command:

```python
# The working directory is `examples/custom_backbone`
PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls custom_net_config.py --work-dir \
tmp --gpus 1
```


## Contributing

We appreciate all contributions to improve mim. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline.
