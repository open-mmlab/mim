## Installation

### Prepare Environment

1. Create a conda virtual environment and activate it.

    ```bash
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```bash
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

### Install MIM

+ from pypi

    ```bash
    pip install openmim
    ```

+ from source

    ```bash
    git clone https://github.com/open-mmlab/mim.git
    cd mim
    pip install -e .
    # python setup.py develop or python setup.py install
    ```

### Optional Features

1. Auto completion

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

    Alternatively, you can activate the script. Please refer to [activation-script](https://click.palletsprojects.com/en/7.x/bashcomplete/#activation-script).
