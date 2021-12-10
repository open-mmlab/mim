## Customization

You can customize MIM using the `~/.mimrc` file, which should be placed in your home directory.

### Customize Default Values of MIM commands

You can customize the default values of MIM commands with `~/.mimrc`:

```ini
[option.train]
gpus = 8
gpus_per_node = 8
cpus_per_task = 4
launcher = 'slurm'

[option.install]
is_yes = True
```

This config file set the default values of options `gpus`, `gpus_per_node`, `cpus_per_task`, `launcher` of sub-command `train` to 8, 8, 4, 'slurm', set the default value of option `is_yes` of sub-command `install` to True.
