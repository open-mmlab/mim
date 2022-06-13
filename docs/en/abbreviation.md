## Abbreviation in MIM

MIM support various kinds of abbreviations, which can be used to shorten the length of commands:

1. Sub-command Name: abbreviation can be used as long as its the prefix of one and only one subcommand, for example:

   1. `g` stands for sub-command `gridsearch`
   2. `tr` stands for sub-command `train`

2. Codebase Name: abbreviation can be used as long as its the substring of one and only one codebase name, for example:

   1. `act` stands for codebase `mmaction`
   2. `cls` stands for codebase `mmcls`

3. Abbreviation for argument / option names: defined in each sub-command, for example, for sub-command `train`:

   1. `-g` stands for `--gpus-per-node`
   2. `-p` stands for `--partition`

### Examples

```shell
# Full Length
mim test mmcls resnet101_b16x8_cifar10.py --checkpoint tmp/epoch_3.pth \
		--gpus 8 --metrics accuracy --partition pname --gpus-per-node 8 \
		--launcher slurm
# w. abbr.
mim te cls resnet101_b16x8_cifar10.py -C tmp/epoch_3.pth -G 8 -g 8 -p pname \
		-l slurm --metrics accuracy

# Full Length
mim gridsearch mmcls resnet101_b16x8_cifar10.py --work-dir tmp --gpus 8 \
        --partition pname --gpus-per-node 8 --launcher slurm --max-jobs 2 \
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 1e-4'
# w. abbr.
mim g cls resnet101_b16x8_cifar10.py --work-dir tmp -G 8 -g 8 -p pname -l slurm -j 2\
        --search-args '--optimizer.lr 1e-2 1e-3 --optimizer.weight_decay 1e-3 1e-4'
```
