# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import tempfile

from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def test_third_party():
    runner = CliRunner()
    # mim install fire
    result = runner.invoke(install, ['fire'])
    assert result.exit_code == 0

    # mim uninstall fire --yes
    result = runner.invoke(uninstall, ['fire', '--yes'])
    assert result.exit_code == 0


def test_mmcv_install():
    runner = CliRunner()
    # mim install mmcv-full --yes
    # install latest version
    result = runner.invoke(install, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    # mim install mmcv-full==1.3.1 --yes
    result = runner.invoke(install, ['mmcv-full==1.3.1', '--yes'])
    assert result.exit_code == 0

    # mim uninstall mmcv-full --yes
    result = runner.invoke(uninstall, ['mmcv-full', '--yes'])
    assert result.exit_code == 0

    # version should be less than latest version
    # mim install mmcv-full==100.0.0 --yes
    result = runner.invoke(install, ['mmcv-full==100.0.0', '--yes'])
    assert result.exit_code == 1


def test_mmrepo_install(tmp_path):
    runner = CliRunner()

    # install local repo
    with tempfile.TemporaryDirectory() as temp_root:
        repo_root = osp.join(temp_root, 'mmclassification')
        subprocess.check_call([
            'git', 'clone',
            'https://github.com/open-mmlab/mmclassification.git', repo_root
        ])

        # mim install .
        current_root = os.getcwd()
        os.chdir(repo_root)
        result = runner.invoke(install, ['.', '--yes'])
        assert result.exit_code == 0

        os.chdir('..')

        # mim install ./mmclassification
        result = runner.invoke(install, ['./mmclassification', '--yes'])
        assert result.exit_code == 0

        # mim install -e ./mmclassification
        result = runner.invoke(install, ['-e', './mmclassification', '--yes'])
        assert result.exit_code == 0

        os.chdir(current_root)

    # mim install mmcls --yes
    result = runner.invoke(install, ['mmcls', '--yes'])
    assert result.exit_code == 0

    # mim install mmcls -f https://github.com/open-mmlab/mmclassification.git
    # install master branch
    result = runner.invoke(install, [
        'mmcls', '--yes', '-f',
        'https://github.com/open-mmlab/mmclassification.git'
    ])

    # mim install mmcls==0.11.0 --yes
    result = runner.invoke(install, ['mmcls==0.11.0', '--yes'])
    assert result.exit_code == 0

    result = runner.invoke(uninstall, ['mmcls', '--yes'])
    assert result.exit_code == 0
