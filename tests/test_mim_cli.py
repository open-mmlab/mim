# Copyright (c) OpenMMLab. All rights reserved.
import subprocess
import sys


def test_mim_cli():
    status_code = subprocess.check_call(['mim', '--help'])
    assert status_code == 0

    status_code = subprocess.check_call(['mim', '--version'])
    assert status_code == 0

    # test `python -m mim`
    status_code = subprocess.check_call(
        [sys.executable, '-m', 'mim', '--help'])
    assert status_code == 0
