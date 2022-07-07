# Copyright (c) OpenMMLab. All rights reserved.

# flake8: noqa

###### Fix mim command crashes problem if requirements version conflict. #######
# `pkg_resources` checks for a `__requires__` attribute in the `__main__` module
# when initializing the default working set, and resolves its dependency to
# ensure a suitable version of each affected distribution is activated.
#
# The entry point scripts create by `setuptools` use the `__requires__` feature
# for compatibility with `easy_install` but may cause mim crash when version
# conflict exists.
#
# To handle this situation, we set the `__requires__` declared in mim entry point
# script from 'openmim' to '' before importing pkg_resources. This workaround works
# fine so far, but not sure if it would cause other unknown problems or not.
#
# Related Links:
# - https://github.com/open-mmlab/mim/issues/143
# - https://setuptools.pypa.io/en/latest/pkg_resources.html?highlight=__requires__
# - https://github.com/pypa/setuptools/issues/2198

import sys

sys.modules['__main__'].__requires__ = ''  # type: ignore
################################# The end! #####################################

from .commands import (
    download,
    get_model_info,
    gridsearch,
    install,
    list_package,
    run,
    test,
    train,
    uninstall,
)

__all__ = [
    'download', 'install', 'list_package', 'download', 'get_model_info',
    'install', 'uninstall', 'train', 'test', 'run', 'gridsearch'
]
