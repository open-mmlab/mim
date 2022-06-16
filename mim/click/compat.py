# Copyright (c) OpenMMLab. All rights reserved.
from distutils.version import LooseVersion

import click


def autocompletion_to_shell_complete(autocompletion: Callable):
    """Convert autocompletion to shell_complete.

    Reference:
    https://github.com/pallets/click/blob/8.0.0/src/click/core.py#L2059

    Args:
        autocompletion: A function that returns custom shell completions.
            Takes ``ctx, param, incomplete`` and must return a list of string.

    Returns:
        A shell_complete function converted from autocompletion.
    """

    def shell_complete(ctx, param, incomplete):
        from click.shell_completion import CompletionItem

        out = []
        for c in autocompletion(ctx, [], incomplete):
            if isinstance(c, tuple):
                c = CompletionItem(c[0], help=c[1])
            elif isinstance(c, str):
                c = CompletionItem(c)

            if c.value.startswith(incomplete):
                out.append(c)
        return out

    return shell_complete


def argument(*param_decls, **attrs):
    """A decorator compatible with click 7.x and 8.x.

    Same as ``click.argument``.
    """
    # 'autocompletion' will be removed in Click 8.1 and its new name is
    # 'shell_complete'.
    if LooseVersion(click.__version__) >= LooseVersion('8.0.0'):
        autocompletion = attrs.pop('autocompletion', None)
        if autocompletion is not None:
            attrs['shell_complete'] = autocompletion_to_shell_complete(
                autocompletion)

    return click.argument(*param_decls, **attrs)
