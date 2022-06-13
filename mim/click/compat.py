# Copyright (c) OpenMMLab. All rights reserved.
from distutils.version import LooseVersion

import click


def is_click_ge_8():
    """Check if the click version is greater than or equal to 8.x."""
    return LooseVersion(click.__version__) >= LooseVersion('8.0.0')


def autocompletion_to_shell_complete(autocompletion):
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
    if is_click_ge_8():
        autocompletion = attrs.pop('autocompletion', None)
        if autocompletion is not None:
            attrs['shell_complete'] = autocompletion_to_shell_complete(
                autocompletion)

    return click.argument(*param_decls, **attrs)
