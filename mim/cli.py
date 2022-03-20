# Copyright (c) OpenMMLab. All rights reserved.
import difflib
import os
import os.path as osp
from configparser import ConfigParser

import click

plugin_folder = os.path.join(os.path.dirname(__file__), 'commands')
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

DEFAULT_CFG = osp.join(osp.expanduser('~'), '.mimrc')


def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    ctx.default_map = {}
    for sect in cfg.sections():
        command_path = sect.split('.')
        if command_path[0] != 'options':
            continue
        defaults = ctx.default_map
        for cmdname in command_path[1:]:
            defaults = defaults.setdefault(cmdname, {})
        defaults.update(cfg[sect])


class MIM(click.MultiCommand):

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(plugin_folder):
            if not filename.startswith('__') and filename.endswith('.py'):
                rv.append(filename[:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        ns = {}
        fn = osp.join(plugin_folder, name + '.py')
        if not osp.exists(fn):
            matches = [
                x for x in self.list_commands(ctx) if x.startswith(name)
            ]
            if not matches:
                return None
            elif len(matches) == 1:
                return self.get_command(ctx, matches[0])

        with open(fn) as f:
            code = compile(f.read(), fn, 'exec')
            eval(code, ns, ns)
        return ns['cli']

    def resolve_command(self, ctx, args):
        # The implementation is modified from https://github.com/click-contrib/
        # click-didyoumean/blob/master/click_didyoumean/__init__.py#L25
        try:
            return super().resolve_command(ctx, args)
        except click.exceptions.UsageError as error:
            error_msg = str(error)
            original_cmd_name = click.utils.make_str(args[0])
            matches = difflib.get_close_matches(original_cmd_name,
                                                self.list_commands(ctx), 3,
                                                0.1)
            if matches:
                error_msg += '\n\nDid you mean one of these?\n'
                error_msg += '\n'.join(matches)

            raise click.exceptions.UsageError(error_msg, error.ctx)


@click.command(cls=MIM, context_settings=CONTEXT_SETTINGS)
@click.option(
    '--user-conf',
    type=click.Path(dir_okay=False),
    default=DEFAULT_CFG,
    callback=configure,
    is_eager=True,
    expose_value=False,
    help='Read option defaults from the .mimrc file',
    show_default=True)
@click.version_option()
def cli():
    """OpenMMLab Command Line Interface.

    MIM provides a unified API for launching and installing OpenMMLab projects
    and their extensions, and managing the OpenMMLab model zoo.
    """
    pass
