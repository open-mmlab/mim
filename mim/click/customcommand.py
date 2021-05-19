import os
import os.path as osp
import subprocess
import typing as t
from gettext import ngettext

import click
from click.core import Context, iter_params_for_processing
from click.formatting import HelpFormatter

from mim.click import get_official_package
from mim.utils import (
    color_echo,
    exit_with_error,
    get_installed_path,
    is_installed,
    recursively_find,
)


class CustomCommand(click.Command):

    def parse_args(self, ctx: Context, args: t.List[str]) -> t.List[str]:
        # remove -h and --help
        self.raw_args = [x for x in args if x not in ['-h', '--help']]
        if (not args and getattr(self, 'no_args_is_help', False)
                and not ctx.resilient_parsing):
            click.echo(ctx.get_help(), color=ctx.color)
            ctx.exit()

        parser = self.make_parser(ctx)
        opts, args, param_order = parser.parse_args(args=args)

        for param in iter_params_for_processing(param_order,
                                                self.get_params(ctx)):
            value, args = param.handle_parse_result(ctx, opts, args)

        if args and not ctx.allow_extra_args and not ctx.resilient_parsing:
            ctx.fail(
                ngettext(
                    'Got unexpected extra argument ({args})',
                    'Got unexpected extra arguments ({args})',
                    len(args),
                ).format(args=' '.join(map(str, args))))

        ctx.args = args
        return args

    def format_help(self, ctx: Context, formatter: HelpFormatter) -> None:
        formatter = ctx.make_formatter()
        self.original_format_help(ctx, formatter)
        click.echo(formatter.getvalue())

        repos = get_official_package()
        if self.name in ['train', 'test', 'search']:
            repo = None if not len(self.raw_args) else self.raw_args[0]
            if repo and repo in repos and is_installed(repo):
                self.name = 'train' if self.name == 'search' else self.name
                script = osp.join(
                    get_installed_path(repo), f'tools/{self.name}.py')
                ret = subprocess.check_output(
                    ['python', '-u', script, '--help'])
                color_echo(
                    'The help message of corresponding script is: ',
                    color='blue')
                color_echo(ret.decode('utf-8'), color='blue')

        if self.name == 'run':
            repo = None if not len(self.raw_args) else self.raw_args[0]
            command = None if len(self.raw_args) <= 1 else self.raw_args[1]

            if not repo:
                return

            if repo not in repos:
                exit_with_error(f'{repo} is not an OpenMMLAB codebase. ')

            if repo in repos and not is_installed(repo):
                exit_with_error(f'Codebase {repo} in not installed. ')

            if command:
                repo_root = get_installed_path(repo)
                files = recursively_find(
                    osp.join(repo_root, 'tools'), command + '.py')
                if len(files) == 0:
                    exit_with_error(
                        f"The command {command} doesn't exist in codebase "
                        f'{repo}.')
                elif len(files) > 1:
                    exit_with_error(
                        f'Multiple scripts with name {command}.py are in '
                        f'codebase {repo}.')
                ret = subprocess.check_output(
                    ['python', '-u', files[0], '--help'])
                click.echo('=' * 80)
                click.echo('The help message of corresponding script is: ')
                click.echo(ret.decode('utf-8'))
            else:
                repo_root = get_installed_path(repo)
                tool_root = osp.join(repo_root, 'tools')
                walk_list = list(os.walk(tool_root))

                files = []
                for item in walk_list:
                    files.extend([osp.join(item[0], x) for x in item[2]])

                files = [x for x in files if x.endswith('.py')]
                files = [x.split(tool_root + '/')[1] for x in files]

                if len(files):
                    click.echo('=' * 80)
                    click.echo(f'Available scripts in {repo} are:')

                    def name(f):
                        return f[:-3].replace('/', ':')

                    for f in files:
                        click.echo(f'{name(f)}: {f}')

    def original_format_help(self, ctx: Context,
                             formatter: HelpFormatter) -> None:
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)
