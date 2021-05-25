import os.path as osp
import pickle
import re
import subprocess
import tempfile
from pkg_resources import resource_filename
from typing import Any, List, Optional

import click
import pandas as pd
from modelindex.load_model_index import load
from modelindex.models.ModelIndex import ModelIndex
from pandas import DataFrame

from mim.click import OptionEatAll, get_downstream_package, param2lowercase
from mim.utils import (
    DEFAULT_CACHE_DIR,
    PKG2PROJECT,
    cast2lowercase,
    echo_success,
    get_github_url,
    get_installed_version,
    highlighted_error,
    is_installed,
    split_package_version,
)

abbrieviation = {
    'batch_size': 'bs',
    'epochs': 'epoch',
    'inference_time': 'fps',
    'inference_time_(fps)': 'fps',
}


@click.command('search')
@click.argument(
    'packages',
    nargs=-1,
    type=click.STRING,
    required=True,
    autocompletion=get_downstream_package,
    callback=param2lowercase)
@click.option(
    '--config', 'configs', cls=OptionEatAll, help='Selected configs.')
@click.option('--valid-config', is_flag=True, help='List all valid config id.')
@click.option('--model', 'models', cls=OptionEatAll, help='Selected models.')
@click.option(
    '--dataset',
    'training_datasets',
    cls=OptionEatAll,
    help='Selected training datasets.')
@click.option(
    '--condition',
    'filter_conditions',
    type=str,
    help='Conditions of searching models.')
@click.option('--sort', 'sorted_fields', cls=OptionEatAll, help='Sort output.')
@click.option(
    '--ascending/--descending',
    is_flag=True,
    help='Sorting with ascending or descending.')
@click.option(
    '--field', 'shown_fields', cls=OptionEatAll, help='Fields to be shown.')
@click.option(
    '--exclude-field',
    'unshown_fields',
    cls=OptionEatAll,
    help='Fields to be hidden.')
@click.option('--valid-field', is_flag=True, help='List all valid field.')
@click.option(
    '--json', 'json_path', type=str, help='Dump output to json_path.')
@click.option('--to-dict', 'to_dict', is_flag=True, help='Return metadata.')
@click.option(
    '--local/--remote', default=True, help='Show local or remote packages.')
def cli(packages: List[str],
        configs: Optional[List[str]] = None,
        valid_config: bool = True,
        models: Optional[List[str]] = None,
        training_datasets: Optional[List[str]] = None,
        filter_conditions: Optional[str] = None,
        sorted_fields: Optional[List[str]] = None,
        ascending: bool = True,
        shown_fields: Optional[List[str]] = None,
        unshown_fields: Optional[List[str]] = None,
        valid_field: bool = True,
        json_path: Optional[str] = None,
        to_dict: bool = False,
        local: bool = True) -> Any:
    """Show the information of packages.

    \b
    Example:
        > mim search mmcls
        > mim search mmcls==0.11.0 --remote
        > mim search mmcls --valid-config
        > mim search mmcls --config resnet18_b16x8_cifar10
        > mim search mmcls --model resnet
        > mim search mmcls --dataset cifar-10
        > mim search mmcls --valid-filed
        > mim search mmcls --condition 'bs>45,epoch>100'
        > mim search mmcls --condition 'bs>45 epoch>100'
        > mim search mmcls --condition '128<bs<=256'
        > mim search mmcls --sort bs epoch
        > mim search mmcls --field epoch bs weight
        > mim search mmcls --exclude-field weight paper
    """
    packages_info = {}
    for package in packages:
        dataframe = get_model_info(
            package=package,
            configs=configs,
            models=models,
            training_datasets=training_datasets,
            filter_conditions=filter_conditions,
            sorted_fields=sorted_fields,
            ascending=ascending,
            shown_fields=shown_fields,
            unshown_fields=unshown_fields,
            local=local)

        if to_dict or json_path:
            packages_info.update(dataframe.to_dict('index'))  # type: ignore
        elif valid_config:
            echo_success('\nvalid config ids:')
            click.echo(dataframe.index.to_list())
        elif valid_field:
            echo_success('\nvalid fields:')
            click.echo(dataframe.columns.to_list())
        elif not dataframe.empty:
            print_df(dataframe)
        else:
            click.echo('can not find matching models.')

    if json_path:
        dump2json(dataframe, json_path)

    if to_dict:
        return packages_info


def get_model_info(package: str,
                   configs: Optional[List[str]] = None,
                   models: Optional[List[str]] = None,
                   training_datasets: Optional[List[str]] = None,
                   filter_conditions: Optional[str] = None,
                   sorted_fields: Optional[List[str]] = None,
                   ascending: bool = True,
                   shown_fields: Optional[List[str]] = None,
                   unshown_fields: Optional[List[str]] = None,
                   local: bool = True,
                   to_dict: bool = False) -> Any:
    """Get model information like metric or dataset.

    Args:
        package (str): Name of package to load metadata.
        configs (List[str], optional): Config ids to query. Default: None.
        models (List[str], optional): Models to query. Default: None.
        training_datasets (List[str], optional): Training datasets to query.
            Default: None.
        filter_conditions (str, optional): Conditions to filter. Default: None.
        sorted_fields (List[str], optional): Sort output by sorted_fields.
            Default: None.
        ascending (bool): Sort by ascending or descending. Default: True.
        shown_fields (List[str], optional): Fields to be outputted.
            Default: None.
        unshown_fields (List[str], optional): Fields to be hidden.
            Default: None.
        local (bool): Query from local environment or remote github.
            Default: True.
        to_dict (bool): Convert dataframe into dict. Default: False.
    """
    metadata = load_metadata(package, local)
    dataframe = convert2df(metadata)
    dataframe = filter_by_configs(dataframe, configs)
    dataframe = filter_by_conditions(dataframe, filter_conditions)
    dataframe = filter_by_models(dataframe, models)
    dataframe = filter_by_training_datasets(dataframe, training_datasets)
    dataframe = sort_by(dataframe, sorted_fields, ascending)
    dataframe = select_by(dataframe, shown_fields, unshown_fields)

    if to_dict:
        return dataframe.to_dict('index')
    else:
        return dataframe


def load_metadata(package: str, local: bool = True) -> Optional[ModelIndex]:
    """Load metadata from local package or remote package.

    Args:
        package (str): Name of package to load metadata.
        local (bool): Query from local environment or remote github.
            Default: True.
    """
    if '=' in package and local:
        raise ValueError(
            highlighted_error(
                'if package is set like "mmcls==0.11.0", the local '
                'flag should be False.'))

    if local:
        return load_metadata_from_local(package)
    else:
        return load_metadata_from_remote(package)


def load_metadata_from_local(package: str):
    """Load metadata from local package.

    Args:
        package (str): Name of package to load metadata.

    Example:
        >>> metadata = load_metadata_from_local('mmcls')
    """
    if is_installed(package):
        version = get_installed_version(package)
        click.echo(f'local verison: {version}')

        metadata_path = resource_filename(package, 'model_zoo.yml')
        metadata = load(metadata_path)

        return metadata
    else:
        raise ImportError(
            highlighted_error(
                f'{package} is not installed. Install {package} by "mim '
                f'install {package}" or use mim search {package} --remote'))


def load_metadata_from_remote(package: str) -> Optional[ModelIndex]:
    """Load metadata from github.

    Download the model_zoo directory from github and parse it into metadata.

    Args:
        package (str): Name of package to load metadata.

    Example:
        >>> # load metadata from master branch
        >>> metadata = load_metadata_from_remote('mmcls')
        >>> # load metadata from 0.11.0
        >>> metadata = load_metadata_from_remote('mmcls==0.11.0')
    """
    package, version = split_package_version(package)

    github_url = get_github_url(package)

    pkl_path = osp.join(DEFAULT_CACHE_DIR, f'{package}-{version}.pkl')
    if osp.exists(pkl_path):
        with open(pkl_path, 'rb') as fr:
            metadata = pickle.load(fr)
    else:
        clone_cmd = ['git', 'clone', github_url]
        if version:
            clone_cmd.extend(['-b', f'v{version}'])
        with tempfile.TemporaryDirectory() as temp:
            repo_root = osp.join(temp, PKG2PROJECT[package])
            clone_cmd.append(repo_root)
            subprocess.check_call(
                clone_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            metadata_path = osp.join(repo_root, 'model_zoo.yml')
            if not osp.exists(metadata_path):
                raise FileNotFoundError(
                    highlighted_error(
                        'current version can not support "mim search '
                        f'{package}", please upgrade your {package}.'))

            metadata = load(metadata_path)

        if not version:
            with open(pkl_path, 'wb') as fw:
                pickle.dump(metadata, fw)

    return metadata


def convert2df(metadata: ModelIndex) -> DataFrame:
    """Convert metadata into DataFrame format."""
    name2model = {}
    name2collection = {}
    for collection in metadata.collections:
        collection_info = {}
        data = getattr(collection.metadata, 'data', None)
        if data:
            for key, value in data.items():
                name = '_'.join(key.split())
                name = cast2lowercase(name)
                name = abbrieviation.get(name, name)
                if isinstance(value, str):
                    collection_info[name] = cast2lowercase(value)
                elif isinstance(value, (list, tuple)):
                    collection_info[name] = ','.join(cast2lowercase(value))
                else:
                    collection_info[name] = value

        paper = getattr(collection, 'paper', None)
        if paper:
            if isinstance(paper, str):
                collection_info['paper'] = paper
            else:
                collection_info['paper'] = ','.join(paper)

        readme = getattr(collection, 'readme', None)
        if readme:
            collection_info['readme'] = readme

        name2collection[collection.name] = collection_info

    for model in metadata.models:
        model_info = {}
        data = getattr(model.metadata, 'data', None)
        if data:
            for key, value in model.metadata.data.items():
                name = '_'.join(key.split())
                name = cast2lowercase(name)
                name = abbrieviation.get(name, name)
                if isinstance(value, str):
                    model_info[name] = cast2lowercase(value)
                elif isinstance(value, (list, tuple)):
                    model_info[name] = ','.join(cast2lowercase(value))
                else:
                    model_info[name] = value

        results = getattr(model, 'results', None)
        for result in results:
            dataset = cast2lowercase(result.dataset)
            metrics = getattr(result, 'metrics', None)
            if metrics is None:
                continue

            for key, value in metrics.items():
                name = '_'.join(key.split())
                name = cast2lowercase(name)
                name = abbrieviation.get(name, name)
                model_info[f'{dataset}/{name}'] = value

        paper = getattr(model, 'paper', None)
        if paper:
            if isinstance(paper, str):
                model_info['paper'] = paper
            else:
                model_info['paper'] = ','.join(paper)

        weight = getattr(model, 'weights', None)
        if weight:
            if isinstance(weight, str):
                model_info['weight'] = weight
            else:
                model_info['weight'] = ','.join(weight)

        config = getattr(model, 'config', None)
        if config:
            if isinstance(config, str):
                model_info['config'] = config
            else:
                model_info['config'] = ','.join(config)

        collection_name = getattr(model, 'in_collection', None)
        if collection_name:
            model_info['model'] = cast2lowercase(collection_name)
            for key, value in name2collection[collection_name].items():
                model_info.setdefault(key, value)

        name2model[model.name] = model_info

    df = DataFrame(name2model)
    df = df.T

    return df


def filter_by_configs(dataframe: DataFrame,
                      configs: Optional[List[str]] = None) -> DataFrame:
    """Filter by configs.

    Args:
        dataframe (DataFrame): Data to be filtered.
        configs (List[str], optional): Config ids to query. Default: None.
    """
    if configs is None:
        return dataframe

    configs = cast2lowercase(configs)
    valid_configs = set(dataframe.index)
    invalid_configs = set(configs) - valid_configs  # type: ignore

    if invalid_configs:
        raise ValueError(
            highlighted_error(
                f'Expected configs: {valid_configs}, but got {invalid_configs}'
            ))

    return dataframe.filter(items=configs, axis=0)


def filter_by_models(
        dataframe: DataFrame,  # type: ignore
        models: Optional[List[str]] = None) -> DataFrame:
    """Filter by models.

    Args:
        dataframe (DataFrame): Data to be filtered.
        models (List[str], optional): Models to query. Default: None.
    """
    if models is None:
        return dataframe

    if 'model' not in dataframe.columns:
        raise ValueError(
            highlighted_error(f'models is not in {dataframe.columns}.'))

    models = cast2lowercase(models)

    valid_models = set(dataframe['model'])
    invalid_models = set(models) - valid_models  # type: ignore

    if invalid_models:
        raise ValueError(
            highlighted_error(
                f'Expected models: {valid_models}, but got {invalid_models}'))

    selected_rows = False
    for model in models:  # type: ignore
        selected_rows |= (dataframe['model'] == model)

    return dataframe[selected_rows]


def filter_by_conditions(
        dataframe: DataFrame,
        filter_conditions: Optional[str] = None) -> DataFrame:  # TODO
    """Filter rows with conditions.

    Args:
        dataframe (DataFrame): Data to be filtered.
        filter_conditions (str, optional): Conditions to filter. Default: None.
    """
    if filter_conditions is None:
        return dataframe

    filter_conditions = cast2lowercase(filter_conditions)

    and_conditions = []
    or_conditions = []

    # 'fps>45,epoch>100' or 'fps>45 epoch>100' -> ['fps>40', 'epoch>100']
    filter_conditions = re.split(r'[ ,]+', filter_conditions)  # type: ignore

    valid_fields = dataframe.columns
    for condition in filter_conditions:  # type: ignore
        search_group = re.search(r'[a-z]+[-@_]*[.\w]*', condition)  # TODO
        if search_group is None:
            raise ValueError(
                highlighted_error(f'Invalid condition: {condition}'))

        field = search_group[0]  # type: ignore

        contain_index = valid_fields.str.contains(field)
        if contain_index.any():
            contain_fields = valid_fields[contain_index]
            for _field in contain_fields:
                new_condition = condition.replace(field, f'`{_field}`')
                if '/' in _field:
                    or_conditions.append(new_condition)
                else:
                    and_conditions.append(condition)
        else:
            raise ValueError(highlighted_error(f'Invalid field: {field}'))

    if and_conditions:
        expr = ' & '.join(and_conditions)
        dataframe = dataframe.query(expr)

    if or_conditions:
        expr = ' | '.join(or_conditions)
        dataframe = dataframe.query(expr)

    return dataframe


def filter_by_training_datasets(dataframe: DataFrame,
                                datasets: Optional[List[str]]) -> DataFrame:
    """Filter by training datasets.

    Args:
        dataframe (DataFrame): Data to be filtered.
        datasets (List[str], optional): Training datasets to query.
            Default: None.
    """
    if datasets is None:
        return dataframe

    if 'training_data' not in dataframe.columns:
        raise ValueError(
            highlighted_error(
                f'training_datasets is not in {dataframe.columns}.'))

    datasets = cast2lowercase(datasets)

    valid_datasets = set(dataframe['training_data'])
    invalid_datasets = set(datasets) - valid_datasets  # type: ignore

    if invalid_datasets:
        raise ValueError(
            highlighted_error(f'Expected datasets: {valid_datasets}, but got '
                              f'{invalid_datasets}'))

    selected_rows = False
    for ds in datasets:  # type: ignore
        selected_rows |= (dataframe['training_data'] == ds)

    return dataframe[selected_rows]


def sort_by(dataframe: DataFrame,
            sorted_fields: Optional[List[str]],
            ascending: bool = True) -> DataFrame:
    """Sort by the fields.

    Args:
        dataframe (DataFrame): Data to be sorted.
        sorted_fields (List[str], optional): Sort output by sorted_fields.
            Default: None.
        ascending (bool): Sort by ascending or descending. Default: True.
    """
    if sorted_fields is None:
        return dataframe

    sorted_fields = cast2lowercase(sorted_fields)

    valid_fields = set(dataframe.columns)
    invalid_fields = set(sorted_fields) - valid_fields  # type: ignore
    if invalid_fields:
        raise ValueError(
            highlighted_error(
                f'Expected fields: {valid_fields}, but got {invalid_fields}'))

    sorted_fields = list(sorted_fields)  # type: ignore
    return dataframe.sort_values(by=sorted_fields, ascending=ascending)


def select_by(dataframe: DataFrame,
              shown_fields: Optional[List[str]] = None,
              unshown_fields: Optional[List[str]] = None) -> DataFrame:
    """Select by the fields.

    Args:
        dataframe (DataFrame): Data to be filtered.
        shown_fields (List[str], optional): Fields to be outputted.
            Default: None.
        unshown_fields (List[str], optional): Fields to be hidden.
            Default: None.
    """
    if shown_fields is None and unshown_fields is None:
        return dataframe

    if shown_fields and unshown_fields:
        raise ValueError(
            highlighted_error(
                'shown_fields and unshown_fields must be mutually exclusive.'))

    valid_fields = set(dataframe.columns)
    if shown_fields:
        shown_fields = cast2lowercase(shown_fields)
        invalid_fields = set(shown_fields) - valid_fields  # type: ignore
        if invalid_fields:
            raise ValueError(
                highlighted_error(f'Expected fields: {valid_fields}, but got '
                                  f'{invalid_fields}'))

        dataframe = dataframe.filter(items=shown_fields)

    else:
        unshown_fields = cast2lowercase(unshown_fields)  # type: ignore
        invalid_fields = set(unshown_fields) - valid_fields  # type: ignore
        if invalid_fields:
            raise ValueError(
                highlighted_error(f'Expected fields: {valid_fields}, but got '
                                  f'{invalid_fields}'))

        dataframe = dataframe.drop(
            columns=list(unshown_fields))  # type: ignore

    dataframe = dataframe.dropna(axis=0, how='all')

    return dataframe


def dump2json(dataframe: DataFrame, json_path: str) -> None:
    """Dump data frame of meta data into JSON.

    Args:
        dataframe (DataFrame): Data to be filtered.
        json_path (str): Dump output to json_path.
    """
    dataframe.to_json(json_path)


def print_df(dataframe: DataFrame) -> None:
    """Print Dataframe into terminal."""

    def _generate_output():
        for row in dataframe.iterrows():
            config_msg = click.style(f'config id: {row[0]}\n', fg='green')
            yield from [
                config_msg, '-' * pd.get_option('display.width'),
                f'\n{row[1].dropna().to_string()}\n'
            ]

    click.echo_via_pager(_generate_output())
