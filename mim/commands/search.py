import os.path as osp
import pickle
import re
import subprocess
import tempfile
import typing
from typing import Any, List, Optional

import click
from modelindex.load_model_index import load
from modelindex.models.ModelIndex import ModelIndex
from pandas import DataFrame, Series

from mim.click import OptionEatAll, get_downstream_package, param2lowercase
from mim.utils import (
    DEFAULT_CACHE_DIR,
    PKG2PROJECT,
    cast2lowercase,
    echo_success,
    get_github_url,
    get_installed_path,
    highlighted_error,
    is_installed,
    split_package_version,
)


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
@click.option(
    '--display-width', type=int, default=80, help='The display width.')
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
        local: bool = True,
        display_width: int = 80) -> Any:
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
        > mim search mmcls --condition 'batch_size>45,epochs>100'
        > mim search mmcls --condition 'batch_size>45 epochs>100'
        > mim search mmcls --condition '128<batch_size<=256'
        > mim search mmcls --sort batch_size epochs
        > mim search mmcls --field epochs batch_size weight
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
            print_df(dataframe, display_width)
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
        # rename the model_zoo.yml to model-index.yml but support both of them
        # for backward compatibility. In addition, model-index.yml will be put
        # in package/.mim in PR #68
        installed_path = get_installed_path(package)
        possible_metadata_paths = [
            osp.join(installed_path, '.mim', 'model-index.yml'),
            osp.join(installed_path, 'model-index.yml'),
            osp.join(installed_path, '.mim', 'model_zoo.yml'),
            osp.join(installed_path, 'model_zoo.yml'),
        ]
        for metadata_path in possible_metadata_paths:
            if osp.exists(metadata_path):
                return load(metadata_path)
        raise FileNotFoundError(
            highlighted_error(
                f'{installed_path}/model-index.yml or {installed_path}'
                '/model_zoo.yml is not found, please upgrade your '
                f'{package} to support search command'))
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
            return pickle.load(fr)
    else:
        clone_cmd = ['git', 'clone', github_url]
        if version:
            clone_cmd.extend(['-b', f'v{version}'])
        with tempfile.TemporaryDirectory() as temp:
            repo_root = osp.join(temp, PKG2PROJECT[package])
            clone_cmd.append(repo_root)
            subprocess.check_call(
                clone_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # rename the model_zoo.yml to model-index.yml but support both of
            # them for backward compatibility
            possible_metadata_paths = [
                osp.join(repo_root, 'model-index.yml'),
                osp.join(repo_root, 'model_zoo.yml'),
            ]
            for metadata_path in possible_metadata_paths:
                if osp.exists(metadata_path):
                    metadata = load(metadata_path)
                    if version:
                        with open(pkl_path, 'wb') as fw:
                            pickle.dump(metadata, fw)
                    return metadata
            raise FileNotFoundError(
                highlighted_error(
                    'model-index.yml or model_zoo.yml is not found, please '
                    f'upgrade your {package} to support search command'))


def convert2df(metadata: ModelIndex) -> DataFrame:
    """Convert metadata into DataFrame format."""

    def _parse(data: dict) -> dict:
        parsed_data = {}
        for key, value in data.items():
            unit = ''
            name = key.split()
            if '(' in key:
                # inference time (ms/im) will be splitted into `inference time`
                # and `(ms/im)`
                name, unit = name[0:-1], name[-1]
            name = '_'.join(name)
            name = cast2lowercase(name)

            if isinstance(value, str):
                parsed_data[name] = cast2lowercase(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    # inference time is a list of dict like List[dict]
                    # each item of inference time represents the environment
                    # where it is tested
                    for _value in value:
                        envs = [
                            str(_value.get(env)) for env in [
                                'hardware', 'backend', 'batch size', 'mode',
                                'resolution'
                            ]
                        ]
                        new_name = f'inference_time{unit}[{",".join(envs)}]'
                        parsed_data[new_name] = _value.get('value')
                else:
                    new_name = f'{name}{unit}'
                    parsed_data[new_name] = ','.join(cast2lowercase(value))
            else:
                new_name = f'{name}{unit}'
                parsed_data[new_name] = value

        return parsed_data

    name2model = {}
    name2collection = {}
    for collection in metadata.collections:
        collection_info = {}
        data = getattr(collection.metadata, 'data', None)
        if data:
            collection_info.update(_parse(data))

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
            model_info.update(_parse(data))

        results = getattr(model, 'results', None)
        for result in results:
            dataset = cast2lowercase(result.dataset)
            metrics = getattr(result, 'metrics', None)
            if metrics is None:
                continue

            for key, value in metrics.items():
                name = '_'.join(key.split())
                name = cast2lowercase(name)
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

    # 'inference_time>45,epoch>100' or 'inference_time>45 epoch>100' will be
    # parsed into ['inference_time>40', 'epoch>100']
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

    When sorting output with some fields, substring is spported. For example,
    if sorted_fields is ['epo'], the actual sorted fieds will be ['epochs'].

    Args:
        dataframe (DataFrame): Data to be sorted.
        sorted_fields (List[str], optional): Sort output by sorted_fields.
            Default: None.
        ascending (bool): Sort by ascending or descending. Default: True.
    """

    @typing.no_type_check
    def _filter_field(valid_fields: Series, input_fields: List[str]):
        matched_fields = []
        invalid_fields = set()
        for input_field in input_fields:
            if any(valid_fields.isin([input_field])):
                matched_fields.append(input_field)
            else:
                contain_index = valid_fields.str.contains(input_field)
                contain_fields = valid_fields[contain_index]
                if len(contain_fields) == 1:
                    matched_fields.extend(contain_fields)
                elif len(contain_fields) > 2:
                    raise ValueError(
                        highlighted_error(
                            f'{input_field} matchs {contain_fields}. However, '
                            'the number of matched fields should be 1, but got'
                            f' {len(contain_fields)}.'))
                else:
                    invalid_fields.add(input_field)
        return matched_fields, invalid_fields

    if sorted_fields is None:
        return dataframe

    sorted_fields = cast2lowercase(sorted_fields)

    valid_fields = dataframe.columns
    matched_fields, invalid_fields = _filter_field(valid_fields, sorted_fields)
    if invalid_fields:
        raise ValueError(
            highlighted_error(
                f'Expected fields: {valid_fields}, but got {invalid_fields}'))

    return dataframe.sort_values(by=matched_fields, ascending=ascending)


def select_by(dataframe: DataFrame,
              shown_fields: Optional[List[str]] = None,
              unshown_fields: Optional[List[str]] = None) -> DataFrame:
    """Select by the fields.

    When selecting some fields to be shown or be hidden, substring is spported.
    For example, if shown_fields is ['epo'], all field contain 'epo' which will
    be chosen. So the new shown field will be ['epochs'].

    Args:
        dataframe (DataFrame): Data to be filtered.
        shown_fields (List[str], optional): Fields to be outputted.
            Default: None.
        unshown_fields (List[str], optional): Fields to be hidden.
            Default: None.
    """

    @typing.no_type_check
    def _filter_field(valid_fields: Series, input_fields: List[str]):
        matched_fields = []
        invalid_fields = set()
        # record those fields which have been added to matched_fields to avoid
        # duplicated fields. Although the seen_fields is not necessary if
        # matched_fields is type of set, the order of matched_fields will be
        # not consistent with the input_fields
        seen_fields = set()
        for input_field in input_fields:
            if any(valid_fields.isin([input_field])):
                matched_fields.append(input_field)
            else:
                contain_index = valid_fields.str.contains(input_field)
                contain_fields = valid_fields[contain_index]
                if len(contain_fields) > 0:
                    matched_fields.extend(
                        field for field in (set(contain_fields) - seen_fields))
                    seen_fields.update(set(contain_fields))
                else:
                    invalid_fields.add(input_field)
        return matched_fields, invalid_fields

    if shown_fields is None and unshown_fields is None:
        return dataframe

    if shown_fields and unshown_fields:
        raise ValueError(
            highlighted_error(
                'shown_fields and unshown_fields must be mutually exclusive.'))

    valid_fields = dataframe.columns
    if shown_fields:
        shown_fields = cast2lowercase(shown_fields)
        matched_fields, invalid_fields = _filter_field(valid_fields,
                                                       shown_fields)
        if invalid_fields:
            raise ValueError(
                highlighted_error(f'Expected fields: {valid_fields}, but got '
                                  f'{invalid_fields}'))

        dataframe = dataframe.filter(items=matched_fields)
    else:
        unshown_fields = cast2lowercase(unshown_fields)  # type: ignore
        matched_fields, invalid_fields = _filter_field(valid_fields,
                                                       unshown_fields)
        if invalid_fields:
            raise ValueError(
                highlighted_error(f'Expected fields: {valid_fields}, but got '
                                  f'{invalid_fields}'))

        dataframe = dataframe.drop(columns=matched_fields)

    dataframe = dataframe.dropna(axis=0, how='all')

    return dataframe


def dump2json(dataframe: DataFrame, json_path: str) -> None:
    """Dump data frame of meta data into JSON.

    Args:
        dataframe (DataFrame): Data to be filtered.
        json_path (str): Dump output to json_path.
    """
    dataframe.to_json(json_path)


def print_df(dataframe: DataFrame, display_width: int = 80) -> None:
    """Print Dataframe into terminal."""

    def _max_len(dataframe):
        key_max_len = 0
        value_max_len = 0
        for row in dataframe.iterrows():
            for key, value in row[1].to_dict().items():
                key_max_len = max(key_max_len, len(key))
                value_max_len = max(value_max_len, len(str(value)))
        return key_max_len, value_max_len

    key_max_len, value_max_len = _max_len(dataframe)
    key_max_len += 2
    if key_max_len + value_max_len > display_width:
        value_max_len = display_width - key_max_len

    def _table(row):
        output = ''
        output += '-' * (key_max_len + value_max_len)
        output += '\n'
        output += click.style(f'config id: {row[0]}\n', fg='green')
        row_dict = row[1].dropna().to_dict()
        keys = sorted(row_dict.keys())
        for key in keys:
            output += key.ljust(key_max_len)
            value = str(row_dict[key])
            if len(value) > value_max_len:
                if value_max_len > 3:
                    output += f'{value[:value_max_len-3]}...'
                else:
                    output += '.' * value_max_len
            else:
                output += value
            output += '\n'
        return output

    def _generate_output():
        for row in dataframe.iterrows():
            yield _table(row)

    click.echo_via_pager(_generate_output())
