# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import importlib
import inspect
import logging
import os
import os.path as osp
import shutil
from functools import wraps
from typing import Callable

import torch.nn
from mmengine import mkdir_or_exist
from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.model import (
    BaseDataPreprocessor,
    BaseModel,
    BaseModule,
    ImgDataPreprocessor,
)
from mmengine.registry import Registry
from yapf.yapflib.yapf_api import FormatCode

from mim.utils import OFFICIAL_MODULES
from .common import REGISTRY_TYPES
from .flatten_func import *  # noqa: F403, F401
from .flatten_func import (
    ImportResolverTransformer,
    RegisterModuleTransformer,
    flatten_inheritance_chain,
    ignore_ast_docstring,
    postprocess_super_call,
)


def format_code(code_text: str):
    """Format the code text with yapf."""
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    try:
        code_text, _ = FormatCode(code_text, style_config=yapf_style)
    except:  # noqa: E722
        raise SyntaxError('Failed to format the config file, please '
                          f'check the syntax of: \n{code_text}')

    return code_text


def _postprocess_registry_locations(export_root_dir: str):
    """Remove the Registry.locations if it doesn't exist.

    Check the location path for Registry to load modules if the path hasn't
    been exported, then need to be removed. Finally will use the root Registry
    to find module until it actually doesn't exist.
    """
    export_module_dir = osp.join(export_root_dir, 'pack')

    with open(
            osp.join(export_module_dir, 'registry.py'), encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())

    for node in ast.walk(ast_tree):
        """node structure.

        Assign(     targets=[         Name(id='EVALUATORS', ctx=Store())],
        value=Call(         func=Name(id='Registry', ctx=Load()), args=[
        Constant(value='evaluator')],         keywords=[ keyword( arg='parent',
        value=Name(id='MMENGINE_EVALUATOR', ctx=Load())), keyword(
        arg='locations',                 value=List( elts=[
        Constant(value='pack.evaluation')], ctx=Load()))])),
        """
        if isinstance(node, ast.Call):
            need_to_be_remove = None

            for keyword in node.keywords:
                if keyword.arg == 'locations':
                    for sub_node in ast.walk(keyword):

                        # the locations of Registry already transfer to `pack`
                        # scope before. if the location path is exist, then
                        # turn to pack scope
                        if isinstance(
                                sub_node,
                                ast.Constant) and 'pack' in sub_node.value:

                            path = sub_node.value
                            if not osp.exists(
                                    osp.join(export_root_dir, path).replace(
                                        '.', osp.sep)):
                                print_log(
                                    '[ Pass ] Remove Registry.locations '
                                    f"'{osp.join(export_root_dir, path).replace('.',osp.sep)}', "  # noqa: E501
                                    'which is no need to export.',
                                    logger='export',
                                    level=logging.DEBUG)
                                need_to_be_remove = keyword
                                break

                if need_to_be_remove is not None:
                    break

            if need_to_be_remove is not None:
                node.keywords.remove(need_to_be_remove)

    with open(
            osp.join(export_module_dir, 'registry.py'), 'w',
            encoding='utf-8') as f:
        f.write(format_code(ast.unparse(ast_tree)))


def _get_all_files(directory: str):
    """Get all files of the directory.

    Args:
        directory (str): The directory path.

    Returns:
        List: Return the a list containing all the files in the directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '__init__' not in file and 'registry.py' not in file:
                file_paths.append(os.path.join(root, file))

    return file_paths


def _postprocess_importfrom_module_to_pack(file_path: str):
    """Transfer the importfrom path from "downstream repo" to export module.

    Args:
        file_path (str): The path of file needed to be transfer.

    Examples:
        >>> from mmdet.models.detectors.two_stage import TwoStageDetector
        >>> # transfer to below, if "TwoStageDetector" had been exported
        >>> from pack.models.detectors.two_stage import TwoStageDetector
    """
    from mmengine import Registry

    # _module_path_dict is a class attribute,
    # already record all the exported module and their path before
    _module_path_dict = Registry._module_path_dict

    with open(file_path, encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())

    # if the import module have the same name with the object in these file,
    # they import path won't be change
    can_not_change_module = []
    for node in ast_tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            can_not_change_module.append(node.name)

    def check_change_importfrom_node(node: ast.ImportFrom):
        """Check if the ImportFrom node should be changed.

        If the modules in node had already been exported, they will be
        separated and compose a new ast.ImportFrom node with the export
        path as the module path.

        Args:
            node (ast.ImportFrom): ImportFrom node.

        Returns:
            ast.ImportFrom | None: Return a new ast.ImportFrom node
            if one of the module in node had been export else ``None``.
        """
        export_module_path = None
        needed_change_alias = []

        for alias in node.names:
            # if the import module's name is equal to the class or function
            # name, it can not be transfer for avoiding circular import.
            if alias.name in _module_path_dict.keys(
            ) and alias.name not in can_not_change_module:

                if export_module_path is None:
                    export_module_path = _module_path_dict[alias.name]
                else:
                    assert _module_path_dict[alias.name] == \
                        export_module_path,\
                        'There are two module from the same downstream repo,'\
                        " but can't change to the same export path."

                needed_change_alias.append(alias)

        if len(needed_change_alias) != 0:
            for alias in needed_change_alias:
                node.names.remove(alias)

            return ast.ImportFrom(
                module=export_module_path, names=needed_change_alias, level=0)

        return None

    # Naming rules for searching ast syntax tree
    # - node: node of ast.Module
    # - func_sub_node: sub_node of ast.FunctionDef
    # - class_sub_node: sub_node of ast.ClassDef
    # - func_sub_class_sub_node: sub_node ast.FunctionDef in ast.ClassDef

    # record the insert_idx and node needed to be insert for later insert.
    insert_idx_and_node = {}

    insert_idx = 0
    for idx, node in enumerate(ast_tree.body):

        # search ast.ImportFrom in ast.Module scope
        # ast.Module -> ast.ImportFrom
        if isinstance(node, ast.ImportFrom):
            insert_idx += 1
            temp_node = check_change_importfrom_node(node)
            if temp_node is not None:
                if len(node.names) == 0:
                    ast_tree.body[idx] = temp_node
                else:
                    insert_idx_and_node[insert_idx] = temp_node
                    insert_idx += 1

        elif isinstance(node, ast.Import):
            insert_idx += 1

        else:
            # search ast.ImportFrom in ast.FunctionDef scope
            # ast.Module -> ast.FunctionDef -> ast.ImportFrom
            if isinstance(node, ast.FunctionDef):
                temp_func_insert_idx = ignore_ast_docstring(node)
                func_need_to_be_removed_nodes = []

                for func_sub_node in node.body:
                    if isinstance(func_sub_node, ast.ImportFrom):
                        temp_node = check_change_importfrom_node(
                            func_sub_node)  # noqa: E501
                        if temp_node is not None:
                            node.body.insert(temp_func_insert_idx, temp_node)

                        # if importfrom module is empty, the node should be remove  # noqa: E501
                        if len(func_sub_node.names) == 0:
                            func_need_to_be_removed_nodes.append(
                                func_sub_node)  # noqa: E501

                for need_to_be_removed_node in func_need_to_be_removed_nodes:
                    node.body.remove(need_to_be_removed_node)

            # search ast.ImportFrom in ast.ClassDef scope
            # ast.Module -> ast.ClassDef -> ast.ImportFrom
            #                            -> ast.FunctionDef -> ast.ImportFrom
            elif isinstance(node, ast.ClassDef):
                temp_class_insert_idx = ignore_ast_docstring(node)
                class_need_to_be_removed_nodes = []

                for class_sub_node in node.body:

                    # ast.Module -> ast.ClassDef -> ast.ImportFrom
                    if isinstance(class_sub_node, ast.ImportFrom):
                        temp_node = check_change_importfrom_node(
                            class_sub_node)
                        if temp_node is not None:
                            node.body.insert(temp_class_insert_idx, temp_node)
                        if len(class_sub_node.names) == 0:
                            class_need_to_be_removed_nodes.append(
                                class_sub_node)

                    # ast.Module -> ast.ClassDef -> ast.FunctionDef -> ast.ImportFrom  # noqa: E501
                    elif isinstance(class_sub_node, ast.FunctionDef):
                        temp_class_sub_insert_idx = ignore_ast_docstring(node)
                        func_need_to_be_removed_nodes = []

                        for func_sub_class_sub_node in class_sub_node.body:
                            if isinstance(func_sub_class_sub_node,
                                          ast.ImportFrom):
                                temp_node = check_change_importfrom_node(
                                    func_sub_class_sub_node)
                                if temp_node is not None:
                                    node.body.insert(temp_class_sub_insert_idx,
                                                     temp_node)
                                if len(func_sub_class_sub_node.names) == 0:
                                    func_need_to_be_removed_nodes.append(
                                        func_sub_class_sub_node)

                        for need_to_be_removed_node in func_need_to_be_removed_nodes:  # noqa: E501
                            class_sub_node.body.remove(need_to_be_removed_node)

                for class_need_to_be_removed_node in class_need_to_be_removed_nodes:  # noqa: E501
                    node.body.remove(class_need_to_be_removed_node)

    # lazy add new ast.ImportFrom node to ast.Module
    for insert_idx, temp_node in insert_idx_and_node.items():
        ast_tree.body.insert(insert_idx, temp_node)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(format_code(ast.unparse(ast_tree)))


def _replace_config_scope_to_pack(cfg: ConfigDict):
    """Replace the config scope from "mmxxx" to "pack".

    Args:
        cfg (ConfigDict): The config dict to be replaced.
    """

    for key, value in cfg.items():
        if key == '_scope_' or key == 'default_scope':
            cfg[key] = 'pack'
        elif isinstance(value, dict):
            _replace_config_scope_to_pack(value)


def _wrapper_all_registries_build_func(export_module_dir: str, scope: str):
    """A function to wrap all registries' build_func.

    Args:
        pack_module_dir (str): The root dir for packing modules.
        scope (str): The default scope of the config.
    """
    # copy the downstream repo.registry to pack.registry
    # and change all the registry.locations
    repo_registries = importlib.import_module('.registry', scope)
    origin_file = inspect.getfile(repo_registries)
    registry_path = osp.join(export_module_dir, 'registry.py')
    shutil.copy(origin_file, registry_path)

    # replace 'repo' name in Registry.locations to 'pack'
    with open(
            osp.join(export_module_dir, 'registry.py'), encoding='utf-8') as f:
        ast_tree = ast.parse(f.read())
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Constant):
                if scope in node.value:
                    node.value = node.value.replace(scope, 'pack')

    with open(osp.join(export_module_dir, 'registry.py'), 'w') as f:
        f.write(format_code(ast.unparse(ast_tree)))

    # prevent circular registration
    Registry._extra_module_set = set()

    # record the exported module for postprocessing the importfrom path
    Registry._module_path_dict = {}

    # prevent circular wrapper
    if Registry.build.__name__ == 'wrapper':
        Registry.build = _wrap_build(Registry.init_build_func,
                                     export_module_dir)
        Registry.get = _wrap_get(Registry.init_get_func, export_module_dir)
    else:
        Registry.init_build_func = copy.deepcopy(Registry.build)
        Registry.init_get_func = copy.deepcopy(Registry.get)
        Registry.build = _wrap_build(Registry.build, export_module_dir)
        Registry.get = _wrap_get(Registry.get, export_module_dir)


def ignore_self_cache(func):
    """Ignore the ``@lru_cache`` for function.

    Args:
        func (Callable): The function to be ignored.

    Returns:
        Callable: The function without ``@lru_cache``.
    """
    cache = {}

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        key = args
        if key not in cache:
            cache[key] = 1
            func(self, *args, **kwargs)
        else:
            return

    return wrapper


@ignore_self_cache
def _export_module(self, obj_cls: type, pack_module_dir, obj_type: str):
    """Export module.

    This function will get the object's file and export to
    ``pack_module_dir``.

    If the object is built by ``MODELS`` registry, all the objects
    as the top classes in this file, will be iteratively flattened.
    Else will be directly exported.

    The flatten logic is:
        1. get the origin file of object, which built
            by ``MODELS.build()``
        2. get all the classes in the origin file
        3. flatten all the classes but not only the object
        4. call ``flatten_module()`` to finish flatten
            according to ``class.mro()``

    Args:
        obj (object): The object to be flatten.
    """
    # find the file by obj class
    file_path = inspect.getfile(obj_cls)

    if osp.exists(file_path):
        print_log(
            f'building class: '
            f'{obj_cls.__name__} from file: {file_path}.',
            logger='export',
            level=logging.DEBUG)
    else:
        raise FileExistsError(f"file [{file_path}] doesn't exist.")

    # local origin module
    module = obj_cls.__module__
    parent = module.split('.')[0]
    new_module = module.replace(parent, 'pack')

    # Not necessary to export module implemented in `mmcv` and `mmengine`
    if parent in set(OFFICIAL_MODULES) - {'mmcv', 'mmengine'}:

        with open(file_path, encoding='utf-8') as f:
            top_ast_tree = ast.parse(f.read())

        # deal with relative import
        ImportResolverTransformer(module).visit(top_ast_tree)

        # NOTE: ``MODELS.build()`` means to flatten model module
        if self.name == 'model':

            # record all the class needed to be flattened
            need_to_be_flattened_class_names = []
            for node in top_ast_tree.body:
                if isinstance(node, ast.ClassDef):
                    need_to_be_flattened_class_names.append(node.name)

            imported_module = importlib.import_module(obj_cls.__module__)
            for cls_name in need_to_be_flattened_class_names:

                # record the exported module for postprocessing the importfrom path  # noqa: E501
                self._module_path_dict[cls_name] = new_module

                cls = getattr(imported_module, cls_name)

                for super_cls in cls.__bases__:

                    # the class only will be flattened when:
                    #   1. super class doesn't exist in this file
                    #   2. and super class is not base class
                    #   3. and super class is not torch module
                    if super_cls.__name__\
                        not in need_to_be_flattened_class_names \
                        and (super_cls not in [BaseModule,
                                               BaseModel,
                                               BaseDataPreprocessor,
                                               ImgDataPreprocessor]) \
                            and 'torch' not in super_cls.__module__:  # noqa: E501

                        print_log(
                            f'need_flatten: {cls_name}\n',
                            logger='export',
                            level=logging.INFO)

                        flatten_inheritance_chain(top_ast_tree, cls)
                        break
            postprocess_super_call(top_ast_tree)

        else:
            self._module_path_dict[obj_cls.__name__] = new_module

        # add ``register_module(force=True)`` to cover the registered modules  # noqa: E501
        RegisterModuleTransformer().visit(top_ast_tree)

        # unparse ast tree and save reformat code
        new_file_path = new_module.split('.', 1)[1].replace('.',
                                                            osp.sep) + '.py'
        new_file_path = osp.join(pack_module_dir, new_file_path)
        new_dir = osp.dirname(new_file_path)
        mkdir_or_exist(new_dir)

        with open(new_file_path, mode='w') as f:
            f.write(format_code(ast.unparse(top_ast_tree)))

    # Downstream repo could register torch module into Registry, such as
    # registering `torch.nn.Linear` into `MODELS`. We need to reserve these
    # codes in the exported module.
    elif 'torch' in module.split('.')[0]:

        # get the root registry, because it can get all the modules
        # had been registered.
        root_registry = self if self.parent is None else self.parent
        if (obj_type not in self._extra_module_set) and (
                root_registry.init_get_func(obj_type) is None):
            self._extra_module_set.add(obj_type)
            with open(osp.join(pack_module_dir, 'registry.py'), 'a') as f:

                # TODO: When the downstream repo registries' name are
                # different with mmengine, the module may not be registried
                # to the right register.
                # For example: `EVALUATOR` in mmengine, `EVALUATORS` in mmdet.
                f.write('\n')
                f.write(f'from {module} import {obj_cls.__name__}\n')
                f.write(
                    f"{REGISTRY_TYPES[self.name]}.register_module('{obj_type}', module={obj_cls.__name__}, force=True)"  # noqa: E501
                )


def _wrap_build(build_func: Callable, pack_module_dir: str):
    """wrap Registry.build()

    Args:
        build_func (Callable): ``Registry.build()``, which will be wrapped.
        pack_module_dir (str): Modules export path.
    """

    def wrapper(self, cfg: dict, *args, **kwargs):

        # obj is class instanace
        obj = build_func(self, cfg, *args, **kwargs)
        args = cfg.copy()  # type: ignore
        obj_type = args.pop('type')  # type: ignore
        obj_type = obj_type if isinstance(obj_type, str) else obj_type.__name__

        # modules in ``torch.nn.Sequential`` should be respectively exported
        if isinstance(obj, torch.nn.Sequential):
            for children in obj.children():
                _export_module(self, children.__class__, pack_module_dir,
                               obj_type)
        else:
            _export_module(self, obj.__class__, pack_module_dir, obj_type)

        return obj

    return wrapper


def _wrap_get(get_func: Callable, pack_module_dir: str):
    """wrap Registry.get()

    Args:
        get_func (Callable): ``Registry.get()``, which will be wrapped.
        pack_module_dir (str): Modules export path.
    """

    def wrapper(self, key: str):

        obj_cls = get_func(self, key)

        _export_module(self, obj_cls, pack_module_dir, key)

        return obj_cls

    return wrapper
