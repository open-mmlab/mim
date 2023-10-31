# Copyright (c) OpenMMLab. All rights reserved.
import ast
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from mmengine.logging import print_log
from mmengine.model import (
    BaseDataPreprocessor,
    BaseModel,
    BaseModule,
    ImgDataPreprocessor,
)

from .common import OBJECTS_TO_BE_PATCHED


@dataclass
class TopClassNodeInfo:
    """Contatins ``top_cls_node`` information which waiting to be flattened.

    Attributes:
        "cls_node": `ast.ClassDef`, needed to be flattened.
        "level_cls_name": The class name of flattened layer.
        "super_cls_name": The class name of super class.
        "end_idx": `len(node.body)`, the last index of all nodes
            in `ast.ClassDef`.
        "sub_func": Collects all `ast.FunctionDef` nodes in this class.
        "sub_assign": Collects all `ast.Assign` nodes in this class
    """
    cls_node: ast.ClassDef
    level_cls_name: Optional[str] = None
    super_cls_name: Optional[str] = None
    end_idx: Optional[int] = None
    sub_func: Dict[str, ast.FunctionDef] = field(default_factory=dict)
    sub_assign: Dict[str, ast.Assign] = field(default_factory=dict)


@dataclass
class TopAstInfo:
    """top_ast_info (TopClassNodeInfo): Contatins the initial information of
    the ``top_ast_tree``.

    Attributes:
        class_dict (TopClassNodeInfo): Contatins ``top_cls_node``
            information which waiting to be flattened.
        importfrom_dict (Dict[str, List[ast.ImportFrom, List[str]]]):
            Contatins the simple alias of `ast.ImportFrom` information.
        import_list (List[str]): Contatins imported module name.
        assign_list (List[ast.Assign]): Contatins global assign.
        if_list (List[ast.If]): Contatins global `ast.If`.
        try_list (List[ast.Try]): Contatins global `ast.Try`.
        importfrom_asname_dict (Dict[str, ast.ImportFrom]): Contatins the
            asname alias of `ast.ImportFrom` information.
    """
    class_dict: TopClassNodeInfo
    importfrom_dict: Dict[str, Tuple[ast.ImportFrom,
                                     List[str]]] = field(default_factory=dict)
    importfrom_asname_dict: Dict[str,
                                 ast.ImportFrom] = field(default_factory=dict)
    import_list: List[str] = field(default_factory=list)
    try_list: List[ast.Try] = field(default_factory=list)
    if_list: List[ast.If] = field(default_factory=list)
    assign_list: List[ast.Assign] = field(default_factory=list)


@dataclass
class ExtraSuperAstInfo:
    """extra_super_ast_info (ExtraSuperAstInfo): Contatins the extra
    information of the ``super_ast_tree`` which needed to be consider.

    Attributes:
        used_module (Dict[ast.AST, Set[str]): The dict records
            the node and a set
        extra_import (Dict[ast.ImportFrom, List[ast.alias]]):
            Records extra `ast.ImportFrom` nodes and their modules
            in the super class file.
        extra_importfrom (List[ast.alias]): Records extra
            `ast.Import` nodes' alias in the super class file.
    """
    used_module: Dict[ast.AST, Set[str]] = field(
        default_factory=lambda: defaultdict(set))
    extra_import: List[ast.alias] = field(default_factory=list)
    extra_importfrom: Dict[str, Tuple[ast.ImportFrom,
                                      List[str]]] = field(default_factory=dict)


@dataclass
class NeededNodeInfo:
    """need_node_info (NeededNodeInfo): Contatins the needed node by comparing
    ``super_ast_tree`` and ``top_ast_tree``.

    Attributes:
        need_importfrom_nodes (set, optional): Collect the needed
            `ast.ImportFrom` node from ``ExtraSuperAstInfo.extra_importfrom``.
        need_import_alias_asname (set, optional): Collect the needed
            `ast.Import` asname nodes from ``ExtraSuperAstInfo.extra_import``.
        need_import_alias (set, optional): Collect the needed `ast.Import`
            nodes from ``ExtraSuperAstInfo.extra_import``.
    """
    need_importfrom_nodes: Set[ast.ImportFrom] = field(default_factory=set)
    need_import_alias_asname: Set[ast.alias] = field(default_factory=set)
    need_import_alias: Set[ast.alias] = field(default_factory=set)


def get_len(used_modules_dict_set: Dict[ast.AST, Set[str]]):
    """Get the sum of used modules.

    Args:
        used_module_dict (Dict[ast.AST, Set[str]]): Records
            the node and a set including module names it uses.

    Returns:
        int: The sum of used modules.
    """
    len_sum = 0
    for name_list in used_modules_dict_set.values():
        len_sum += len(name_list)

    return len_sum


def record_used_node(node: ast.AST, used_module_dict_set: Dict[ast.AST,
                                                               Set[str]]):
    """Recode the node had been use and no need to remove.

    Args:
        node (ast.AST): AST Node..
        used_module_dict_set (Dict[ast.AST, Set[str]): The dict records
            the node and a set including module names it uses.

    Examples:
        >>> # a = nn.MaxPool1d()
        >>> Assign(
        >>>     targets=[
        >>>         Name(id='a', ctx=Store())],
        >>>     value=Call(
        >>>         func=Attribute(
        >>>             value=Name(id='nn', ctx=Load()),
        >>>             attr='MaxPool1d',
        >>>             ctx=Load()),
        >>>         args=[],
        >>>         keywords=[]))
        >>>
        >>> used_module_dict_set[ast.Assign] = set('nn', 'a')
    """
    # only traverse the body of FunctionDef to ignore the input args
    if isinstance(node, ast.FunctionDef):
        for func_sub_node in node.body:
            for func_sub_sub_node in ast.walk(func_sub_node):
                if isinstance(func_sub_sub_node, ast.Name):
                    used_module_dict_set[node].add(func_sub_sub_node.id)

    # iteratively process the body of ClassDef
    elif isinstance(node, ast.ClassDef):
        for class_sub_node in node.body:
            record_used_node(class_sub_node, used_module_dict_set)

    else:
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Name):
                used_module_dict_set[node].add(sub_node.id)


def if_need_remove(node: ast.AST, used_module_dict_set: Dict[ast.AST,
                                                             Set[str]]):
    """Justify if the node should be remove.

    If the node not be use actually, it will be removed.

    Args:
        node (ast.AST): AST Node.
        used_module_dict_set (Dict[ast.AST, Set[str]]): The dict records the
            node and a set including module names it uses.

    Returns:
        bool: if not be used then return "True" meaning to be removed,
            else "False".
    """
    if isinstance(node, ast.Assign):
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            raise TypeError(f'expect the targets in ast.Assign is ast.Name\
                            but got {type(node.targets[0])}')
    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
        name = node.name
    else:
        # HARD CODE: if not the above type will directly remove.
        return True

    for name_list in used_module_dict_set.values():
        if name in name_list:
            return False

    return True


def is_in_top_ast_tree(node: ast.AST,
                       top_ast_info: TopAstInfo,
                       top_cls_and_func_node_name_list: List[str] = []):
    """Justify if the module name already exists in ``top_ast_tree``.

    Args:
        node (ast.AST): AST Node.
        top_ast_info (TopClassNodeInfo): Contatins the initial information
                    of the ``top_ast_tree``.
        top_cls_and_func_node_name_list (List[str], optional): Containing
            `Class` or `Function` name in ``top_ast_tree``. Defaults to "[]"

    Returns:
        bool: if the module name already exists in ``top_ast_tree`` return
            "True", else "False".
    """
    if isinstance(node, ast.Assign):
        for _assign in top_ast_info.assign_list:
            if ast.dump(_assign) == ast.dump(node):
                return True

    elif isinstance(node, ast.Try):
        for _try in top_ast_info.try_list:
            if ast.dump(_try) == ast.dump(node):
                return True

    elif isinstance(node, ast.If):
        for _if in top_ast_info.if_list:
            if ast.dump(_if) == ast.dump(node):
                return True

    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
        if node.name in top_cls_and_func_node_name_list:
            return True

    return False


def ignore_ast_docstring(node: Union[ast.ClassDef, ast.FunctionDef]):
    """Get the insert key ignoring the docstring.

    Args:
        node (ast.ClassDef | ast.FunctionDef): AST Node.

    Returns:
        int: The beginning insert position of the node.
    """
    insert_index = 0

    for sub_node in node.body:
        if isinstance(sub_node, ast.Expr):
            insert_index += 1

            # HARD CODE: prevent from some ast.Expr like warning.warns which
            # need module "warning"
            for sub_sub_node in ast.walk(sub_node):
                if isinstance(sub_sub_node, ast.Name):
                    return 0
        else:
            break

    return insert_index


def find_local_import(node: Union[ast.FunctionDef, ast.Assign],
                      extra_super_ast_info: ExtraSuperAstInfo,
                      need_node_info: Optional[NeededNodeInfo] = None):
    """Find the needed Import and ImportFrom of the node.

    Args:
        node (ast.FunctionDef | ast.Assign)
        extra_super_ast_info (ExtraSuperAstInfo): Contatins the extra
            information of the ``super_ast_tree`` which needed to be consider.
        need_node_info (NeededNodeInfo, optional): Contatins the needed node
            by comparing ``super_ast_tree`` and ``top_ast_tree``.

    Returns:
        need_node_info
    """
    if need_node_info is None:
        need_node_info = NeededNodeInfo()

    # get all the used modules' name in specific node
    used_module = extra_super_ast_info.used_module[node]

    if len(used_module) != 0:

        # record all used ast.ImportFrom nodes
        for import_node, alias_list in \
                extra_super_ast_info.extra_importfrom.values():
            for module in used_module:
                if module in alias_list:  # type: ignore[operator]
                    need_node_info.need_importfrom_nodes.add(
                        import_node)  # type: ignore[arg-type]  # noqa: E501
                    continue

        # record all used ast.Import nodes
        for alias in extra_super_ast_info.extra_import:
            if alias.asname is not None:
                if alias.asname in used_module:
                    need_node_info.need_import_alias_asname.add(alias)
            else:
                if alias.name in used_module:
                    need_node_info.need_import_alias.add(alias)

    return need_node_info


def add_local_import_to_func(node, need_node_info: NeededNodeInfo):
    """Add the needed ast.ImportFrom and ast.Import to ast.Function.

    Args:
        node (ast.FunctionDef | ast.Assign)
        need_node_info (NeededNodeInfo): Contatins the needed node by
            comparing ``super_ast_tree`` and ``top_ast_tree``.
    """
    insert_index = ignore_ast_docstring(node)

    for importfrom_node in need_node_info.need_importfrom_nodes:
        node.body.insert(insert_index, importfrom_node)

    if len(need_node_info.need_import_alias) != 0:
        node.body.insert(
            insert_index,
            ast.Import(
                names=[alias for alias in need_node_info.need_import_alias]))

    for alias in need_node_info.need_import_alias_asname:
        node.body.insert(insert_index, ast.Import(names=[alias]))


def add_local_import_to_class(cls_node: ast.ClassDef,
                              extra_super_ast_info: ExtraSuperAstInfo,
                              new_node_begin_index=-9999):
    """Add the needed `ast.ImportFrom` and `ast.Import` to `ast.Class`'s
    sub_nodes, including `ast.Assign` and `ast.Function`.

    Traverse `ast.ClassDef` node, recode all the used modules of class
    attributes like the `ast.Assign`, and this needed `ast.ImportFrom` and
    `ast.Import` will be add to the top of the cls_node.body. More, for each
    sub functions in this class, we will process them as glabal functions
    by using :func:`find_local_import` and :func:`add_local_import_to_func`.

    Args:
        cls_node (ast.ClassDef)
        used_module_dict_super (Dict[ast.AST, Set[str]]): Records
            the node and a set including module names it uses.
        new_node_begin_index (int, optional): The index of the last node
            of cls_node.body.
    """
    # for later add all the needed ast.ImportFrom and ast.Import nodes for
    # class attributes
    later_need_node_info = NeededNodeInfo()

    for i, cls_sub_node in enumerate(cls_node.body):
        if isinstance(cls_sub_node, ast.Assign):

            find_local_import(
                node=cls_sub_node,
                extra_super_ast_info=extra_super_ast_info,
                need_node_info=later_need_node_info,
            )

        # ``i >= new_node_begin_index`` means only processing those
        # newly added nodes.
        elif isinstance(cls_sub_node,
                        ast.FunctionDef) and i >= new_node_begin_index:
            need_node_info = find_local_import(
                node=cls_sub_node,
                extra_super_ast_info=extra_super_ast_info,
            )

            add_local_import_to_func(
                node=cls_sub_node, need_node_info=need_node_info)

    # add all the needed ast.ImportFrom and ast.Import nodes for
    # class attributes
    add_local_import_to_func(
        node=cls_node, need_node_info=later_need_node_info)


def init_prepare(top_ast_tree: ast.Module, flattened_cls_name: str):
    """Collect the initial information of the ``top_ast_tree``.

    Args:
        top_ast_tree (ast.Module): Ast tree which will be continuelly updated
            contains the class needed to be flattened.
        flattened_cls_name (str): The name of the class needed to
            be flattened.

    Returns:
        top_ast_info (TopClassNodeInfo): Contatins the initial information
            of the ``top_ast_tree``.
    """
    class_dict = TopClassNodeInfo(None)  # type: ignore
    top_ast_info = TopAstInfo(class_dict)

    # top_ast_tree scope
    for node in top_ast_tree.body:

        # ast.Module -> ast.ImporFrom
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.names[0].asname is not None:
                top_ast_info.importfrom_asname_dict[node.module] = node
            elif node.module is not None:
                # yapf: disable
                top_ast_info.importfrom_dict[node.module] = (
                    node,
                    [alias.name for alias in node.names]  # type: ignore
                )
                # yapf: enable

        # ast.Module -> ast.Import
        elif isinstance(node, ast.Import):
            top_ast_info.import_list.extend([
                alias.name if alias.asname is None else alias.asname
                for alias in node.names
            ])

        # ast.Module -> ast.Assign
        elif isinstance(node, ast.Assign):
            top_ast_info.assign_list.append(node)

        # ast.Module -> ast.Try
        elif isinstance(node, ast.Try):
            top_ast_info.try_list.append(node)

        # ast.Module -> ast.If
        elif isinstance(node, ast.If):
            top_ast_info.if_list.append(node)

        # ast.Module -> specific ast.ClassDef
        elif isinstance(node,
                        ast.ClassDef) and node.name == flattened_cls_name:

            # ``level_cls_name`` is the actual name in mro in this
            # flatten level
            #
            # Examples:
            #   >>> # level_cls_name = 'A'
            #   >>> class A(B)  class B(C)
            #   >>>
            #   >>> # after flattened
            #   >>> # level_cls_name = 'B'
            #   >>> class A(C)
            top_ast_info.class_dict.cls_node = node
            top_ast_info.class_dict.level_cls_name = flattened_cls_name
            top_ast_info.class_dict.super_cls_name = node.bases[
                0].id  # type: ignore[attr-defined]
            top_ast_info.class_dict.end_idx = len(node.body)

            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    top_ast_info.class_dict.sub_func[sub_node.name] = sub_node

                elif isinstance(sub_node, ast.Assign):
                    top_ast_info.class_dict.sub_assign[sub_node.targets[
                        0].id] = sub_node  # type: ignore[attr-defined]

    assert top_ast_info.class_dict is not None, \
        f"The class [{flattened_cls_name}] doesn't exist in the ast tree."

    return top_ast_info


def collect_needed_node_from_super(super_ast_tree: ast.Module,
                                   top_ast_info: TopAstInfo):
    """Flatten specific model class.

    This function traverses `super_ast_tree` and collection information
    comparing with `top_cls_node` in ``top_ast_tree``.

    Need to process `ImportFrom, Import, ClassDef, If, Try, Assign`.
        - ImportFrom soulution: If the node.module already exist in
            ``top_ast_tree``, we will merge it's alias, but separately
            deal with asname and simple ImportFrom node. Else will be
            consider extra ImportFrom.
            1. asname alias use :func:`ast.dump` to compare.
            2. simple alias use :func:`set` to get the union set.

        - ClassDef solution: The main part. First we get the `top_cls_node`
            and replace :func:`super` call in it with information in
            `super_cls_node`. Second, traverse `super_ast_tree` to get those
            super class `ast.FunctionDef` and super class `ast.Assign` needed
            to add to `top_cls_node`. We should rename the function called by
            :func:`super`. Last, insert all the needed super node into
            `top_cls_node` and update `top_cls_node.bases`.
            Finish class flatten.

    Args:
        super_ast_tree (ast.Module): The super ast tree including the super
            class in the specific flatten class's mro.
        top_ast_info (TopClassNodeInfo): Contatins the initial information
            of the ``top_ast_tree``.

    Returns:
        extra_super_ast_info (ExtraSuperAstInfo): Contatins the extra
            information of the ``super_ast_tree`` which needed to be consider.
    """
    extra_super_ast_info = ExtraSuperAstInfo()

    # super_ast_tree scope
    for node in super_ast_tree.body:

        # ast.Module -> ast.ImportFrom
        if isinstance(node, ast.ImportFrom):

            # HARD CODE: if ast.alias has asname, we consider it only contains
            # one module

            # Examples:
            #   >>> # common style
            #   >>> for abc import a as A
            #   >>> # not recommonded style
            #   >>> for abc import a as A, B
            if node.names[0].asname is not None:
                if node.module in top_ast_info.importfrom_asname_dict:
                    top_importfrom_node = \
                        top_ast_info.importfrom_asname_dict.get(node.module)

                    if ast.dump(top_importfrom_node  # type: ignore[arg-type]
                                ) != ast.dump(node):
                        # yapf: disable
                        alias_names = [alias.name
                                       if alias.asname is None
                                       else alias.asname
                                       for alias in node.names]
                        extra_super_ast_info.extra_importfrom[node.module] = \
                            (node, alias_names)
                        # yapf: enable

            # only name
            else:
                # the ast.alias import from the same module will be merge into
                # one ast.ImportFrom
                if node.module in top_ast_info.importfrom_dict:
                    (top_importfrom_node, last_names) = \
                        top_ast_info.importfrom_dict.get(node.module)  # type: ignore  # noqa: E501

                    current_names = [alias.name for alias in node.names
                                     ]  # type: ignore[misc]  # noqa: E501
                    last_names = list(set(last_names + current_names))
                    top_importfrom_node.names = [
                        ast.alias(name=name) for name in last_names
                    ]  # type: ignore[attr-defined]

                    # NOTE: update the information of top_ast_tree
                    top_ast_info.importfrom_dict[node.module] = (
                        top_importfrom_node, last_names)

                # those don't exist ast.ImportFrom will be later added
                elif node.module is not None:
                    # yapf: disable
                    alias_names = [alias.name
                                   if alias.asname is None
                                   else alias.asname
                                   for alias in node.names]
                    extra_super_ast_info.extra_importfrom[node.module] = \
                        (node, alias_names)
                    # yapf: enable

        # ast.Module -> ast.Import
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    if alias.asname not in top_ast_info.import_list:
                        extra_super_ast_info.extra_import.append(alias)
                else:
                    if alias.name not in top_ast_info.import_list:
                        extra_super_ast_info.extra_import.append(alias)

        # ast.Module -> ast.Try / ast.Assign / ast.If
        elif (isinstance(node, ast.Try) or isinstance(node, ast.Assign) or
                isinstance(node, ast.If)) and \
                not is_in_top_ast_tree(node, top_ast_info=top_ast_info):
            record_used_node(node, extra_super_ast_info.used_module)

        # ast.Module -> ast.ClassDef
        elif isinstance(
                node, ast.ClassDef
        ) and node.name == top_ast_info.class_dict.super_cls_name:

            # get the specific flattened class node in the top_ast_tree
            top_cls_node = top_ast_info.class_dict.cls_node

            # process super, including below  circumstances:
            #   class A(B) and class B(C)
            #   1. super().xxx(): directly replace to self.B_xxx()
            #   2. super(A, self).xxx(): directly replace to self.B_xxx()
            #   3. super(B, self).xxx(): waiting the level_cls_name=B, then
            #      replace to self.C_xxx()

            # HARD CODE: if B doesn't exist self.xxx(), it will not deal with
            # super(A, self).xxx() until the :func:`postprocess_super()` will
            # remove all the args in ``super(args)``, then change to
            # ``super()``. In another word, if super doesn't replace in the
            # correct level, it will be turn to use the root super
            # class' method.
            super_func = []
            for sub_node in ast.walk(top_cls_node):  # type: ignore[arg-type]

                if isinstance(sub_node, ast.Attribute) \
                        and hasattr(sub_node, 'value') \
                        and isinstance(sub_node.value, ast.Call) \
                        and isinstance(sub_node.value.func, ast.Name) \
                        and sub_node.value.func.id == 'super':  # noqa: E501
                    """
                    Examples: super().__init__()
                        >>> Expr(
                        >>>     value=Call(
                        >>>         func=Attribute(
                        >>>             value=Call(
                        >>>                 func=Name(id='super',
                        ctx=Load()),
                        >>>                 args=[],
                        >>>                 keywords=[]),
                        >>>             attr='__init__',
                        >>>             ctx=Load()),
                        >>>         args=[],
                        >>>         keywords=[]))],
                    """
                    # Only flatten super syntax:
                    # 1. super().func_call
                    # 2. super(top_cls_name, self).func_call
                    if len(
                            sub_node.value.args
                    ) != 0 and sub_node.value.args[  # type: ignore[attr-defined]  # noqa: E501
                            0].id != top_ast_info.class_dict.level_cls_name:
                        continue

                    # search and justify if the .xxx() function in the
                    # super node
                    for super_cls_sub_node in node.body:
                        if isinstance(
                                super_cls_sub_node, ast.FunctionDef
                        ) and sub_node.attr == \
                                super_cls_sub_node.name:
                            super_func.append(sub_node.attr)
                            sub_node.value = \
                                sub_node.value.func
                            sub_node.value.id = 'self'
                            sub_node.value.args = [  # type: ignore[attr-defined]  # noqa: E501
                            ]
                            sub_node.attr = node.name + \
                                '_' + sub_node.attr
                            break

            # record all the needed ast.ClassDef -> ast.FunctionDef
            #                   and ast.ClassDef -> ast.Assign
            func_need_append = []
            assign_need_append = []
            for super_cls_sub_node in node.body:

                # ast.Module -> ast.ClassDef -> ast.FunctionDef
                if isinstance(super_cls_sub_node, ast.FunctionDef):

                    # the function call as super().xxx() should be rename to
                    # super_cls_name_xxx()
                    if super_cls_sub_node.name in super_func:
                        super_cls_sub_node.name = node.name + '_' + \
                            super_cls_sub_node.name
                        func_need_append.append(super_cls_sub_node)

                        # NOTE: update the information of top_ast_tree
                        top_ast_info.class_dict.sub_func[
                            super_cls_sub_node.name] = super_cls_sub_node
                        record_used_node(super_cls_sub_node,
                                         extra_super_ast_info.used_module)

                    # the function don't exist in top class node will be
                    # directly imported
                    elif super_cls_sub_node.name not in \
                            top_ast_info.class_dict.sub_func:
                        func_need_append.append(super_cls_sub_node)
                        # if super_cls_sub_node.name == "_init_cls_convs":
                        # NOTE: update the information of top_ast_tree
                        top_ast_info.class_dict.sub_func[
                            super_cls_sub_node.name] = super_cls_sub_node
                        record_used_node(super_cls_sub_node,
                                         extra_super_ast_info.used_module)

                # ast.Module -> ast.ClassDef -> ast.Assign
                elif isinstance(super_cls_sub_node, ast.Assign):
                    add_flag = True

                    for name in top_ast_info.class_dict.sub_assign.keys():
                        if name == super_cls_sub_node.targets[
                                0].id:  # type: ignore[attr-defined]
                            add_flag = False

                    if add_flag:
                        assign_need_append.append(super_cls_sub_node)

                        # NOTE: update the information of top_ast_tree
                        top_ast_info.class_dict.end_idx += 1  # type: ignore
                        top_ast_info.class_dict.sub_assign[
                            super_cls_sub_node.
                            targets[0].  # type: ignore[attr-defined]
                            id] = super_cls_sub_node
                        record_used_node(super_cls_sub_node,
                                         extra_super_ast_info.used_module)

            # add all the needed ast.ClassDef -> ast.FunctionDef and
            # ast.ClassDef -> ast.Assign to top_cls_node
            if len(assign_need_append) != 0:
                insert_idx = ignore_ast_docstring(
                    top_cls_node)  # type: ignore[arg-type]  # noqa: E501

                assign_need_append.reverse()
                for assign in assign_need_append:
                    top_cls_node.body.insert(
                        insert_idx,
                        assign)  # type: ignore[arg-type]  # noqa: E501

            func_name = [func.name for func in func_need_append]
            print_log(
                f'Add function {func_name}.',
                logger='export',
                level=logging.DEBUG)
            top_cls_node.body.extend(
                func_need_append)  # type: ignore[arg-type]  # noqa: E501

            # complete this level flatten, change the super class of
            # top_cls_node
            top_cls_node.bases = node.bases  # type: ignore[attr-defined]

            # NOTE: update the information of top_ast_tree
            top_ast_info.class_dict.level_cls_name = node.name
            # HARD CODE: useless, only for preventing error when ``nn.xxx``
            # as the last super class
            top_ast_info.class_dict.super_cls_name = node.bases[  # type: ignore  # noqa: E501
                0].id \
                if isinstance(node.bases[0], ast.Name) else node.bases[0]

    return extra_super_ast_info


def postprocess_top_ast_tree(
    super_ast_tree: ast.Module,
    top_ast_tree: ast.Module,
    extra_super_ast_info: ExtraSuperAstInfo,
    top_ast_info: TopAstInfo,
):
    """Postprocess ``top_ast_tree`` with the information collected by
    traversing super_ast_tree.

    This function finishes:
        1. get all the nodes needed by ``top_ast_tree`` and
            exist in super_ast_tree
        2. add as local import for the new add function from super_ast_tree
            preventing from covering by the same name modules on the top.
        3. add extra Import/ImportFrom of super_ast_tree to
            the top of ``top_ast_tree``

    Args:
        super_ast_tree (ast.Module): The super ast tree including the super
            class in the specific flatten class's mro.
        top_ast_tree (ast.Module): The top ast tree contains the classes
            directly called, which is  continuelly updated.
        extra_super_ast_info (ExtraSuperAstInfo): Contatins the extra
            information of the ``super_ast_tree`` which needed to be consider.
        top_ast_info (TopClassNodeInfo): Contatins the initial information
            of the ``top_ast_tree``.
    """

    # record all the imported module
    imported_module_name_upper = set()
    for importfrom_node, alias_list in top_ast_info.importfrom_dict.values():
        for alias in alias_list:
            imported_module_name_upper.add(alias)

    for name in top_ast_info.import_list:
        imported_module_name_upper.add(name)

    # HARD CODE: there will be a situation that the super class and the sub
    # class exist in the same file, the super class should
    imported_module_name_upper.discard(
        top_ast_info.class_dict.super_cls_name)  # type: ignore  # noqa: E501

    # find the needed ast.ClassDef or ast.FunctionDef in super_ast_tree
    need_append_node_name: Set[str] = set()
    if get_len(extra_super_ast_info.used_module) != 0:

        while True:
            origin_len = get_len(extra_super_ast_info.used_module)

            # super_ast_tree scope
            for node in super_ast_tree.body:
                if (isinstance(node, ast.ClassDef)
                    or isinstance(node, ast.FunctionDef)) \
                    and not if_need_remove(node,
                                           extra_super_ast_info.used_module) \
                        and node.name not in imported_module_name_upper:

                    need_append_node_name.add(node.name)
                    record_used_node(node, extra_super_ast_info.used_module)

            # if there is no longer extra new module, then search break
            if get_len(extra_super_ast_info.used_module) == origin_len:
                break

    # record insert_idx and classes and functions' name in top_ast_tree
    insert_idx = 0
    top_cls_func_node_name_list = []
    for top_node in top_ast_tree.body:

        if isinstance(top_node, ast.Import) or isinstance(
                top_node, ast.ImportFrom):
            insert_idx += 1
        elif isinstance(top_node, ast.FunctionDef) or isinstance(
                top_node, ast.ClassDef):
            top_cls_func_node_name_list.append(top_node.name)
    # super_ast_tree scope
    for node in super_ast_tree.body:

        # ast.Module -> ast.Try / ast.Assign / ast.If
        if (isinstance(node, ast.Try) or isinstance(node, ast.Assign)
            or isinstance(node, ast.If)) \
            and not is_in_top_ast_tree(node,
                                       top_ast_info,
                                       top_cls_func_node_name_list):

            # NOTE: postprocess top_ast_tree
            top_ast_tree.body.insert(insert_idx, node)
            insert_idx += 1

            # NOTE: update the information of top_ast_tree
            if isinstance(node, ast.Try):
                top_ast_info.try_list.append(node)
            elif isinstance(node, ast.Assign):
                top_ast_info.assign_list.append(node)
            elif isinstance(node, ast.If):
                top_ast_info.if_list.append(node)

        elif not if_need_remove(node, extra_super_ast_info.used_module) \
            and not is_in_top_ast_tree(node,
                                       top_ast_info,
                                       top_cls_func_node_name_list) \
                and node.name in need_append_node_name:  # type: ignore[attr-defined]  # noqa: E501

            # ast.Module -> ast.FunctionDef
            if isinstance(node, ast.FunctionDef):

                need_node_info = find_local_import(
                    node=node, extra_super_ast_info=extra_super_ast_info)

                add_local_import_to_func(
                    node=node, need_node_info=need_node_info)

                # NOTE: postprocess top_ast_tree
                top_ast_tree.body.insert(insert_idx, node)
                insert_idx += 1

            # ast.Module -> ast.ClassDef
            elif isinstance(node, ast.ClassDef):
                add_local_import_to_class(
                    cls_node=node, extra_super_ast_info=extra_super_ast_info)

                # NOTE: postprocess top_ast_tree
                top_ast_tree.body.insert(insert_idx, node)
                insert_idx += 1

    # the newly add functions in top_cls_node also should add local import
    top_cls_node = top_ast_info.class_dict.cls_node
    add_local_import_to_class(
        cls_node=top_cls_node,  # type: ignore[arg-type]
        extra_super_ast_info=extra_super_ast_info,
        new_node_begin_index=top_ast_info.class_dict.end_idx)

    # update the end_idx for next time postprocess
    top_ast_info.class_dict.end_idx = len(
        top_cls_node.body)  # type: ignore[attr-defined]

    # postprocess global import
    # all the extra import will be inserted to the top of the top_ast_tree
    need_node_info = NeededNodeInfo()

    for module_name, (
            sub_node,
            name_list) in extra_super_ast_info.extra_importfrom.items():
        need_node_info.need_importfrom_nodes.add(sub_node)
        top_ast_info.importfrom_dict[module_name] = (sub_node, name_list)

    for alias in extra_super_ast_info.extra_import:  # type: ignore
        if alias.asname is not None:  # type: ignore
            need_node_info.need_import_alias_asname.add(alias)
            top_ast_info.import_list.append(alias.asname)  # type: ignore
        else:
            need_node_info.need_import_alias.add(alias)
            top_ast_info.import_list.append(alias.name)  # type: ignore

    add_local_import_to_func(node=top_ast_tree, need_node_info=need_node_info)


def postprocess_super_call(ast_tree: ast.Module):
    """Postprocess those don't successfully process ``super()`` call.

    This is a hard code.
    All the ``super(args)`` with args will be remove and turn to ``super()``.

    Args:
        ast_tree (ast.Module)
    """
    for node in ast_tree.body:
        if isinstance(node, ast.ClassDef):
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Attribute) \
                        and hasattr(sub_node, 'value') \
                        and isinstance(sub_node.value, ast.Call) \
                        and isinstance(sub_node.value.func, ast.Name) \
                        and sub_node.value.func.id == 'super':
                    sub_node.value.args = []


def flatten_inheritance_chain(top_ast_tree: ast.Module, obj_cls: type):
    """Flatten the module. (Key Interface)

    The logic of the ``flatten_module`` are as below.
    First, get the inheritance_chain by ``class.mro()`` and prune it.
    Second, get the file of chosen top class and parse it to
        be ``top_ast_tree``.
    Third, call ``init_prepare()`` to collect the information of
        ``top_ast_tree``.

    Last, for each super class in the inheritance_chain, we will do:
        1. parse the super class file as  ``super_ast_tree`` and
            do preprocess.
        2. call ``flatten_model()`` to visit necessary node
            in ``super_ast_tree`` to change needed flattened class node and
            record the information for flatten.
        3. call ``postprocess_ast_tree()`` with the information got from
           ``flatten_model()`` to change the ``top_ast_tree``.

    In summary, ``top_ast_tree`` is the most important ast tree maintained and
    updated from the begin to the end.

    Args:
        top_ast_tree (ast.Module): The top ast tree contains the classes
            directly called, which is continually updated.
        obj_cls (object): The chosen top class to be flattened.
    """
    print_log(
        f'------------- Starting flatten model [{obj_cls.__name__}] '
        f'-------------\n'
        f'\n    *[mro]: {obj_cls.mro()}\n',
        logger='export',
        level=logging.INFO)

    # get inheritance_chain
    inheritance_chain = []
    for cls in obj_cls.mro()[1:]:
        if cls in [
                BaseModule, BaseModel, BaseDataPreprocessor,
                ImgDataPreprocessor
        ] or 'torch' in cls.__module__:
            break
        inheritance_chain.append(cls)

    # collect the init information of ``top_ast_tree``
    top_ast_info = init_prepare(top_ast_tree, obj_cls.__name__)

    # iteratively deal with the super class
    for cls in inheritance_chain:

        modul_pth = inspect.getfile(cls)
        with open(modul_pth) as f:
            super_ast_tree = ast.parse(f.read())

        ImportResolverTransformer(cls.__module__).visit(super_ast_tree)
        # collect the difference between ``top_ast_tree`` and ``super_ast_tree``  # noqa: E501
        extra_super_ast_info = collect_needed_node_from_super(
            super_ast_tree=super_ast_tree, top_ast_info=top_ast_info)

        # update ``top_ast_tree``
        postprocess_top_ast_tree(
            super_ast_tree,
            top_ast_tree,
            extra_super_ast_info=extra_super_ast_info,
            top_ast_info=top_ast_info,
        )

    print_log(
        f'------------- Ending flatten model [{obj_cls.__name__}] '
        f'-------------\n',
        logger='export',
        level=logging.INFO)


class RegisterModuleTransformer(ast.NodeTransformer):
    """Deal with repeatedly registering same module.

    Add "force=True" to register_module(force=True) for covering registered
    modules.
    """

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.attr == 'register_module':
                    new_keyword = ast.keyword(
                        arg='force', value=ast.NameConstant(value=True))
                    if node.keywords is None:
                        node.keywords = [new_keyword]
                    else:
                        node.keywords.append(new_keyword)
        return node


class ImportResolverTransformer(ast.NodeTransformer):
    """Deal with the relative import problem.

    Args:
        import_prefix (str): The import prefix for the visit ast code

    Examples:
        >>> # file_path = '/home/username/miniconda3/envs/env_name/lib' \
        >>>               '/python3.9/site-packages/mmdet/models/detectors' \
        >>>               '/dino.py'
        >>> import_prefix = mmdet.models.detector
    """

    def __init__(self, import_prefix: str):
        super().__init__()
        self.import_prefix = import_prefix

    def visit_ImportFrom(self, node):
        matched = self._match_alias_registry(node)
        if matched is not None:
            # In an ideal scenario, the `ImportResolverTransformer` would
            # modify the import sources of all `Registry` from downstream
            # algorithm libraries (`mmdet`) to `pack`, for example, convert
            # `from mmdet.models import DETECTORS` to
            # `from pack.models import DETECTORS`.

            # However, some algorithm libraries, such as `mmpose`, provide
            # aliases for `MODELS`, `TASK_UTILS`, and other registries,
            # as seen here: https://github.com/open-mmlab/mmpose/blob/537bd8e543ab463fb55120d5caaa1ae22d6aaf06/mmpose/models/builder.py#L13.  # noqa: E501

            # For these registries with aliases, we cannot directly import from
            # `pack.registry` because `pack.registry` is copied from
            # `mmpose.registry` and does not contain these aliases.

            # Therefore, we gather all registries with aliases under
            # `mim._internal.export.patch_utils` and hardcode the redirection
            # of import sources.
            if matched == 'MODELS':
                node.module = 'mim._internal.export.patch_utils.patch_model'
            elif matched == 'TASK_UTILS':
                node.module = 'mim._internal.export.patch_utils.patch_task'
            node.level = 0
            return node

        else:
            # deal with relative import
            if node.level != 0:
                import_prefix = '.'.join(
                    self.import_prefix.split('.')[:-node.level])
                if node.module is not None:
                    node.module = import_prefix + '.' + node.module
                else:
                    # from . import xxx
                    node.module = import_prefix
                node.level = 0

            if 'registry' in node.module \
                    and not node.module.startswith('mmengine'):
                node.module = 'pack.registry'

        return node

    def _match_alias_registry(self, node) -> Optional[str]:
        match_patch_key = None
        for key, list_value in OBJECTS_TO_BE_PATCHED.items():
            for alias in node.names:
                if alias.name in list_value:
                    match_patch_key = key
                    break

            if match_patch_key is not None:
                break
        return match_patch_key
