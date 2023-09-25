# Copyright (c) OpenMMLab. All rights reserved.
import ast
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union


def record_no_need_remove_node(node: ast.AST,
                               used_module_dict_super: Dict[ast.AST,
                                                            Set[str]]):
    """Recode the node had been use and no need to remove.

    Args:
        node (ast.AST): AST Node..
        used_module_dict_set_super (dict): The dict records the node and a set
        including module names it uses.

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
        >>> used_module_dict_set_super[ast.Assign] = set('nn', 'a')
    """
    # only traverse the body of FunctionDef to ignore the input args
    if isinstance(node, ast.FunctionDef):
        for func_sub_node in node.body:
            for func_sub_sub_node in ast.walk(func_sub_node):
                if isinstance(func_sub_sub_node, ast.Name):
                    used_module_dict_super[node].add(func_sub_sub_node.id)

    # iteratively process the body of ClassDef
    elif isinstance(node, ast.ClassDef):
        for class_sub_node in node.body:
            record_no_need_remove_node(class_sub_node, used_module_dict_super)

    else:
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Name):
                used_module_dict_super[node].add(sub_node.id)


def if_need_remove(node: ast.AST, used_module_dict_super: Dict[ast.AST,
                                                               Set[str]]):
    """Justify if the node should be remove.

    If the node not be use actually, it will be removed.

    Args:
        node (ast.AST): AST Node.
        used_module_dict_super (dict): The dict records the node
            and a set including module names it uses.

    Returns:
        bool: if not be used then return True meaning to be removed,
            else False.
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

    for name_list in used_module_dict_super.values():
        if name in name_list:
            return False

    return True


def is_in_top_ast_tree(node: ast.AST,
                       assign_list_top: List[ast.Assign],
                       try_list_top: List[ast.Try],
                       if_list_top: List[ast.If],
                       top_cls_and_func_node_name_list: List[str] = []):
    """Justify if the module name already exists in ``top_ast_tree``.

    Args:
        node (ast.AST): AST Node.
        assign_list_top (list): ast.Assign node in ``top_ast_tree``.
        try_list_top (list): ast.Try node in ``top_ast_tree``.
        if_list_top (list): ast.If node in ``top_ast_tree``.
        top_cls_and_func_node_name_list (list): class's or
            function's name in ``top_ast_tree``

    Returns:
        bool: if the module name already exists in ``top_ast_tree`` return
            True, else False.
    """
    if isinstance(node, ast.Assign):
        for _assign in assign_list_top:
            if ast.dump(_assign) == ast.dump(node):
                return True

    elif isinstance(node, ast.Try):
        for _try in try_list_top:
            if ast.dump(_try) == ast.dump(node):
                return True

    elif isinstance(node, ast.If):
        for _if in if_list_top:
            if ast.dump(_if) == ast.dump(node):
                return True

    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
        if node.name in top_cls_and_func_node_name_list:
            return True

    return False


def find_local_import(node: Union[ast.FunctionDef, ast.Assign],
                      used_module_dict_super: Dict[ast.AST, Set[str]],
                      extra_importfrom_dict_super: Dict[ast.ImportFrom,
                                                        List[ast.alias]],
                      extra_import_list_super: List[ast.alias],
                      need_importfrom_nodes: Optional[set] = None,
                      need_import_alias_asname: Optional[set] = None,
                      need_import_alias: Optional[set] = None):
    """Find the needed Import and ImportFrom of the node.

    Args:
        node (ast.FunctionDef or ast.Assign)
        used_module_dict_super (dict): The dict records the node and
            a set including module names it uses.
        extra_importfrom_dict_super (dict): The dict records extra
            ``ast.ImportFrom`` nodes and their modules in the super
            class file.
        extra_import_list_super (list): The list records extra
            ``ast.Import`` nodes' alias in the super class file.
        need_importfrom_nodes (set, optional): Collect the needed
            ``ast.ImportFrom`` nodes from ``extra_importfrom_dict_super``.
            Defaults to None.
        need_import_alias_asname (set, optional): Collect the needed
            ``ast.Import`` asname nodes from ``extra_import_list_super``.
            Defaults to None.
        need_import_alias (set, optional): Collect the needed ``ast.Import``
            nodes from ``extra_import_list_super``. Defaults to None.

    Returns:
        need_importfrom_nodes (set): collect the extra ast.ImportFrom nodes
            needed to be added.
        need_import_alias_asname (set): collect the extra ast.Import nodes
            containing asname alias.
        need_import_alias (set): collect the extra ast.Import nodes
            containing simple alias.
    """
    need_importfrom_nodes = set(
    ) if need_importfrom_nodes is None else need_importfrom_nodes
    need_import_alias_asname = set(
    ) if need_import_alias_asname is None else need_import_alias_asname
    need_import_alias = set(
    ) if need_import_alias is None else need_import_alias

    # get all the used modules' name in specific node
    used_module = used_module_dict_super[node]

    if len(used_module) != 0:

        # record all used ast.ImportFrom nodes
        for import_node, alias_list in extra_importfrom_dict_super.values():
            for module in used_module:
                if module in alias_list:  # type: ignore[operator]
                    need_importfrom_nodes.add(import_node)
                    continue

        # record all used ast.Import nodes
        for alias in extra_import_list_super:
            if alias.asname is not None:
                if alias.asname in used_module:
                    need_import_alias_asname.add(alias)
            else:
                if alias.name in used_module:
                    need_import_alias.add(alias)

    return need_importfrom_nodes, need_import_alias_asname, need_import_alias


def ignore_ast_docstring(node: Union[ast.ClassDef, ast.FunctionDef]):
    """Get the insert key ignoring the docstring.

    Args:
        node (ast.ClassDef or ast.FunctionDef): AST Node.

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


def add_local_import_to_func(node, need_importfrom_nodes: set,
                             need_import_alias_asname: set,
                             need_import_alias: set):
    """Add the needed ast.ImportFrom and ast.Import to ast.Function.

    Args:
        node (ast.FunctionDef or ast.Assign)
        need_importfrom_nodes (set): Includes the needed ``ast.ImportFrom``
            nodes from ``extra_importfrom_dict_super``. Defaults to None.
        need_import_alias_asname (set): Includes the needed ``ast.Import``
            asname nodes from ``extra_import_list_super``. Defaults to None.
        need_import_alias (set): Includes the needed ``ast.Import`` nodes from
            ``extra_import_list_super``. Defaults to None.
    """
    insert_index = ignore_ast_docstring(node)

    for importfrom_node in need_importfrom_nodes:
        node.body.insert(insert_index, importfrom_node)

    if len(need_import_alias) != 0:
        node.body.insert(
            insert_index,
            ast.Import(names=[alias for alias in need_import_alias]))

    for alias in need_import_alias_asname:
        node.body.insert(insert_index, ast.Import(names=[alias]))


def add_local_import_to_class(
        cls_node: ast.ClassDef,
        used_module_dict_super: Dict[ast.AST, Set[str]],
        extra_importfrom_dict_super: Dict[ast.ImportFrom, List[ast.alias]],
        extra_import_list_super: List[ast.alias],
        new_node_begin_index=-9999):
    """Add the needed ast.ImportFrom and ast.Import to ast.Class sub_nodes,
    including ast.Assign and ast.Function.

    Traverse ast.ClassDef node, recode all the used modules of class
    attributes like the ast.Assign, and this needed ast.ImportFrom and
    ast.Import will be add to the top of the cls_node.body. More, for each
    sub functions in this class, we will process them as glabal functions
    by using ``find_local_import`` and ``add_local_import_to_func``.

    Args:
        cls_node (ast.ClassDef)
        used_module_dict_super (dict): The dict records the node and a set
            including module names it uses.
        extra_importfrom_dict_super (dict): The dict records extra
            ``ast.ImportFrom`` nodes and their modules in super class file.
        extra_import_list_super (list): The list records extra ``ast.Import``
            nodes' alias in the super class file.
        new_node_begin_index (int, optional): The begin index of the
            ast.FunctionDef node in ast.ClassDef in order to only process the
            newly added ast.FunctionDef node. Defaults to -9999.
    """
    # for later add all the needed ast.ImportFrom and ast.Import nodes for
    # class attributes
    later_need_importfrom_nodes: Set[ast.ImportFrom] = set()
    later_need_import_alias_asname: Set[ast.alias] = set()
    later_need_import_alias: Set[ast.alias] = set()

    for i, cls_sub_node in enumerate(cls_node.body):
        if isinstance(cls_sub_node, ast.Assign):

            find_local_import(
                node=cls_sub_node,
                used_module_dict_super=used_module_dict_super,
                extra_importfrom_dict_super=extra_importfrom_dict_super,
                extra_import_list_super=extra_import_list_super,
                need_importfrom_nodes=later_need_importfrom_nodes,
                need_import_alias_asname=later_need_import_alias_asname,
                need_import_alias=later_need_import_alias,
            )

        # ``i >= new_node_begin_index`` means only processing those
        # newly added nodes.
        elif isinstance(cls_sub_node,
                        ast.FunctionDef) and i >= new_node_begin_index:
            need_importfrom_alias, need_import_alias_asname, \
                need_import_alias = find_local_import(
                    used_module_dict_super=used_module_dict_super,
                    node=cls_sub_node,
                    extra_importfrom_dict_super=extra_importfrom_dict_super,
                    extra_import_list_super=extra_import_list_super,
                )

            add_local_import_to_func(
                node=cls_sub_node,
                need_importfrom_nodes=need_importfrom_alias,
                need_import_alias_asname=need_import_alias_asname,
                need_import_alias=need_import_alias)

    # add all the needed ast.ImportFrom and ast.Import nodes for
    # class attributes
    add_local_import_to_func(
        node=cls_node,
        need_importfrom_nodes=later_need_importfrom_nodes,
        need_import_alias_asname=later_need_import_alias_asname,
        need_import_alias=later_need_import_alias)


def init_prepare(top_ast_tree: ast.Module, flattened_cls_name: str):
    """Collect the initial information of the ``top_ast_tree``.

    Args:
        top_ast_tree (ast.Module): The top ast tree contains the classes
            directly called, which is  continuelly updated.
        flattened_cls_name (str): The name of the class needed to
            be flattened.

    Returns:
        importfrom_dict_top (dict): The dict contatins the simple alias of
            ast.ImportFrom information of top_ast_tree.
        class_dict_top (dict): The dict contatins top_cls_node information
            which waiting to be flattened.
        import_list_top (list): The dict contatins imported module name
            of top_ast_tree.
        assign_list_top (list): The dict contatins global assign
            of top_ast_tree.
        if_list_top (list): The dict contatins global if
            of top_ast_tree.
        try_list_top (list): The dict contatins global try
            of top_ast_tree.
        importfrom_asname_dict_top (dict): The dict contatins the asname alias
            of ast.ImportFrom information of top_ast_tree.
    """
    # TODO
    importfrom_dict_top = {}
    importfrom_asname_dict_top = {}
    import_list_top = []
    class_dict_top = None
    try_list_top = []
    if_list_top = []
    assign_list_top = []

    # top_ast_tree scope
    for node in top_ast_tree.body:

        # ast.Module -> ast.ImporFrom
        if isinstance(node, ast.ImportFrom):
            if node.names[0].asname is not None:
                importfrom_asname_dict_top[node.module] = [node]
            else:
                # yapf: disable
                importfrom_dict_top[node.module] = [node] + [
                    [alias.name for alias in node.names]  # type: ignore
                ]
                # yapf: enable

        # ast.Module -> ast.Import
        elif isinstance(node, ast.Import):
            import_list_top.extend([
                alias.name if alias.asname is None else alias.asname
                for alias in node.names
            ])

        # ast.Module -> ast.Assign
        elif isinstance(node, ast.Assign):
            assign_list_top.append(node)

        # ast.Module -> ast.Try
        elif isinstance(node, ast.Try):
            if_list_top.append(node)

        # ast.Module -> ast.If
        elif isinstance(node, ast.If):
            try_list_top.append(node)

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

            class_dict_top = {
                'cls_node': node,
                'upper_cls_name': flattened_cls_name,
                'level_cls_name': flattened_cls_name,
                'super_cls_name':
                node.bases[0].id,  # type: ignore[attr-defined]
                'end_idx': len(node.body),
                'sub_func': {},
                'sub_assign': {}
            }

            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    class_dict_top['sub_func'][sub_node.name] = sub_node

                elif isinstance(sub_node, ast.Assign):
                    class_dict_top['sub_assign'][sub_node.targets[
                        0].id] = sub_node  # type: ignore[attr-defined]

    assert class_dict_top is not None, f"The class [{flattened_cls_name}] \
        doesn't exist in the ast tree."

    return importfrom_dict_top, import_list_top, class_dict_top, \
        assign_list_top, try_list_top, if_list_top, importfrom_asname_dict_top


def flatten_model(super_ast_tree: ast.Module, class_dict_top: dict,
                  importfrom_dict_top: dict, import_list_top: list,
                  assign_list_top: list, if_list_top: list, try_list_top: list,
                  importfrom_asname_dict_top: dict):
    """Flatten specific model class.

    This function traverses super_ast_tree and collection information
    comparing with top_cls_node in top_ast_tree.

    Need to process ImportFrom, Import, ClassDef, If, Try, Assign.
    ImportFrom soulution: If the node.module already exist in top_ast_tree, we
    will merge it's alias, but separately deal with asname and simple
    ImportFrom node. Else will be consider extra ImportFrom.
        1. asname alias use ``ast.dump()`` to compare.
        2. simple alias use ``set()``  to get the union set.

    ClassDef solution: The main part. First we get the top_cls_node and
    replace ``super()`` call in it with information in super_cls_node.
    Second, traverse super_ast_tree to get those super class ast.FunctionDef
    and super class ast.Assign needed to add to top_cls_node. We should rename
    the function called by super(). Last, insert all the needed super node
    into top_cls_node and update top_cls_node.bases. Finish class flatten.

    Args:
        super_ast_tree (ast.Module): The super ast tree including the super
            class in the specific flatten class's mro.
        class_dict_top (dict): The dict contatins top_cls_node information.
        importfrom_dict_top (dict): The dict contatins the simple alias of
            ast.ImportFrom information of top_ast_tree.
        import_list_top (list): The dict contatins imported module name
            of top_ast_tree.
        assign_list_top (list): The dict contatins global assign
            of top_ast_tree.
        if_list_top (list): The dict contatins global if
            of top_ast_tree.
        try_list_top (list): The dict contatins global try
            of top_ast_tree.
        importfrom_asname_dict_top (dict): The dict contatins the asname alias
            of ast.ImportFrom information of top_ast_tree.

    Returns:
        used_module_dict_set_super (dict): The dict records the node and a set
        including module names it uses.
        extra_importfrom_dict_super (dict): The dict records extra
            ``ast.ImportFrom`` nodes and their modules in super class file.
        extra_import_list_super (list): The list records extra ``ast.Import``
            nodes' alias in the super class file.
    """

    used_module_dict_set_super: Dict[ast.AST, Set[str]] = defaultdict(set)
    extra_import_list_super: List[ast.alias] = []
    extra_importfrom_dict_super = {}

    # super_ast_tree scope
    for node in super_ast_tree.body:

        # ast.Module -> ast.ImportFrom
        if isinstance(node, ast.ImportFrom):

            # HARD CODE: if ast.alias has asname, we consider it only contains
            # one module
            #
            # Examples:
            #   >>> # common style
            #   >>> for abc import a as A
            #   >>> # not recommonded style
            #   >>> for abc import a as A, B
            if node.names[0].asname is not None:
                if node.module in importfrom_asname_dict_top:
                    top_importfrom_node = importfrom_asname_dict_top.get(
                        node.module)

                    if ast.dump(top_importfrom_node  # type: ignore[arg-type]
                                ) != ast.dump(node):
                        # yapf: disable
                        extra_importfrom_dict_super[node.module] = [node] + [
                            [
                                alias.name
                                if alias.asname is None else alias.asname
                                for alias in node.names
                            ]  # type: ignore
                        ]
                        # yapf: enable

            # only name
            else:
                # the ast.alias import from the same module will be merge into
                # one ast.ImportFrom
                if node.module in importfrom_dict_top:
                    top_importfrom_node, last_names = importfrom_dict_top.get(
                        node.module)  # type: ignore[misc]

                    current_names = [alias.name for alias in node.names]
                    last_names = list(set(last_names + current_names))
                    top_importfrom_node.names = [
                        ast.alias(name=name) for name in last_names
                    ]

                    # NOTE: update the information of top_ast_tree
                    importfrom_dict_top[node.module] = [top_importfrom_node
                                                        ] + [last_names]

                # those don't exist ast.ImportFrom will be later added
                else:
                    # yapf: disable
                    extra_importfrom_dict_super[node.module] = [node] + [
                        [
                            alias.name
                            if alias.asname is None else alias.asname
                            for alias in node.names
                        ]  # type: ignore
                    ]
                    # yapf: enable

        # ast.Module -> ast.Import
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    if alias.asname not in import_list_top:
                        extra_import_list_super.append(alias)
                else:
                    if alias.name not in import_list_top:
                        extra_import_list_super.append(alias)

        # ast.Module -> ast.Try / ast.Assign / ast.If
        elif (isinstance(node, ast.Try) or isinstance(node, ast.Assign) or
              isinstance(node, ast.If)) \
            and not is_in_top_ast_tree(node, assign_list_top, try_list_top,
                                       if_list_top):

            record_no_need_remove_node(node, used_module_dict_set_super)

        # ast.Module -> ast.ClassDef
        elif isinstance(node, ast.ClassDef
                        ) and node.name == class_dict_top['super_cls_name']:

            # get the specific flattened class node in the top_ast_tree
            top_cls_node = class_dict_top['cls_node']

            # process super, including below  circumstances:
            #   class A(B) and class B(C)
            #   1. super().xxx(): directly replace to self.B_xxx()
            #   2. super(A, self).xxx(): directly replace to self.B_xxx()
            #   3. super(B, self).xxx(): waiting the level_cls_name=B, then
            #      replace to self.C_xxx()
            # HARD CODE: if B doesn't exist self.xxx(), it will not deal with
            # super(A, self).xxx() until the ``postprocess_super()`` will
            # remove all the args in ``super(args)``, then change to
            # ``super()``. In another word, if super doesn't replace in the
            # correct level, it will be turn to use the root super
            # class' method.
            super_func = []
            for top_cls_sub_node in ast.walk(top_cls_node):

                if isinstance(top_cls_sub_node, ast.FunctionDef):

                    for top_cls_sub_sub_node in ast.walk(top_cls_sub_node):

                        if isinstance(top_cls_sub_sub_node, ast.Attribute) \
                            and hasattr(top_cls_sub_sub_node, 'value') \
                            and isinstance(top_cls_sub_sub_node.value, ast.Call) \
                            and isinstance(top_cls_sub_sub_node.value.func, ast.Name) \
                                and top_cls_sub_sub_node.value.func.id == 'super':  # noqa: E501
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
                            if len(
                                    top_cls_sub_sub_node.value.args
                            ) != 0 and top_cls_sub_sub_node.value.args[  # type: ignore[attr-defined]  # noqa: E501
                                    0].id != class_dict_top['level_cls_name']:
                                continue

                            # search and justify if the .xxx() function in the
                            # super node
                            for super_cls_sub_node in node.body:
                                if isinstance(
                                        super_cls_sub_node, ast.FunctionDef
                                ) and top_cls_sub_sub_node.attr == \
                                        super_cls_sub_node.name:
                                    super_func.append(
                                        top_cls_sub_sub_node.attr)
                                    top_cls_sub_sub_node.value = \
                                        top_cls_sub_sub_node.value.func
                                    top_cls_sub_sub_node.value.id = 'self'
                                    top_cls_sub_sub_node.value.args = [  # type: ignore[attr-defined]  # noqa: E501
                                    ]
                                    top_cls_sub_sub_node.attr = node.name + \
                                        '_' + top_cls_sub_sub_node.attr
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
                        class_dict_top['sub_func'][
                            super_cls_sub_node.name] = super_cls_sub_node
                        record_no_need_remove_node(super_cls_sub_node,
                                                   used_module_dict_set_super)

                    # the function don't exist in top class node will be
                    # directly imported
                    elif super_cls_sub_node.name not in class_dict_top[
                            'sub_func']:
                        func_need_append.append(super_cls_sub_node)

                        # NOTE: update the information of top_ast_tree
                        class_dict_top['sub_func'][
                            super_cls_sub_node.name] = super_cls_sub_node
                        record_no_need_remove_node(super_cls_sub_node,
                                                   used_module_dict_set_super)

                # ast.Module -> ast.ClassDef -> ast.Assign
                elif isinstance(super_cls_sub_node, ast.Assign):
                    add_flag = True

                    for name in class_dict_top['sub_assign'].keys():
                        if name == super_cls_sub_node.targets[
                                0].id:  # type: ignore[attr-defined]
                            add_flag = False

                    if add_flag:
                        assign_need_append.append(super_cls_sub_node)

                        # NOTE: update the information of top_ast_tree
                        class_dict_top['end_idx'] += 1
                        class_dict_top['sub_assign'][
                            super_cls_sub_node.
                            targets[0].  # type: ignore[attr-defined]
                            id] = super_cls_sub_node
                        record_no_need_remove_node(super_cls_sub_node,
                                                   used_module_dict_set_super)

            # add all the needed ast.ClassDef -> ast.FunctionDef and
            # ast.ClassDef -> ast.Assign to top_cls_node
            if len(assign_need_append) != 0:
                insert_idx = ignore_ast_docstring(top_cls_node)

                assign_need_append.reverse()
                for assign in assign_need_append:
                    top_cls_node.body.insert(insert_idx, assign)

            top_cls_node.body.extend(func_need_append)

            # complete this level flatten, change the super class of
            # top_cls_node
            top_cls_node.bases = node.bases

            # NOTE: update the information of top_ast_tree
            class_dict_top['level_cls_name'] = node.name
            # HARD CODE: useless, only for preventing error when ``nn.xxx``
            # as the last super class
            class_dict_top['super_cls_name'] = node.bases[0].id if isinstance(
                node.bases[0], ast.Name) else node.bases[0]

    return used_module_dict_set_super, extra_import_list_super, \
        extra_importfrom_dict_super


def get_len(used_module_dict_super: Dict[ast.AST, Set[str]]):
    """Get the sum of used modules.

    Args:
        used_module_dict_super (dict): The dict records the node and a set
            including module names it uses.

    Returns:
        int: The sum of used modules.
    """
    len_sum = 0
    for name_list in used_module_dict_super.values():
        len_sum += len(name_list)

    return len_sum


def postprocess_top_ast_tree(
    super_ast_tree: ast.Module,
    top_ast_tree: ast.Module,
    used_module_dict_super: Dict[ast.AST, Set[str]],
    extra_importfrom_dict_super: Dict[ast.ImportFrom, List[ast.alias]],
    extra_import_list_super: List[ast.alias],
    class_dict_top: dict,
    assign_list_top: List[ast.Assign],
    try_list_top: List[ast.Try],
    if_list_top: List[ast.If],
    importfrom_dict_top: dict,
    import_list_top: list,
):
    """Postprocess top_ast_tree with the information collected by traversing
    super_ast_tree.

    This function finishes:
        1. get all the nodes needed by top_ast_tree and
            exist in super_ast_tree
        2. add as local import for the new add function from super_ast_tree
            preventing from covering by the same name modules on the top.
        3. add extra Import/ImportFrom of super_ast_tree to
            the top of top_ast_tree

    Args:
        super_ast_tree (ast.Module): The super ast tree including the super
            class in the specific flatten class's mro.
        top_ast_tree (ast.Module): The top ast tree contains the classes
            directly called, which is  continuelly updated.
        used_module_dict_super (dict): The dict records the node and a set
            including module names it uses.
        extra_importfrom_dict_super (dict): The dict records extra
            ``ast.ImportFrom`` nodes and their modules in super class file.
        extra_import_list_super (list): The list records extra ``ast.Import``
            nodes' alias in the super class file.
        class_dict_top (dict): The dict contatins top_cls_node information.
        assign_list_top (list): The dict contatins global assign
            of top_ast_tree.
        if_list_top (list): The dict contatins global if
            of top_ast_tree.
        try_list_top (list): The dict contatins global try
            of top_ast_tree.
        importfrom_dict_top (dict): The dict contatins the simple alias of
            ast.ImportFrom information of top_ast_tree.
        import_list_top (list): The dict contatins imported module name
            of top_ast_tree.
    """

    # record all the imported module
    imported_module_name_upper = set()
    for importfrom_node, alias_list in importfrom_dict_top.values(
    ):  # [node, [str]]
        for alias in alias_list:
            imported_module_name_upper.add(alias)

    for name in import_list_top:
        imported_module_name_upper.add(name)

    # HARD CODE: there will be a situation that the super class and the sub
    # class exist in the same file, the super class should
    imported_module_name_upper.discard(class_dict_top['super_cls_name'])

    # find the needed ast.ClassDef or ast.FunctionDef in super_ast_tree
    need_append_node_name: Set[str] = set()
    if get_len(used_module_dict_super) != 0:

        while True:
            origin_len = get_len(used_module_dict_super)

            # super_ast_tree scope
            for node in super_ast_tree.body:
                if (isinstance(node, ast.ClassDef)
                    or isinstance(node, ast.FunctionDef)) \
                    and not if_need_remove(node, used_module_dict_super) \
                        and node.name not in imported_module_name_upper:

                    need_append_node_name.add(node.name)
                    record_no_need_remove_node(node, used_module_dict_super)

            # if there is no longer extra new module, then search break
            if get_len(used_module_dict_super) == origin_len:
                break

    # record insert_idx and classes and functions' name in top_ast_tree
    insert_idx = 0
    top_cls_func_node_name_list = []
    for top_cls_node in top_ast_tree.body:

        if isinstance(top_cls_node, ast.Import) or isinstance(
                top_cls_node, ast.ImportFrom):
            insert_idx += 1
        elif isinstance(top_cls_node, ast.FunctionDef) or isinstance(
                top_cls_node, ast.ClassDef):
            top_cls_func_node_name_list.append(top_cls_node.name)

    # super_ast_tree scope
    for node in super_ast_tree.body:

        # ast.Module -> ast.Try / ast.Assign / ast.If
        if (isinstance(node, ast.Try) or isinstance(node, ast.Assign)
            or isinstance(node, ast.If)) \
            and not is_in_top_ast_tree(node, assign_list_top,
                                       try_list_top, if_list_top,
                                       top_cls_func_node_name_list):

            # NOTE: postprocess top_ast_tree
            top_ast_tree.body.insert(insert_idx, node)
            insert_idx += 1

            # NOTE: update the information of top_ast_tree
            if isinstance(node, ast.Try):
                try_list_top.append(node)
            elif isinstance(node, ast.Assign):
                assign_list_top.append(node)
            elif isinstance(node, ast.If):
                if_list_top.append(node)

        elif not if_need_remove(node, used_module_dict_super) \
            and not is_in_top_ast_tree(node, assign_list_top,
                                       try_list_top, if_list_top,
                                       top_cls_func_node_name_list) \
                and node.name in need_append_node_name:  # type: ignore[attr-defined]  # noqa: E501

            # ast.Module -> ast.FunctionDef
            if isinstance(node, ast.FunctionDef):

                need_importfrom_alias, need_import_alias_asname, \
                    need_import_alias = find_local_import(
                        used_module_dict_super=used_module_dict_super,
                        node=node,
                        extra_importfrom_dict_super=extra_importfrom_dict_super,  # noqa: E501
                        extra_import_list_super=extra_import_list_super)

                add_local_import_to_func(
                    node=node,
                    need_importfrom_nodes=need_importfrom_alias,
                    need_import_alias_asname=need_import_alias_asname,
                    need_import_alias=need_import_alias)

                # NOTE: postprocess top_ast_tree
                top_ast_tree.body.insert(insert_idx, node)
                insert_idx += 1

            # ast.Module -> ast.ClassDef
            elif isinstance(node, ast.ClassDef):
                add_local_import_to_class(
                    cls_node=node,
                    used_module_dict_super=used_module_dict_super,
                    extra_importfrom_dict_super=extra_importfrom_dict_super,
                    extra_import_list_super=extra_import_list_super,
                )

                # NOTE: postprocess top_ast_tree
                top_ast_tree.body.insert(insert_idx, node)
                insert_idx += 1

    # the newly add functions in top_cls_node also should add local import
    top_cls_node = class_dict_top['cls_node']
    add_local_import_to_class(
        cls_node=top_cls_node,  # type: ignore[arg-type]
        used_module_dict_super=used_module_dict_super,
        extra_importfrom_dict_super=extra_importfrom_dict_super,
        extra_import_list_super=extra_import_list_super,
        new_node_begin_index=class_dict_top['end_idx'])

    # update the end_idx for next time postprocess
    class_dict_top['end_idx'] = len(
        top_cls_node.body)  # type: ignore[attr-defined]

    # postprocess global import
    # all the extra import will be inserted to the top of the top_ast_tree
    need_importfrom_alias = set()
    need_import_alias_asname = set()
    need_import_alias = set()

    for module_name, (sub_node,
                      name_list) in extra_importfrom_dict_super.items():
        need_importfrom_alias.add(sub_node)
        importfrom_dict_top[module_name] = [sub_node] + [name_list]

    for alias in extra_import_list_super:
        if alias.asname is not None:
            need_import_alias_asname.add(alias)
            import_list_top.append(alias.asname)
        else:
            need_import_alias.add(alias)
            import_list_top.append(alias.name)

    add_local_import_to_func(
        top_ast_tree,
        need_importfrom_nodes=need_importfrom_alias,
        need_import_alias_asname=need_import_alias_asname,
        need_import_alias=need_import_alias,
    )


def postprocess_super(ast_tree: ast.Module):
    """Postprocess those don't successfully process ``super()`` call.

    This is a hard code.
    All the ``super(args)`` with args will be remove and turn to ``super()``.

    Args:
        ast_tree (ast.Module)
    """
    for node in ast_tree.body:
        if isinstance(node, ast.ClassDef):
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.FunctionDef):
                    for sub_sub_node in ast.walk(sub_node):
                        if isinstance(sub_sub_node, ast.Attribute) \
                            and hasattr(sub_sub_node, 'value') \
                            and isinstance(sub_sub_node.value, ast.Call) \
                            and isinstance(sub_sub_node.value.func,
                                           ast.Name) \
                                and sub_sub_node.value.func.id == 'super':
                            sub_sub_node.value.args = []
