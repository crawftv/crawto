import astor
import ast
from itertools import zip_longest
import os
import sys
import re

def alphabetize_imports(module_object):
    import_object_list, import_from_object_list, other_list = module_parse(
        module_object
    )
    import_object_list.sort(key=lambda x: x.names[0].name)
    import_from_object_list.sort(key=lambda x: x.module)
    new_body = list(import_object_list + import_from_object_list + other_list)
    new_module = replace_module_body(module_object, new_body)
    return new_module


def correct_class_docstring(module_object):
    import_object_list, import_from_object_list, other_list = module_parse(
        module_object
    )

    for i in other_list:
        if type(i) is ast.ClassDef:
            if ast.get_docstring(i) is None:
                for j in i.body:
                    if type(j) is ast.FunctionDef:
                        if check_dunder(j.name):
                            class_docstring = create_class_docstring(j, i)
                            expr = ast.parse(class_docstring)
                            expr = expr.body[0]
                            new_class_doc = [expr]
                i.body = new_class_doc + i.body
    for i in other_list:
        if type(i) is ast.ClassDef:
            for j in i.body:
                if type(j) is ast.FunctionDef:
                    if check_dunder(j.name) is False:
                        if ast.get_docstring(j) is None:
                            function_docstring = create_function_docstring(j)
                            expr = ast.parse(function_docstring)
                            expr = expr.body[0]
                            new_body = [expr] + j.body
                            new_module = replace_module_body(j, new_body)
                        else:
                            pass
        elif type(i) is ast.FunctionDef:
            if ast.get_docstring(i) is None:
                function_docstring = create_function_docstring(i)
                expr = ast.parse(function_docstring)
                expr = expr.body[0]
                new_body = [expr] + i.body
                new_module = replace_module_body(i, new_body)
            else:
                pass

    new_body = list(import_object_list + import_from_object_list + other_list)
    new_module = replace_module_body(module_object, new_body)
    return new_module


def replace_module_body(ast_object, new_body):
    ast_object.body = new_body
    #        else:
    #            raise Exception(
    #                "Module Body is a strange shape, please check your file structure for errors"
    #            )
    return ast_object


def module_parse(module_object):
    import_object_list = []
    import_from_object_list = []
    other_list = []
    for i in module_object.body:
        import_selector(i, import_object_list, import_from_object_list, other_list)
    return import_object_list, import_from_object_list, other_list


def import_selector(
    parsed_object, import_object_list, import_from_object_list, other_list
):
    t = type(parsed_object)

    if t == ast.Import:
        import_object_list.append(parsed_object)
    elif t == ast.ImportFrom:
        import_from_object_list.append(parsed_object)
    else:
        other_list.append(parsed_object)


def get_function_arg_default(ast_object):
    if type(ast_object) is ast.Num:
        return ast_object.n
    elif type(ast_object) is ast.Str:
        return ast_object.s
    else:
        return "error"


def create_function_parameter_doc(arg_and_default, init_doc):
    if arg_and_default[1] is not None:
        inferred_type = type(arg_and_default[1])
    else:
        inferred_type = "#TYPE"
    f = f"    {arg_and_default[0]} : {inferred_type}, default = {arg_and_default[1]}\n        #TODO description\n\n"
    return f


def create_class_docstring(function_def_object, ast_class_def_object):
    class_doc = (
        '"""#TODO add an overview of the function\n\n    Parameters\n    ----------\n'
    )
    args = [i.arg for i in function_def_object.args.args]
    defaults = list(
        [get_function_arg_default(i) for i in function_def_object.args.defaults]
    )
    args.reverse()
    defaults.reverse()
    args_and_defaults = list(zip_longest(args, defaults))
    args_and_defaults.reverse()

    for i in args_and_defaults:
        class_doc += create_function_parameter_doc(i, class_doc)
    class_doc += "\n    Attributes\n    ----------\n\n"
    class_doc += create_class_attributes_doc(ast_class_def_object)
    class_doc += "\n    Examples\n    --------\n\n"
    class_doc += create_class_examples_doc(ast_class_def_object)
    class_doc += "\n    See also\n    --------\n\n"
    class_doc += '\n    References\n    ----------\n\n    """'
    return class_doc


def create_class_examples_import():
    dirname, file = os.path.split(os.path.abspath(__file__))
    dirname = dirname.split(os.sep)[-1]
    import_statement = f"    >>>from {dirname} import {file}\n"
    return import_statement


def create_class_examples_doc(ast_class_def_object):
    example_doc = ""
    example_doc += create_class_examples_import()
    for i in ast_class_def_object.body:
        if type(i) is ast.FunctionDef:
            if i.name is "__init__":
                example_class_instance = "    >>>" + ast_class_def_object.name
                example_class_parameters = [
                    f.arg for f in i.args.args if f.arg is not "self"
                ]
                example_class_parameters = ", ".join(example_class_parameters)
                example_class_instance += f"({example_class_parameters})\n"
                example_doc += example_class_instance
            elif i.name is not "__init__":
                args = [j.arg for j in i.args.args if j.arg is not "self"]
                args_string = ", ".join(args)
                f = f"    >>>{i.name}({args_string})\n    #TODO return value goes here\n"
                example_doc += f
    return example_doc


def search_for_attributes(ast_module_object, attributes=[]):
    def recursive_search_for_attributes(ast_object, attributes):
        if "body" in dir(ast_object):
            for i in ast_object.body:
                if type(i) is ast.FunctionDef:
                    if i.name is "__init__":
                        pass
                elif type(i) is ast.Assign:
                    attributes.append(i)
                else:
                    search_for_attributes(i, attributes)

        return attributes

    l = recursive_search_for_attributes(ast_module_object, attributes)
    ll = [
        {"id": i.targets[0].value.id, "attr": i.targets[0].attr}
        for i in l
        if type(i.targets[0]) is ast.Attribute
    ]
    return ll


def create_class_attributes_doc(ast_module_object):
    attributes = search_for_attributes(ast_module_object)
    attributes_doc = ""
    for i in attributes:
        f = f'    {i["attr"]} : #TYPE\n        #TODO Description\n'
        attributes_doc += f
    return attributes_doc


def create_function_docstring(ast_function_object):
    function_doc = (
        '"""#TODO add an overview of the function\n\n    Parameters\n    ----------\n\n'
    )
    args = [i.arg for i in ast_function_object.args.args]
    defaults = list(
        [get_function_arg_default(i) for i in ast_function_object.args.defaults]
    )
    args.reverse()
    defaults.reverse()
    args_and_defaults = list(zip_longest(args, defaults))
    args_and_defaults.reverse()

    for i in args_and_defaults:
        function_doc += create_function_parameter_doc(i, function_doc)
    function_doc += "    Returns\n    -------\n"
    returns_list = [
        i.value.id for i in ast_function_object.body if type(i) is ast.Return
    ]
    for i in returns_list:
        function_doc += create_returns_doc(i, function_doc)
    function_doc += '    """'
    return function_doc


def create_returns_doc(returns_list_item, function_doc):
    f = f"    {returns_list_item} :  #TYPE\n       #TODO Description\n"
    return f


def check_dunder(function_name):
    if re.fullmatch(r"__[a-z]+__", function_name) is not None:
        return True
    else:
        return False


"""Testing"""
if __name__ == "__main__":
    a = astor.code_to_ast.parse_file("test.py")
    astor.strip_tree(a)
    z = alphabetize_imports(a)
    y = correct_class_docstring(z)
    y = ast.fix_missing_locations(y)
    y = astor.to_source(y)
    print(y)
    # new_file = open("fixed_test.py","w")
    # new_file.write(y)
    # new_file.close()
