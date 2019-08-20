import astor
import ast
from itertools import zip_longest


def alphabetize_imports(module_object):
    import_object_list = []
    import_from_object_list = []
    other_list = []

    module_parse(module_object, import_object_list, import_from_object_list, other_list)
    import_object_list.sort(key=lambda x: x.names[0].name)
    import_from_object_list.sort(key=lambda x: x.module)
    new_body = list(import_object_list + import_from_object_list + other_list)
    new_module = replace_module_body(module_object, new_body)
    return new_module


def correct_docstring(module_object):
    import_object_list = []
    import_from_object_list = []
    other_list = []

    import_object_list, import_from_object_list, other_list = module_parse(
        module_object, import_object_list, import_from_object_list, other_list
    )

    for i in other_list:
        if ast.get_docstring(i) is None:
            init_docstring = create_class_docstring(i)
            expr = ast.parse(init_docstring)
            expr = expr.body[0]
            new_body = [expr] + i.body
            new_module = replace_module_body(i, new_body)
        else:
            pass

    new_body = list(import_object_list + import_from_object_list + other_list)
    new_module = replace_module_body(module_object, new_body)
    return new_module


def replace_module_body(module_object, new_body):
    module_object.body = new_body
    #        else:
    #            raise Exception(
    #                "Module Body is a strange shape, please check your file structure for errors"
    #            )
    return module_object


def module_parse(
    module_object, import_object_list, import_from_object_list, other_list
):
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



def create_class_docstring(ast_object):
    for i in ast_object.body:
        if i.name == "__init__":
            init_doc = create_init_doc(i)
    return init_doc


def create_init_doc_parameter(arg_and_default, init_doc):
    if arg_and_default[1] is not None:
        inferred_type = type(arg_and_default[1])
    else:
        inferred_type = "#TYPE"
    f = f"    {arg_and_default[0]} : {inferred_type}, default = {arg_and_default[1]}\n         #TODO description\n\n"
    return f


def create_init_doc(function_def_object):
    init_doc = '"""#TODO add an overview of the function\n\n    Parameters\n    ----------\n\n'
    args = [i.arg for i in function_def_object.args.args]
    defaults = list(
        [get_function_arg_default(i) for i in function_def_object.args.defaults]
    )
    args.reverse()
    defaults.reverse()
    args_and_defaults = list(zip_longest(args, defaults))
    args_and_defaults.reverse()

    for i in args_and_defaults:
        init_doc += create_init_doc_parameter(i, init_doc)
    init_doc += "    Attributes\n    ----------\n\n"
    init_doc += "    Examples\n    --------\n\n"
    init_doc += "    See also\n    --------\n\n"
    init_doc += '    References\n    ----------\n\n    """'
    return init_doc


"""Testing"""
if __name__ == "__main__":
    a = astor.code_to_ast.parse_file("test.py")
    astor.strip_tree(a)
    z = alphabetize_imports(a)
    y = correct_docstring(z)
    y = ast.fix_missing_locations(y)
    print(astor.to_source(y))
