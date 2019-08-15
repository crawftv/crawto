import ast


def alphabetize_imports(filename):
    c = open(filename)
    d = c.read()
    a = ast.parse(d)
    import_object_list = []
    import_from_object_list = []
    other_list = []

    def module_parse(module_object):
        for i in module_object.body:
            import_selector(i, import_object_list, import_from_object_list, other_list)

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

    def replace_module_body(module_object, new_body):
        module_object.body[0].body = new_body
#        else:
#            raise Exception(
#                "Module Body is a strange shape, please check your file structure for errors"
#            )
        return module_object

    module_parse(a)
    import_object_list.sort(key=lambda x: x.names[0].name)
    import_from_object_list.sort(key=lambda x: x.names[0].name)
    new_body = list(import_object_list + import_from_object_list + other_list)
    return replace_module_body(a, new_body)


"""Testing"""
if __name__=="__main__":
    print(alphabetize_imports("test.py"))
