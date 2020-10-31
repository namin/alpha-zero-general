def import_non_local(name, custom_name=None):
    import imp, sys

    custom_name = custom_name or name

    f, pathname, desc = imp.find_module(name, sys.path[2:])
    module = imp.load_module(custom_name, f, pathname, desc)
    #f.close()

    return module

_chess = import_non_local('chess', '_chess') # python-chess library
