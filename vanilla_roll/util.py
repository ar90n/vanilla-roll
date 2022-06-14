from importlib import import_module


def has_module(module_name: str) -> bool:
    try:
        import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False
