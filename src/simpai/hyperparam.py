# True 代表已上锁
_simpai_hyperparam_set_lock = True
_simpai_hyperparam_read_lock = True
_simpai_hyperparam = dict()

def set_hp_begin():
    """
    超参数不仅是程序的输入，更是系统运行的基石。
    允许在代码的任意位置、任意时刻随意修改超参数，
    是导致逻辑混乱与状态不可控的根源。

    因而，此处引入严格的生命周期管理机制：
    - 显式开启：
        只有在调用 `set_hp_begin()` 开启配置窗口后，才允许定义超参数。
    - 集中管理：
        强制开发者将超参数的设置逻辑集中在特定的代码块中，
        而非散落在业务逻辑的各个角落。
    - 不可篡改：
        一旦调用 `set_hp_end()` 关闭配置窗口，超参数即进入“只读”状态，
        确保系统在运行时的稳定性。
    - 完备可见：
        在 `set_hp_end()` 锁定之前，参数处于正在构建状态，禁止读取。
        这确保了业务逻辑获取到的永远是完整、已定稿的配置集合。

    请审慎对待超参数的设置，保持配置的纯洁性与确定性。
    """
    global _simpai_hyperparam_set_lock
    global _simpai_hyperparam_read_lock
    _simpai_hyperparam_set_lock = False
    _simpai_hyperparam_read_lock = True

def set_hp_end():
    """
    超参数不仅是程序的输入，更是系统运行的基石。
    允许在代码的任意位置、任意时刻随意修改超参数，
    是导致逻辑混乱与状态不可控的根源。

    因而，此处引入严格的生命周期管理机制：
    - 显式开启：
        只有在调用 `set_hp_begin()` 开启配置窗口后，才允许定义超参数。
    - 集中管理：
        强制开发者将超参数的设置逻辑集中在特定的代码块中，
        而非散落在业务逻辑的各个角落。
    - 不可篡改：
        一旦调用 `set_hp_end()` 关闭配置窗口，超参数即进入“只读”状态，
        确保系统在运行时的稳定性。
    - 完备可见：
        在 `set_hp_end()` 锁定之前，参数处于正在构建状态，禁止读取。
        这确保了业务逻辑获取到的永远是完整、已定稿的配置集合。

    请审慎对待超参数的设置，保持配置的纯洁性与确定性。
    """
    global _simpai_hyperparam_set_lock
    global _simpai_hyperparam_read_lock
    _simpai_hyperparam_set_lock = True
    _simpai_hyperparam_read_lock = False

def set_hp(key:str, value = None):
    """
    set_hp - Set Hyperparameter
    设置超参数，或者作为装饰器注册函数。

    该函数有两种用法：
    1. 直接赋值模式：如果提供了 `value` 参数，将直接保存该值。
    2. 装饰器模式：如果未提供 `value` (即为 None)，将返回一个装饰器，
       用于将被装饰的函数注册到超参数字典中。

    Args:
        key (str): 超参数的名称（键）。
        value (Any, optional): 要存储的超参数值。默认为 None。

    Returns:
        function or None: 
        - 如果作为装饰器使用（value is None），返回装饰器函数。
        - 如果作为普通赋值函数使用，返回 None。
    
    Example:
        # 用法 1: 直接赋值
        set_hp('epoch_num', 100)
        
        # 用法 2: 装饰器
        @set_hp('loss')
        def my_loss_func(a, b):
            return a + b
    """
    if _simpai_hyperparam_set_lock:
        raise RuntimeError('调用set_hp()之前应该先调用set_hp_begin()!')

    if value is None:
        def decorator(func):
            _simpai_hyperparam[key] = func
            return func
        return decorator
    else:
        _simpai_hyperparam[key] = value

def get_hp(key:str):
    """
    get_hp - Get Hyperparameter
    根据键名获取已注册的超参数值或函数。

    Args:
        key (str): 要检索的超参数名称。

    Returns:
        Any: 对应的超参数值（可以是数值、字符串、函数等）。
             如果键不存在，则返回 None。
    
    Example:
        epoch = get_hp('epoch_num')
        loss_func = get_hp('loss')
    """
    if _simpai_hyperparam_read_lock:
        raise RuntimeError('还未设置任何超参数!')

    if key in _simpai_hyperparam:
        return _simpai_hyperparam[key]
    else:
        return None
