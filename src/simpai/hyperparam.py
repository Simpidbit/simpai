from typing import Any, Callable
from typeguard import typechecked

# True represents locked status
_simpai_hyperparam_set_lock = True
_simpai_hyperparam_read_lock = True
_simpai_hyperparam = dict()

def set_hp_begin() -> None:
    """
    Hyperparameters are not only program inputs but also the cornerstone of system operation.
    Allowing arbitrary modification of hyperparameters at any time and any location in the code
    is the root cause of logical confusion and uncontrollable state.

    Therefore, a strict lifecycle management mechanism is introduced here:
    - Explicit enablement:
        Hyperparameters can only be defined after calling `set_hp_begin()` to open the configuration window.
    - Centralized management:
        Developers are forced to concentrate hyperparameter setting logic in specific code blocks,
        rather than scattering it across various corners of business logic.
    - Immutable:
        Once `set_hp_end()` is called to close the configuration window, hyperparameters enter a "read-only" state,
        ensuring system stability at runtime.
    - Complete visibility:
        Before `set_hp_end()` locks, parameters are in a building state and reading is prohibited.
        This ensures that business logic always retrieves complete and finalized configuration sets.

    Please treat hyperparameter setting with care and maintain the purity and determinism of the configuration.
    """
    global _simpai_hyperparam_set_lock
    global _simpai_hyperparam_read_lock
    _simpai_hyperparam_set_lock = False
    _simpai_hyperparam_read_lock = True

def set_hp_end() -> None:
    """
    Hyperparameters are not only program inputs but also the cornerstone of system operation.
    Allowing arbitrary modification of hyperparameters at any time and any location in the code
    is the root cause of logical confusion and uncontrollable state.

    Therefore, a strict lifecycle management mechanism is introduced here:
    - Explicit enablement:
        Hyperparameters can only be defined after calling `set_hp_begin()` to open the configuration window.
    - Centralized management:
        Developers are forced to concentrate hyperparameter setting logic in specific code blocks,
        rather than scattering it across various corners of business logic.
    - Immutable:
        Once `set_hp_end()` is called to close the configuration window, hyperparameters enter a "read-only" state,
        ensuring system stability at runtime.
    - Complete visibility:
        Before `set_hp_end()` locks, parameters are in a building state and reading is prohibited.
        This ensures that business logic always retrieves complete and finalized configuration sets.

    Please treat hyperparameter setting with care and maintain the purity and determinism of the configuration.
    """
    global _simpai_hyperparam_set_lock
    global _simpai_hyperparam_read_lock
    _simpai_hyperparam_set_lock = True
    _simpai_hyperparam_read_lock = False

@typechecked
def set_hp(
        key: str, 
        value: Any = None
) -> None | Callable:
    """
    set_hp - Set Hyperparameter
    Set hyperparameters or register functions as decorators.

    This function has two usage patterns:
    1. Direct assignment mode: If the `value` parameter is provided, the value will be saved directly.
    2. Decorator mode: If `value` is not provided (i.e., None), a decorator is returned,
       used to register the decorated function into the hyperparameter dictionary.

    Args:
        key (str): The name (key) of the hyperparameter.
        value (Any, optional): The hyperparameter value to store. Defaults to None.

    Returns:
        function or None:
        - If used as a decorator (value is None), returns the decorator function.
        - If used as a regular assignment function, returns None.

    Example:
        # Usage 1: Direct assignment
        set_hp('epoch_num', 100)

        # Usage 2: Decorator
        @set_hp('loss')
        def my_loss_func(a, b):
            return a + b
    """
    if _simpai_hyperparam_set_lock:
        raise RuntimeError('You should call set_hp_begin() before calling set_hp()!')

    if value is None:
        def decorator(func):
            _simpai_hyperparam[key] = func
            return func
        return decorator
    else:
        _simpai_hyperparam[key] = value

@typechecked
def get_hp(key: str) -> Any:
    """
    get_hp - Get Hyperparameter
    Retrieve a registered hyperparameter value or function by key name.

    Args:
        key (str): The name of the hyperparameter to retrieve.

    Returns:
        Any: The corresponding hyperparameter value (can be a number, string, function, etc.).
             Returns None if the key does not exist.

    Example:
        epoch = get_hp('epoch_num')
        loss_func = get_hp('loss')
    """
    if _simpai_hyperparam_read_lock:
        raise RuntimeError('No hyperparameters have been set yet!')

    if key in _simpai_hyperparam:
        return _simpai_hyperparam[key]
    else:
        return None
