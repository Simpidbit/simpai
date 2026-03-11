def is_notebook() -> bool:
    """
    Check if the current environment is a Notebook (Jupyter/Colab/VS Code, etc.).

    Returns:
        bool: True if running in a notebook environment, False otherwise.
    """
    try:
        # get_ipython is a global function automatically injected in IPython/Jupyter environments
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        elif 'google.colab' in str(get_ipython().__class__):
            return True   # Google Colab
        else:
            return False  # Other type
    except NameError:
        return False      # Standard Python Interpreter (no get_ipython)
