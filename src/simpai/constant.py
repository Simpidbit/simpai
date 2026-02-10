import platform

def get_SEP():
    if platform.system() == 'Darwin':
        return '/'
    elif platform.system() == 'Linux':
        return '/'
    elif platform.system() == 'Windows':
        return '\\'
SEP = get_SEP()
