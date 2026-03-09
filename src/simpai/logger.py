import os
from threading import Thread
from queue import Queue
from time import time as get_timestamp
from datetime import datetime
from platform import system as get_system

def color_print_str(
    s: str,
    ccode: int,
) -> str:
    system = get_system()
    if system == 'Darwin' or system == 'Linux':
        return f'\033[{ccode}m{s}\033[0m'
    elif system == 'Windows':
        return f'`e[{ccode}m{s}`e[0m'
    else:
        return s

def red_print_str(s: str) -> str:
    return color_print_str(s, 31)

def green_print_str(s: str) -> str:
    return color_print_str(s, 32)

def yellow_print_str(s: str) -> str:
    return color_print_str(s, 33)

def blue_print_str(s: str) -> str:
    return color_print_str(s, 34)

'''
-1 - stop
0 - info
1 - warning
2 - error
3 - debug
'''
_SIMPAI_LOGGER_STOP_CODE      = -1
_SIMPAI_LOGGER_INFO_CODE      = 0
_SIMPAI_LOGGER_WARNING_CODE   = 1
_SIMPAI_LOGGER_ERROR_CODE     = 2
_SIMPAI_LOGGER_DEBUG_CODE     = 3
class _Message:
    def __init__(
        self,
        msg: str,
        msg_type: int,
        output: bool,
        timestamp: float | None = None
    ) -> None:
        self.msg: str = msg
        self.msg_type: int = msg_type
        self.output: bool = output
        if timestamp is None:
            self.timestamp: float = get_timestamp()
        else:
            self.timestamp: float = timestamp

_simpai_logger_msg_queue: Queue[_Message] = Queue(maxsize = 0)

def worker() -> None:
    global _simpai_logger_msg_queue
    start_time = get_timestamp()
    start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')

    dir_name = f'logs/{start_time_str}'
    os.makedirs(dir_name, exist_ok = True)
    info_file = open(f'{dir_name}/info.txt', 'at')
    warning_file = open(f'{dir_name}/warning.txt', 'at')
    error_file = open(f'{dir_name}/error.txt', 'at')
    debug_file = open(f'{dir_name}/debug.txt', 'at')

    try:
        while True:
            msgobj = _simpai_logger_msg_queue.get(block = True)
            timestr = datetime.fromtimestamp(msgobj.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            msgstr = f'({timestr}) {msgobj.msg}'
            if msgobj.msg_type == _SIMPAI_LOGGER_INFO_CODE:
                msgstr = '[ INFO    ] ' + msgstr
                if msgobj.output: print(green_print_str(msgstr))
                info_file.write(msgstr + '\n')
                info_file.flush()
            elif msgobj.msg_type == _SIMPAI_LOGGER_WARNING_CODE:
                msgstr = '[ WARNING ] ' + msgstr
                if msgobj.output: print(yellow_print_str(msgstr))
                warning_file.write(msgstr + '\n')
                warning_file.flush()
            elif msgobj.msg_type == _SIMPAI_LOGGER_ERROR_CODE:
                msgstr = '[ ERROR   ] ' + msgstr
                if msgobj.output: print(red_print_str(msgstr))
                error_file.write(msgstr + '\n')
                error_file.flush()
            elif msgobj.msg_type == _SIMPAI_LOGGER_DEBUG_CODE:
                msgstr = '[ DEBUG   ] ' + msgstr
                if msgobj.output: print(blue_print_str(msgstr))
                debug_file.write(msgstr + '\n')
                debug_file.flush()
            elif msgobj.msg_type == _SIMPAI_LOGGER_STOP_CODE:
                info_file.close()
                warning_file.close()
                error_file.close()
                debug_file.close()
                break
    except:
        info_file.close()
        warning_file.close()
        error_file.close()
        debug_file.close()
        raise

_simpai_logger_worker_thread = Thread(target = worker, daemon = True)
_simpai_logger_worker_thread.start()

def wait_for_log_io() -> None:
    _simpai_logger_msg_queue.put(_Message(str(), _SIMPAI_LOGGER_STOP_CODE, False))
    _simpai_logger_worker_thread.join()

def info(
    msg: str,
    output: bool = True,
) -> None | str:
    global _simpai_logger_msg_queue
    _simpai_logger_msg_queue.put(_Message(msg, _SIMPAI_LOGGER_INFO_CODE, output))
    if not output:
        return green_print_str(msg)

def warning(
    msg: str,
    output: bool = True,
) -> None | str:
    global _simpai_logger_msg_queue
    _simpai_logger_msg_queue.put(_Message(msg, _SIMPAI_LOGGER_WARNING_CODE, output))
    if not output:
        return yellow_print_str(msg)

def error(
    msg: str,
    output: bool = True,
) -> None | str:
    global _simpai_logger_msg_queue
    _simpai_logger_msg_queue.put(_Message(msg, _SIMPAI_LOGGER_ERROR_CODE, output))
    if not output:
        return red_print_str(msg)

def debug(
    msg: str,
    output: bool = False,
) -> None | str:
    global _simpai_logger_msg_queue
    _simpai_logger_msg_queue.put(_Message(msg, _SIMPAI_LOGGER_DEBUG_CODE, output))
    if not output:
        return blue_print_str(msg)
