#!/usr/bin/env python3
import ast
import inspect
import json
import os
import os.path
import shutil
import sys
from os import listdir
from os.path import exists, isfile

import numpy as np
import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger
from vectorbtpro import register_jitted

from Utilities import _typing as tp


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line.

    Args:
        expr: The expression to evaluate.
        context: The context to evaluate the expression in.

    Returns:
        The result of the last line of the expression.

    Raises:
        SyntaxError: If the expression is not valid Python.
        ValueError: If the expression is not valid Python.
    """
    if context is None:
        context = {}
    tree = ast.parse(inspect.cleandoc(expr))
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, "file", "exec"), context)
    return eval(compile(eval_expr, "file", "eval"), context)


@register_jitted(cache=True)
def compute_np_arrays_mean_nb(values: object) -> object:
    """

    Args:
        values (object):
    """
    if isinstance(values, list):
        result = []
        for value in values:
            result.append(vbt.nb.mean_reduce_nb(value))
        return result
    else:
        return vbt.nb.mean_reduce_nb(values)


@ray.remote
def resample_apply_remote(ask_dataframe, apply_function, timeframe, dropnan=False):
    # noinspection PyUnresolvedReferences
    import vectorbtpro as vbt
    result = ask_dataframe.vbt.resample_apply(timeframe, apply_function)
    if dropnan:
        result = result.dropna()
    return result


def put_objects_list_to_ray(objects):
    return (ray.put(obj) for obj in objects)


def get_objects_list_from_ray(objects):
    return (ray.get(obj) for obj in objects)


def delete_non_filled_elements(a):
    output = []
    for elem in a:
        if elem:
            output.append(elem)
    return np.array(output)


def delete_filled_elements(a):
    output = []
    for elem in a:
        if not elem:
            output.append(elem)
    return np.array(output)


def rm_field_from_record(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


def _write_header_df(path: object, cols_names: object, compression: object = 'gzip') -> object:
    header_df = pd.DataFrame(columns=cols_names)
    header_df.to_parquet(path, compression=compression)
    return header_df


def fetch_non_filled_elements_indexes(a):
    output = []
    for index, elem in enumerate(a):
        if not elem:
            output.append(index)
    return np.array(output)


def fetch_filled_elements_indexes(a):
    output = []
    for index, elem in enumerate(a):
        if elem:
            output.append(index)
    return np.array(output)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_dictionary_to_file(output_file_name, dictionary):
    with open(output_file_name, 'w') as json_file:
        json.dump(dictionary, json_file, cls=NpEncoder, indent=4)


def load_dict_from_file(input_file_name):
    with open(input_file_name, 'r') as json_file:
        return json.load(json_file)


def print_dict(self, optional_object=None):
    """

    Returns:
        object:
    """
    import pprint
    object = self if not optional_object else optional_object
    if hasattr(object, '__dict__'):
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)
    else:
        pprint.pprint(object)


def shuffle_it(x, n_times=None):
    from sklearn.utils import shuffle
    if not n_times:
        return shuffle(x)
    else:
        for i in range(n_times):
            x = shuffle(x)
        return x


def is_empty_dir(_path):
    return exists(_path) and not isfile(_path) and not listdir(_path)


def create_dir(directory):
    if not os.path.exists(directory):
        logger.info(f'Creating directory {directory}')
        os.mkdir(directory)


def create_dirs(*directories):
    logger.info(f'Accepting list of directories {directories}')
    for directory in directories:
        create_dir(directory)


def delete_everything(directory):
    """Deletes everything within a directory.

    Args:
        directory: The directory to delete everything from.

    Returns:
        None
    """

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.info('Failed to delete %s. Reason: %s' % (file_path, e))


def create_or_clean_directory(directory, delete_content=False):
    if not os.path.exists(directory):
        logger.info(f'Creating directory {directory}')
        os.mkdir(directory)
    elif delete_content:
        logger.info(f'Cleaning directory {directory}')
        delete_everything(directory)
    else:
        logger.info(f'Found directory {directory}')


def create_or_clean_directories(*directories, delete_content=False):
    logger.info(f'Accepting list of directories {directories}')
    for directory in directories:
        create_or_clean_directory(directory, delete_content)


def clean_params_record(a):
    indexes_to_keep = np.where(a["trial_id"] != 0)  #
    indexes_to_keep = list(np.insert(indexes_to_keep, 0, 0))  #
    return np.take(a, indexes_to_keep)


def indexes_where_eq_1d(array, value):
    if not type(array).__module__ == np.__name__:
        array = np.array(array)
    #
    return np.where(array == value)[0]


def flip_bool(param):
    """ Takes in a boolean value and returns the opposite """
    if param:
        return False
    else:
        return True


def next_path(path_pattern):
    import os

    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


def Execute(command):
    from subprocess import Popen, PIPE, CalledProcessError
    # >Executes to command line
    with Popen(command, stdout=PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            print(line, end='')  # process line here
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)


def Execute2(command, goodcall, badcall):
    from subprocess import Popen, PIPE
    # >Executes and allows variable prints
    p = Popen(command, stdout=PIPE, shell=True)
    p_status = p.wait()
    if p_status > 0:
        print("Errors found:: ", p_status)
        print(str(badcall))
        exit()
    else:
        print(str(goodcall))


def ExecuteNoWrite(command):
    from subprocess import Popen, PIPE

    # >Executes to command line but does not print
    p = Popen(command, stdout=PIPE, shell=True)
    p_status = p.wait()
    if p_status > 0:
        print("Errors found:: ", p_status)
        sys.exit()
