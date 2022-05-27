import json

import numpy as np
import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger


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


def rm_field_from_record(a, *fieldnames_to_remove):
    return a[[name for name in a.dtype.names if name not in fieldnames_to_remove]]


def _write_header_df(path: object, cols_names: object, compression: object = 'gzip') -> object:
    header_df = pd.DataFrame(columns=cols_names)
    header_df.to_parquet(path, compression=compression)
    return header_df


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
        json.dump(dictionary, json_file, cls=NpEncoder)


def load_dict_from_file(input_file_name):
    with open(input_file_name, 'r') as json_file:
        return json.load(json_file)


def shuffle_it(x, n_times=None):
    from sklearn.utils import shuffle
    if not n_times:
        return shuffle(x)
    else:
        for i in range(n_times):

            x = shuffle(x)
        return x


def create_dir(dir):
    """

    Args:
        dir:

    Returns:
        object:
    """
    from os import path, mkdir
    if not path.exists(dir):
        logger.info(f'Creating directory {dir}')
        mkdir(dir)
    else:
        logger.info(f'Found {dir}')
