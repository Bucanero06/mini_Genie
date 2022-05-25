import ray
import vectorbtpro as vbt


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
