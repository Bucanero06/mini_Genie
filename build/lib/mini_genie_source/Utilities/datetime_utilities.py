#!/usr/bin/env python3

import dateparser
import pandas as pd

from vectorbtpro import _typing as tp

PandasDatetimeIndex = (pd.DatetimeIndex, pd.PeriodIndex)


def try_to_datetime_index(index: tp.IndexLike, **kwargs) -> tp.Index:
    """Try converting an index to a datetime index.
    Keyword arguments are passed to `pd.to_datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            try:
                index = pd.to_datetime(index, **kwargs)
                index = [index]
            except Exception as e:
                if datetime_cfg["parse_index"]:
                    try:
                        parsed_index = dateparser.parse(index)
                        if parsed_index is None:
                            raise Exception
                        index = pd.to_datetime(parsed_index, **kwargs)
                        index = [index]
                    except Exception as e2:
                        pass
        try:
            index = pd.Index(index)
        except Exception as e:
            index = pd.Index([index])
    if isinstance(index, pd.DatetimeIndex):
        return index
    if index.dtype == object:
        try:
            return pd.to_datetime(index, **kwargs)
        except Exception as e:
            if datetime_cfg["parse_index"]:
                try:
                    def _parse(x):
                        _parsed_index = dateparser.parse(x)
                        if _parsed_index is None:
                            raise Exception
                        return _parsed_index

                    return pd.to_datetime(index.map(_parse), **kwargs)
                except Exception as e2:
                    pass
    return index
