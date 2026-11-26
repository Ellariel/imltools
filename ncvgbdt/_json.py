"""JSON encoding of the objects stored by the estimators.

Plain JSON cannot hold numpy arrays, SHAP explanations or dictionaries with
non-string keys (class labels). They are written as small tagged objects and
restored exactly by :func:`decode`:

* ``{"__ndarray__": [...], "dtype": "float64"}``
* ``{"__dict__": [[key, value], ...]}`` for dictionaries with non-str keys
* ``{"__explanation__": {...}}`` for ``shap.Explanation``

Floats are written with Python's shortest round-trip representation, so a
saved and reloaded value is identical to the original. ``NaN`` and
``Infinity`` are written as such, which Python's ``json`` module reads back but
which is not part of the JSON standard.
"""

import json
from pathlib import Path

import numpy as np
from shap import Explanation


def encode(obj):
    """Convert ``obj`` into something ``json.dump`` accepts."""
    if isinstance(obj, np.ndarray):
        return {'__ndarray__': obj.tolist(), 'dtype': str(obj.dtype)}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Explanation):
        return {'__explanation__': {
            'values': encode(np.asarray(obj.values)),
            'base_values': encode(np.asarray(obj.base_values)),
            'data': encode(np.asarray(obj.data)),
            'feature_names': encode(list(obj.feature_names)),
            'compute_time': encode(obj.compute_time),
        }}
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: encode(v) for k, v in obj.items()}
        return {'__dict__': [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f'Cannot store object of type {type(obj).__name__}.')


def decode(obj):
    """Inverse of :func:`encode`."""
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    if isinstance(obj, dict):
        if '__ndarray__' in obj:
            return np.asarray(obj['__ndarray__'], dtype=obj['dtype'])
        if '__dict__' in obj:
            return {_hashable(decode(k)): decode(v)
                    for k, v in obj['__dict__']}
        if '__explanation__' in obj:
            e = {k: decode(v) for k, v in obj['__explanation__'].items()}
            return Explanation(e['values'],
                               base_values=e['base_values'],
                               data=e['data'],
                               feature_names=e['feature_names'],
                               compute_time=e['compute_time'])
        return {k: decode(v) for k, v in obj.items()}
    return obj


def _hashable(key):
    """Dictionary keys decode to lists if they were tuples."""
    return tuple(key) if isinstance(key, list) else key


def dump(obj, path):
    """Write ``obj`` as JSON file (parent directory must exist)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(encode(obj), f)


def load(path):
    """Read a JSON file written by :func:`dump`."""
    with open(Path(path), 'r', encoding='utf-8') as f:
        return decode(json.load(f))
