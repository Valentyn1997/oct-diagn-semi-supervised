import hashlib

import torch
from typing import Any, Dict, Optional
import logging


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def _to_dot(config: Dict[str, Any], prefix=None) -> Dict[str, Any]:
    result = dict()
    for k, v in config.items():
        if prefix is not None:
            k = f'{prefix}.{k}'
        if isinstance(v, dict):
            v = _to_dot(v, prefix=k)
        elif hasattr(v, '__call__'):
            v = {k: v.__name__}
        else:
            v = {k: v}
        result.update(v)
    return result


def calculate_hash(params):
    # Check, if run with current parameters already exists
    query = ' and '.join(list(map(lambda item: f"params.{item[0]} = '{str(item[1])}'", _to_dot(params).items())))
    logging.info(query)
    return hashlib.md5(query.encode()).hexdigest()