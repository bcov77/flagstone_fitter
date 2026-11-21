"""Atomic, versioned JSON checkpoint with state integrity verification."""
import hashlib
import hmac
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timezone
import numpy as np
import shapely
from .phase1 import _layout_data, _layout_from_data

_BIT_GENERATORS = {cls.__name__: cls for cls in
                   (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
                    np.random.Philox, np.random.SFC64)}


def _json_rng_state(value):
    """Convert NumPy state arrays and scalars to portable JSON values."""
    if isinstance(value, dict):
        return {key: _json_rng_state(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_rng_state(item) for item in value]
    return value


def _digest(data):
    """Canonical JSON SHA-256 for accidental corruption detection, not authenticity."""
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _geometry_digest(data):
    geometry = {'units': data['units'], 'patio': data['patio'],
                'stones': [{k: stone[k] for k in ('id', 'outline') if k in stone}
                           for stone in data['stones']]}
    return _digest(geometry)


class _GeometryDigestCache:
    """Reuse the fingerprint of the private, immutable geometry loaded for a run."""
    def __init__(self, layout):
        self.patio, self.shapes, self.ids, self.units = layout.patio, layout.shapes, layout.ids, layout.units
        self.digest = None
        self.enabled = True

    def get(self, layout, data):
        if (layout.patio is not self.patio or layout.shapes is not self.shapes or
                layout.ids is not self.ids or layout.units != self.units):
            self.enabled = False
        if not self.enabled:
            return _geometry_digest(data)
        if self.digest is None:
            self.digest = _geometry_digest(data)
        return self.digest


def save_checkpoint(layout, path, metadata=None, *, _geometry_cache=None):
    """Atomically replace path after writing and syncing a complete checkpoint."""
    generator_name = layout.rng.bit_generator.__class__.__name__
    if generator_name not in _BIT_GENERATORS or type(layout.rng.bit_generator) is not _BIT_GENERATORS[generator_name]:
        raise ValueError('Unsupported random generator for checkpoint')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        data = _layout_data(layout)
        data['geometry_sha256'] = (_geometry_digest(data) if _geometry_cache is None
                                   else _geometry_cache.get(layout, data))
        data['checkpoint_version'] = 2
        data['rng_state'] = _json_rng_state(layout.rng.bit_generator.state)
        data['metadata'] = {} if metadata is None else dict(metadata)
        data['metadata'].setdefault('run_id', uuid.uuid4().hex)
        data['run_id'] = data['metadata']['run_id']
        data['saved_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            package_version = importlib.metadata.version('flagstone-patio')
        except importlib.metadata.PackageNotFoundError:
            package_version = 'unknown'
        data['software'] = {'flagstone': package_version, 'python': platform.python_version(), 'numpy': np.__version__, 'shapely': shapely.__version__}
        data['integrity'] = {'algorithm': 'sha256', 'digest': _digest(data)}
        with open(temp, 'w') as f:
            f.write(json.dumps(data, separators=(',', ':'), allow_nan=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path):
    """Read v1 or verify v2, then reconstruct the exact parsed content."""
    data = json.loads(Path(path).read_text())
    version = data.get('checkpoint_version')
    if version not in (1, 2):
        raise ValueError('Unsupported checkpoint version')
    if version == 2:
        integrity = data.get('integrity')
        if not isinstance(integrity, dict) or integrity.get('algorithm') != 'sha256' or not isinstance(integrity.get('digest'), str):
            raise ValueError('Checkpoint integrity metadata is missing or unsupported')
        unsigned = dict(data)
        unsigned.pop('integrity')
        if not hmac.compare_digest(integrity['digest'], _digest(unsigned)):
            raise ValueError('Checkpoint integrity check failed')
    layout = _layout_from_data(data)
    generator_name = data['rng_state'].get('bit_generator')
    if generator_name not in _BIT_GENERATORS:
        raise ValueError('Unsupported random generator')
    if type(layout.rng.bit_generator) is not _BIT_GENERATORS[generator_name]:
        layout.rng = np.random.Generator(_BIT_GENERATORS[generator_name]())
    layout.rng.bit_generator.state = data['rng_state']
    return layout, data.get('metadata', {})
