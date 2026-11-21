"""Checkpointed repeated insertion passes, resumable at stone boundaries."""
import time
import uuid
import numpy as np
from .checkpoint import save_checkpoint, load_checkpoint, _GeometryDigestCache
from .phase1 import validate
from .search import coverage, greedy_insert, _VALIDATED_INPUT
from .uncertainty import verify_uncertainty_model


def _order(layout):
    return sorted((sid for i, sid in enumerate(layout.ids) if not layout.active[i]),
                  key=lambda sid: -layout.shapes[sid].area)


def start_iterative(layout, path, *, gap=.01, edge=.01, proposals=20, strategy='edge',
                    relaxation_steps=0, relaxation_step=.02, local_relaxation=False,
                    max_relaxation_penalty=None, mobility_policy='circle',
                    stone_margins=None, patio_margin=0, frozen_ids=None, uncertainty_model=None, tolerance=1e-9):
    """Save an initial valid state and deterministic insertion configuration."""
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    config = {'gap': gap, 'edge': edge, 'tolerance': tolerance, 'proposals': proposals, 'strategy': strategy,
              'relaxation_steps': relaxation_steps, 'relaxation_step': relaxation_step,
              'local_relaxation': local_relaxation, 'max_relaxation_penalty': max_relaxation_penalty,
              'mobility_policy': mobility_policy, 'stone_margins': stone_margins,
              'patio_margin': patio_margin, 'frozen_ids': list(frozen_ids) if frozen_ids is not None else []}
    greedy_insert(layout, stone_ids=(), **config)
    state = {'kind':'iterative-insertion-v1', 'run_id':uuid.uuid4().hex, 'config':config,
             'uncertainty_model':uncertainty_model, 'completed_passes':0, 'next_index':0,
             'order':_order(layout), 'stones_tried':0, 'attempts':0, 'accepted':0,
             'relaxation_attempts':0, 'relaxation_skipped':0, 'relaxation_successes':0,
             'relaxation_failures':0, 'mobile_stone_visits':0,
             'improvements':[{'stones_tried':0, 'coverage':coverage(layout), 'placed':int(layout.active.sum())}]}
    save_checkpoint(layout, path, state)
    return state


def continue_iterative(path, total_passes, *, max_stones=None, max_seconds=None):
    """Continue at stone boundaries, returning a valid layout and saved state.

    Wall-clock and stone limits are checked between stones, not during a stone's
    proposals. An interrupted stone is replayed from the last checkpoint. Timed
    stop points vary with machine load; fixed pass counts are deterministic in
    the same software environment.
    """
    if not isinstance(total_passes, int) or total_passes < 0:
        raise ValueError('total_passes must be a nonnegative integer')
    if max_stones is not None and (not isinstance(max_stones, int) or max_stones < 0):
        raise ValueError('max_stones must be a nonnegative integer or None')
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError('max_seconds must be finite and nonnegative or None')
    layout, state = load_checkpoint(path)
    if state.get('kind') != 'iterative-insertion-v1':
        raise ValueError('Not an iterative insertion checkpoint')
    if total_passes < state['completed_passes']:
        raise ValueError('total_passes precedes checkpoint')
    config = state['config']
    verify_uncertainty_model(layout, state.get('uncertainty_model'), config.get('stone_margins'), config.get('patio_margin', 0))
    if validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9), stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0)):
        raise ValueError('Checkpoint layout is invalid')
    if (state['next_index'] < 0 or (state['order'] and state['next_index'] >= len(state['order'])) or
            (not state['order'] and state['next_index'] != 0) or len(set(state['order'])) != len(state['order'])):
        raise ValueError('Checkpoint insertion schedule is invalid')
    if not set(state['order']).issubset(layout.ids):
        raise ValueError('Checkpoint insertion schedule contains unknown IDs')
    start = time.perf_counter(); tried = 0
    geometry_cache = _GeometryDigestCache(layout)
    while state['completed_passes'] < total_passes and state['order']:
        if max_stones is not None and tried >= max_stones:
            break
        if max_seconds is not None and time.perf_counter()-start >= max_seconds:
            break
        sid = state['order'][state['next_index']]
        if layout.active[layout.ids.index(sid)]:
            raise ValueError('Checkpoint insertion schedule is inconsistent with layout')
        result = greedy_insert(layout, stone_ids=(sid,), _validation_token=_VALIDATED_INPUT, **config)
        tried += 1; state['stones_tried'] += 1; state['next_index'] += 1
        for key in ('attempts', 'accepted', 'relaxation_attempts', 'relaxation_skipped',
                    'relaxation_successes', 'relaxation_failures', 'mobile_stone_visits'):
            state[key] += getattr(result, key)
        if result.accepted:
            state['improvements'].append({'stones_tried':state['stones_tried'],
                                          'coverage':result.coverage, 'placed':int(layout.active.sum())})
        if state['next_index'] == len(state['order']):
            state['completed_passes'] += 1
            state['next_index'] = 0
            state['order'] = _order(layout)
        save_checkpoint(layout, path, state, _geometry_cache=geometry_cache)
    if validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9), stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0)):
        raise ValueError('Final iterative layout is invalid')
    return layout, state
