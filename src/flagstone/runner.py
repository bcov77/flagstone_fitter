"""Durable, round-boundary checkpointing for greedy patch rebuilds."""
import uuid
from .checkpoint import save_checkpoint, load_checkpoint, _GeometryDigestCache
from .phase1 import validate
from .search import coverage, rebuild_search
from .uncertainty import verify_uncertainty_model


def start_rebuild(layout, path, *, gap=.01, edge=.01, patch_size=2, proposals=20,
                  strategy='mixed', relaxation_steps=0, stone_margins=None, patio_margin=0, frozen_ids=None, relaxation_step=.02, uncertainty_model=None, patch_padding=None, max_new_stones=None, local_relaxation=False, max_relaxation_penalty=None, mobility_policy="circle", tolerance=1e-9,
                  secondary_resolution=None, secondary_broad_distance=None, secondary_edge_strip_width=0, restore_removed=False):
    """Create the initial checkpoint before performing any rebuild work."""
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    if validate(layout, gap, edge, tolerance, stone_margins=stone_margins, patio_margin=patio_margin):
        raise ValueError('Initial layout must be valid')
    config = {'gap': gap, 'edge': edge, 'tolerance': tolerance, 'patch_size': patch_size, 'proposals': proposals,
              'strategy': strategy, 'relaxation_steps': relaxation_steps, 'relaxation_step': relaxation_step,
              'patch_padding': patch_padding, 'max_new_stones': max_new_stones, 'local_relaxation': local_relaxation,
              'secondary_resolution': secondary_resolution, 'secondary_broad_distance': secondary_broad_distance,
              'secondary_edge_strip_width': secondary_edge_strip_width,
              'restore_removed': restore_removed,
              'max_relaxation_penalty': max_relaxation_penalty, 'mobility_policy': mobility_policy,
              'stone_margins': stone_margins, 'patio_margin': patio_margin,
              'frozen_ids': list(frozen_ids) if frozen_ids is not None else []}
    # Check arguments through a zero-round call before writing a checkpoint.
    initial_score = rebuild_search(layout, rounds=0, **config)
    state = {'run_id': uuid.uuid4().hex, 'kind': 'rebuild-run-v1', 'config': config, 'rounds': 0, 'attempts': 0,
             'improvements': 0, 'secondary_improvements': 0, 'secondary_score': initial_score.secondary_score, 'relaxation_attempts': 0, 'relaxation_skipped': 0, 'relaxation_successes': 0, 'relaxation_failures': 0, 'mobile_stone_visits': 0, 'best_history': [coverage(layout)], 'uncertainty_model': uncertainty_model}
    save_checkpoint(layout, path, state)
    return state


def continue_rebuild(path, total_rounds):
    """Resume until total_rounds, checkpointing each completed round atomically.

    An interruption in a round leaves the preceding checkpoint intact, including
    its RNG state; restarting repeats that round. No wall-clock determinism is
    implied. The current state is also the incumbent because acceptance is greedy.
    """
    if not isinstance(total_rounds, int) or total_rounds < 0:
        raise ValueError('total_rounds must be a nonnegative integer')
    layout, state = load_checkpoint(path)
    if state.get('kind') != 'rebuild-run-v1':
        raise ValueError('Not a rebuild-run checkpoint')
    if total_rounds < state['rounds']:
        raise ValueError('total_rounds precedes checkpoint')
    state.setdefault('run_id', uuid.uuid4().hex)
    config = state['config']
    verify_uncertainty_model(layout, state.get('uncertainty_model'), config.get('stone_margins'), config.get('patio_margin', 0))
    if validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9), stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0)):
        raise ValueError('Checkpoint layout is invalid')
    geometry_cache = _GeometryDigestCache(layout)
    while state['rounds'] < total_rounds:
        result = rebuild_search(layout, rounds=1, **config)
        if result.rounds == 0:
            break
        state['rounds'] += 1
        state['attempts'] += result.attempts
        state['relaxation_attempts'] = state.get('relaxation_attempts', 0) + result.relaxation_attempts
        state['relaxation_skipped'] = state.get('relaxation_skipped', 0) + result.relaxation_skipped
        state['relaxation_successes'] = state.get('relaxation_successes', 0) + result.relaxation_successes
        state['relaxation_failures'] = state.get('relaxation_failures', 0) + result.relaxation_failures
        state['mobile_stone_visits'] = state.get('mobile_stone_visits', 0) + result.mobile_stone_visits
        state['improvements'] += result.improvements
        state['secondary_improvements'] = state.get('secondary_improvements', 0) + result.secondary_improvements
        state['secondary_score'] = result.secondary_score
        state['best_history'].append(result.coverage)
        save_checkpoint(layout, path, state, _geometry_cache=geometry_cache)
    return layout, state


def start_insertion(layout, path, *, gap=.01, edge=.01, proposals=50,
                    strategy='random', relaxation_steps=0, grid_resolution=16,
                    stone_margins=None, patio_margin=0, frozen_ids=None, batch_size=1, relaxation_step=.02, uncertainty_model=None, local_relaxation=False, max_relaxation_penalty=None, mobility_policy="circle", tolerance=1e-9):
    """Start a deterministic insertion pass with a checkpoint before work."""
    from .search import greedy_insert
    config = {'gap': gap, 'edge': edge, 'tolerance': tolerance, 'proposals': proposals, 'strategy': strategy,
              'relaxation_steps': relaxation_steps, 'relaxation_step': relaxation_step, 'grid_resolution': grid_resolution, 'local_relaxation': local_relaxation, 'max_relaxation_penalty': max_relaxation_penalty, 'mobility_policy': mobility_policy,
              'stone_margins': stone_margins, 'patio_margin': patio_margin,
              'frozen_ids': list(frozen_ids) if frozen_ids is not None else []}
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if batch_size > 1 and (strategy != 'random' or relaxation_steps != 0):
        raise ValueError('batched insertion requires random strategy and no relaxation')
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    # Empty selection performs argument and layout checks without consuming RNG.
    greedy_insert(layout, stone_ids=(), **config)
    config['batch_size'] = batch_size
    order = sorted((sid for i, sid in enumerate(layout.ids) if not layout.active[i]),
                   key=lambda sid: -layout.shapes[sid].area)
    state = {'run_id': uuid.uuid4().hex, 'kind': 'insertion-run-v1', 'config': config, 'order': order,
             'next_index': 0, 'attempts': 0, 'accepted': 0, 'relaxation_attempts': 0, 'relaxation_skipped': 0, 'relaxation_successes': 0, 'relaxation_failures': 0, 'mobile_stone_visits': 0,
             'best_history': [coverage(layout)], 'uncertainty_model': uncertainty_model}
    save_checkpoint(layout, path, state)
    return state


def continue_insertion(path, max_stones=None):
    """Continue at a stone boundary; interrupted stones are tried again."""
    from .search import greedy_insert
    if max_stones is not None and (not isinstance(max_stones, int) or max_stones < 0):
        raise ValueError('max_stones must be a nonnegative integer or None')
    layout, state = load_checkpoint(path)
    if state.get('kind') != 'insertion-run-v1':
        raise ValueError('Not an insertion-run checkpoint')
    state.setdefault('run_id', uuid.uuid4().hex)
    config = state['config']
    verify_uncertainty_model(layout, state.get('uncertainty_model'), config.get('stone_margins'), config.get('patio_margin', 0))
    if validate(layout, config['gap'], config['edge'], config.get('tolerance', 1e-9), stone_margins=config.get('stone_margins'), patio_margin=config.get('patio_margin', 0)):
        raise ValueError('Checkpoint layout is invalid')
    limit = len(state['order']) if max_stones is None else min(len(state['order']), state['next_index'] + max_stones)
    geometry_cache = _GeometryDigestCache(layout)
    while state['next_index'] < limit:
        sid = state['order'][state['next_index']]
        if sid not in layout.ids or layout.active[layout.ids.index(sid)]:
            raise ValueError('Checkpoint insertion schedule is inconsistent with layout')
        options = dict(config)
        batch_size = options.pop('batch_size', 1)
        if batch_size == 1:
            result = greedy_insert(layout, stone_ids=(sid,), **options)
        else:
            from .search import batched_random_insert
            options.pop('strategy')
            options.pop('relaxation_steps')
            options.pop('relaxation_step', None)
            options.pop('grid_resolution')
            options.pop('local_relaxation', None)
            options.pop('max_relaxation_penalty', None)
            options.pop('mobility_policy', None)
            result = batched_random_insert(layout, stone_ids=(sid,), batch_size=batch_size, **options)
        state['next_index'] += 1
        state['attempts'] += result.attempts
        state['relaxation_attempts'] = state.get('relaxation_attempts', 0) + result.relaxation_attempts
        state['relaxation_skipped'] = state.get('relaxation_skipped', 0) + result.relaxation_skipped
        state['relaxation_successes'] = state.get('relaxation_successes', 0) + result.relaxation_successes
        state['relaxation_failures'] = state.get('relaxation_failures', 0) + result.relaxation_failures
        state['mobile_stone_visits'] = state.get('mobile_stone_visits', 0) + result.mobile_stone_visits
        state['accepted'] += result.accepted
        state['best_history'].append(result.coverage)
        save_checkpoint(layout, path, state, _geometry_cache=geometry_cache)
    return layout, state
