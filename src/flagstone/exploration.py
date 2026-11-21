"""Checkpointed valid-state exploration with a separate best incumbent."""
import copy
import uuid
import numpy as np
from .checkpoint import load_checkpoint, save_checkpoint, _GeometryDigestCache
from .phase1 import Layout, validate
from .search import SearchResult, coverage, greedy_insert, _frozen_indices, _patch_scope
from .uncertainty import verify_uncertainty_model


def start_exploration(layout, path, *, gap=.01, edge=.01, patch_size=2, proposals=20,
                      strategy='opening', relaxation_steps=0, stone_margins=None,
                      patio_margin=0, frozen_ids=None, max_coverage_drop=.1, relaxation_step=.02, uncertainty_model=None, local_relaxation=False, max_relaxation_penalty=None, mobility_policy="circle", tolerance=1e-9,
                      patch_padding=None, max_new_stones=None, nudge_probability=0.,
                      nudge_translation=0., nudge_angle=0.):
    """Create a checkpoint before exploring any patch moves.

    Valid moves up to max_coverage_drop below the best-so-far coverage may be
    accepted. This threshold is an absolute patio-area fraction, not temperature.
    """
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError('patch_size must be positive')
    if not np.isfinite(max_coverage_drop) or not 0 <= max_coverage_drop <= 1:
        raise ValueError('max_coverage_drop must be between 0 and 1')
    config = {'gap': gap, 'edge': edge, 'tolerance': tolerance, 'patch_size': patch_size, 'proposals': proposals,
              'strategy': strategy, 'relaxation_steps': relaxation_steps, 'relaxation_step': relaxation_step, 'local_relaxation': local_relaxation, 'max_relaxation_penalty': max_relaxation_penalty, 'mobility_policy': mobility_policy,
              'stone_margins': stone_margins, 'patio_margin': patio_margin,
              'frozen_ids': list(frozen_ids) if frozen_ids is not None else [],
              'max_coverage_drop': max_coverage_drop, 'patch_padding': patch_padding,
              'max_new_stones': max_new_stones, 'nudge_probability': nudge_probability,
              'nudge_translation': nudge_translation, 'nudge_angle': nudge_angle}
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    _check_config(layout, config)
    initial = coverage(layout)
    state = {'run_id': uuid.uuid4().hex, 'kind': 'exploration-run-v1', 'config': config, 'rounds': 0,
             'attempts': 0, 'accepted': 0, 'improvements': 0, 'relaxation_attempts': 0, 'relaxation_skipped': 0, 'relaxation_successes': 0, 'relaxation_failures': 0, 'mobile_stone_visits': 0,
             'nudge_attempts': 0, 'nudge_valid': 0,
             'best_poses': layout.poses.tolist(), 'best_active': layout.active.tolist(),
             'best_history': [initial], 'current_history': [initial], 'uncertainty_model': uncertainty_model}
    save_checkpoint(layout, path, state)
    return state


def _check_config(layout, config):
    padding, limit = config.get('patch_padding'), config.get('max_new_stones')
    if padding is not None and (not np.isfinite(padding) or padding < 0):
        raise ValueError('patch_padding must be finite and nonnegative')
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError('max_new_stones must be a nonnegative integer')
    probability = config.get('nudge_probability', 0.)
    translation = config.get('nudge_translation', 0.)
    angle = config.get('nudge_angle', 0.)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('nudge_probability must be between 0 and 1')
    if not all(np.isfinite(v) and v >= 0 for v in (translation, angle)):
        raise ValueError('nudge bounds must be finite and nonnegative')
    if probability and not (translation or angle):
        raise ValueError('nudge probability requires a positive movement bound')
    greedy_insert(layout, gap=config['gap'], edge=config['edge'], proposals=config['proposals'],
                  strategy=config['strategy'], relaxation_steps=config['relaxation_steps'],
                  stone_margins=config['stone_margins'], patio_margin=config['patio_margin'],
                  frozen_ids=config['frozen_ids'], tolerance=config.get('tolerance', 1e-9), stone_ids=(), relaxation_step=config.get('relaxation_step', .02), local_relaxation=config.get('local_relaxation', False), max_relaxation_penalty=config.get('max_relaxation_penalty'), mobility_policy=config.get('mobility_policy', 'circle'))


def _incumbent(layout, state):
    best = Layout(layout.patio, layout.shapes, state['best_poses'], state['best_active'],
                  layout.ids, copy.deepcopy(layout.rng), units=layout.units,
                  source_outlines=layout.source_outlines)
    c = state['config']
    if validate(best, c['gap'], c['edge'], c.get('tolerance', 1e-9), stone_margins=c['stone_margins'], patio_margin=c['patio_margin']):
        raise ValueError('Checkpoint incumbent is invalid')
    frozen = _frozen_indices(layout, c['frozen_ids'])
    if any(not best.active[i] or not np.array_equal(best.poses[i], layout.poses[i]) for i in frozen):
        raise ValueError('Checkpoint incumbent differs on frozen stones')
    if not np.isclose(coverage(best), state['best_history'][-1], rtol=1e-12, atol=1e-12):
        raise ValueError('Checkpoint incumbent coverage is inconsistent')
    return best


def continue_exploration(path, total_rounds):
    """Continue a run from its last completed round; return current, best, state."""
    if not isinstance(total_rounds, int) or total_rounds < 0:
        raise ValueError('total_rounds must be a nonnegative integer')
    layout, state = load_checkpoint(path)
    if state.get('kind') != 'exploration-run-v1':
        raise ValueError('Not an exploration checkpoint')
    if total_rounds < state['rounds']:
        raise ValueError('total_rounds precedes checkpoint')
    state.setdefault('run_id', uuid.uuid4().hex)
    c = state['config']
    verify_uncertainty_model(layout, state.get('uncertainty_model'), c.get('stone_margins'), c.get('patio_margin', 0))
    _check_config(layout, c)
    best = _incumbent(layout, state)
    frozen = _frozen_indices(layout, c['frozen_ids'])
    geometry_cache = _GeometryDigestCache(layout)
    while state['rounds'] < total_rounds:
        movable = np.array([i for i in np.flatnonzero(layout.active) if i not in frozen], dtype=int)
        if not len(movable):
            break
        before = layout.snapshot()
        try:
            anchor = int(layout.rng.choice(movable))
            probability = c.get('nudge_probability', 0.)
            nudge = bool(probability and layout.rng.random() < probability)
            nudge_violations = []
            if nudge:
                state['nudge_attempts'] = state.get('nudge_attempts', 0) + 1
                translation, angle = c.get('nudge_translation', 0.), c.get('nudge_angle', 0.)
                layout.poses[anchor, :2] += layout.rng.uniform(-translation, translation, 2)
                layout.poses[anchor, 2] += layout.rng.uniform(-angle, angle)
                nudge_violations = validate(layout, c['gap'], c['edge'], c.get('tolerance', 1e-9),
                                            stone_margins=c['stone_margins'], patio_margin=c['patio_margin'])
                if not nudge_violations:
                    state['nudge_valid'] = state.get('nudge_valid', 0) + 1
                region, selected = _patch_scope(layout, np.array([anchor]), c.get('patch_padding'), c.get('max_new_stones'))
            else:
                distances = np.linalg.norm(layout.poses[movable, :2] - layout.poses[anchor, :2], axis=1)
                removed = movable[np.argsort(distances, kind='stable')[:c['patch_size']]]
                region, selected = _patch_scope(layout, removed, c.get('patch_padding'), c.get('max_new_stones'))
                layout.active[removed] = False
            if nudge_violations:
                result = SearchResult(0, 0, coverage(layout), 0, nudge_violations)
            else:
                result = greedy_insert(layout, gap=c['gap'], edge=c['edge'], proposals=c['proposals'],
                                   strategy=c['strategy'], relaxation_steps=c['relaxation_steps'],
                                   stone_margins=c['stone_margins'], patio_margin=c['patio_margin'],
                                   frozen_ids=c['frozen_ids'], tolerance=c.get('tolerance', 1e-9), relaxation_step=c.get('relaxation_step', .02), local_relaxation=c.get('local_relaxation', False), max_relaxation_penalty=c.get('max_relaxation_penalty'), mobility_policy=c.get('mobility_policy', 'circle'),
                                   proposal_bounds=region, stone_ids=selected)
            candidate = coverage(layout)
            best_value = state['best_history'][-1]
            accepted = not result.violations and candidate >= best_value - c['max_coverage_drop'] - 1e-12
            if not accepted:
                # Geometry rolls back; RNG advances to avoid repeating the move.
                layout.poses[:] = before[0]
                layout.active[:] = before[1]
            else:
                state['accepted'] += 1
                if candidate > best_value + 1e-12:
                    state['best_poses'] = layout.poses.tolist()
                    state['best_active'] = layout.active.tolist()
                    best_value = candidate
                    state['improvements'] += 1
            state['attempts'] += result.attempts
            state['relaxation_attempts'] = state.get('relaxation_attempts', 0) + result.relaxation_attempts
            state['relaxation_skipped'] = state.get('relaxation_skipped', 0) + result.relaxation_skipped
            state['relaxation_successes'] = state.get('relaxation_successes', 0) + result.relaxation_successes
            state['relaxation_failures'] = state.get('relaxation_failures', 0) + result.relaxation_failures
            state['mobile_stone_visits'] = state.get('mobile_stone_visits', 0) + result.mobile_stone_visits
            state['rounds'] += 1
            state['best_history'].append(best_value)
            state['current_history'].append(coverage(layout))
            save_checkpoint(layout, path, state, _geometry_cache=geometry_cache)
        except BaseException:
            layout.restore(before)
            raise
    return layout, _incumbent(layout, state), state
