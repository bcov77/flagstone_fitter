"""Checkpointable proposal search. Empty searches never consume physics attempts."""
import copy
import time
import traceback
from shapely.errors import GEOSException
from common import read, write

ATTEMPT_POLICY = ('Only a chosen single/double placement evaluated by physics consumes an attempt. '
                  'Empty or numerically failed searches retry with broader budgets and saved RNG state; '
                  'STOP checkpoints between searches. Inherited attempts keep their original semantics.')


def search_config(cfg, document, round_number):
    """Expand search breadth without weakening eligibility or fit prerequisites."""
    expanded = dict(cfg)
    multiplier = 2 ** min(round_number - 1, 8)
    pieces = len(document['stones'])
    for key in ['piece_budget', 'ordinary_piece_budget', 'scaled_piece_budget', 'double_piece_budget']:
        if key in cfg: expanded[key] = min(pieces, cfg[key] * multiplier)
    for key in ['target_count', 'target_budget']:
        if key in cfg: expanded[key] = min(pieces + 8, cfg[key] * multiplier)
    for key in ['coarse_per_target', 'scaled_coarse_per_target', 'refine_per_target', 'planning_pool']:
        if key in cfg: expanded[key] = min(max(cfg[key], 2 * pieces), cfg[key] * multiplier)
    return expanded


def search(folder, number, document, start, cfg, mode, rng, generate, choose, excluded, stopped, progress):
    """Return a real proposal or None on STOP; persist each unsuccessful round."""
    directory = folder / f'proposal_search_{number:03d}'
    checkpoint = directory / 'checkpoint.json'
    saved = read(checkpoint) if checkpoint.exists() else None
    if saved:
        if saved['start'] != start: raise RuntimeError('Proposal search checkpoint board differs')
        rng.bit_generator.state = saved['rng_state']
    round_number = saved['next_round'] if saved else 1
    while True:
        if stopped(): return None
        expanded = search_config(cfg, document, round_number)
        progress(round_number)
        before = copy.deepcopy(rng.bit_generator.state)
        error = None
        try:
            pool, diagnostics = generate(document, expanded, rng, excluded=excluded)
            chosen, selection = choose(pool, mode, cfg, rng)
        except GEOSException as exc:
            error = dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            pool, chosen, selection, diagnostics = [], None, {}, dict(numerical_failure=error)
        if chosen is not None:
            selection['search_round'] = round_number
            selection['search_config'] = expanded
            selection['search_rng_state_before'] = before
            return pool, diagnostics, chosen, selection
        write(directory / f'round_{round_number:06d}.json.gz', dict(
            round=round_number, reason='selection_geometry_failure' if error else 'no_candidate_in_bounded_search',
            config=expanded, diagnostics=diagnostics, rng_state_before=before,
            rng_state_after=rng.bit_generator.state, updated_unix=time.time()))
        round_number += 1
        write(checkpoint, dict(start=start, next_round=round_number, rng_state=rng.bit_generator.state))
