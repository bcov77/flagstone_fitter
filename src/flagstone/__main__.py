"""Run inspectable examples with `python -m flagstone`."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .phase1 import synthetic, plot, save_json, load_json, validate
from .generate import inventory, cut_witness
from .checkpoint import save_checkpoint, load_checkpoint
from .search import greedy_insert, batched_random_insert, rebuild_search, STRATEGIES
from .runner import start_rebuild, continue_rebuild, start_insertion, continue_insertion
from .diagnostics import layout_metrics
from .placement import save_placement_manifest, plot_placement
from .exploration import start_exploration, continue_exploration
from .moves import pulse
from .iterative import start_iterative, continue_iterative
from .uncertainty import parse_uncertainty_model, verify_uncertainty_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['grid', 'witness', 'search'], default='grid')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--proposals', type=int, default=100)
    parser.add_argument('--passes', type=int, help='total repeated insertion passes')
    parser.add_argument('--max-seconds', type=float, help='time budget for this iterative invocation')
    parser.add_argument('--batch-size', type=int, help='random insertion batch size (default 1)')
    parser.add_argument('--strategy', choices=STRATEGIES, help='insertion proposal strategy (default random)')
    parser.add_argument('--relaxation-steps', type=int, help='local relaxation steps per insertion (default 0)')
    parser.add_argument('--relaxation-step', type=float, help='translation step in layout units')
    parser.add_argument('--local-relaxation', action='store_true', default=None, help='relax a nearby subset and verify the full layout')
    parser.add_argument('--max-relaxation-penalty', type=float, help='skip relaxation above this normalized candidate penalty')
    parser.add_argument('--mobility-policy', choices=['circle', 'clearance'], help='choose mobile neighbors for relaxation')
    parser.add_argument('--pulse-rounds', type=int, default=0, help='bounded vibration or pressure rounds')
    parser.add_argument('--translation-noise', type=float, default=0., help='per-axis vibration bound in layout units')
    parser.add_argument('--angle-noise', type=float, default=0., help='vibration angle bound in radians')
    parser.add_argument('--pressure', type=float, nargs=2, metavar=('X', 'Y'), help='directional pressure potential')
    parser.add_argument('--pulse-steps', type=int, default=20, help='maximum relaxation steps per pulse')
    parser.add_argument('--rebuild-rounds', type=int, default=0)
    parser.add_argument('--explore-rounds', type=int, default=0, help='total exploration round target')
    parser.add_argument('--max-coverage-drop', type=float, help='maximum absolute coverage drop during exploration (default 0.1)')
    parser.add_argument('--nudge-probability', type=float, help='fraction of exploration rounds proposing a single-stone nudge')
    parser.add_argument('--nudge-translation', type=float, help='per-axis nudge bound in layout units')
    parser.add_argument('--nudge-angle', type=float, help='nudge angle bound in radians')
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--patch-padding', type=float, help='limit rebuild proposals to the removed patch plus this padding')
    parser.add_argument('--patch-candidates', type=int, help='maximum unused stones considered per rebuild patch')
    parser.add_argument('--restore-removed', action='store_true', help='try original poses for removed stones left unused after a rebuild')
    parser.add_argument('--secondary-resolution', type=int, help='grid resolution for equal-coverage empty-space ranking')
    parser.add_argument('--secondary-broad-distance', type=float, help='empty-space threshold in layout units')
    parser.add_argument('--secondary-edge-strip', type=float, help='exclude this patio edge strip from secondary ranking')
    parser.add_argument('--durable-insertion', action='store_true', help='checkpoint after each stone insertion attempt')
    parser.add_argument('--durable-rebuild', action='store_true', help='checkpoint each rebuild round; --rebuild-rounds is the total target')
    parser.add_argument('--output', type=Path, default=Path('examples'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume', type=Path)
    source.add_argument('--input', type=Path, help='version 1 layout JSON to pack')
    parser.add_argument('--gap', type=float, help='minimum stone gap in layout units')
    parser.add_argument('--edge', type=float, help='minimum patio edge clearance in layout units')
    parser.add_argument('--tolerance', type=float, help='numeric distance comparison tolerance in layout units (default 1e-9)')
    parser.add_argument('--patio-margin', type=float, help='additional patio uncertainty in layout units')
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument('--stone-margins', type=Path, help='JSON object mapping stone IDs to extra margins')
    bounds.add_argument('--uncertainty-model', type=Path, help='versioned per-stone uncertainty components')
    parser.add_argument('--freeze', action='append', default=[], metavar='ID', help='keep an active stone fixed; repeat for multiple IDs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated_paths = {(args.output / name).resolve() for name in ('layout.json', 'layout.png', 'placement.json', 'placement_map.png', 'metrics.json', 'checkpoint.json', 'insertion_checkpoint.json', 'rebuild_checkpoint.json', 'exploration_checkpoint.json', 'iterative_checkpoint.json')}
    if args.uncertainty_model and args.uncertainty_model.resolve() in generated_paths:
        parser.error('uncertainty model would be overwritten by output; choose a different --output directory')
    if args.uncertainty_model and args.patio_margin is not None:
        parser.error('--uncertainty-model includes the patio bound; do not also pass --patio-margin')
    if args.stone_margins and args.stone_margins.resolve() in generated_paths:
        parser.error('stone margins file would be overwritten by output; choose a different --output directory')
    if args.resume and args.resume.resolve() in {(args.output / name).resolve() for name in ('layout.json', 'placement.json', 'metrics.json')}:
        parser.error('checkpoint file would be overwritten by another output; choose a different --output directory')
    durable_resume = False
    iterative_resume = False
    explore_resume = False
    insertion_resume = False
    stone_margins = None
    uncertainty_model = None
    frozen_ids = []
    run_id = None
    patio_margin = 0
    gap, edge = .01, .01
    tolerance = args.tolerance if args.tolerance is not None else 1e-9
    batch_size = args.batch_size if args.batch_size is not None else 1
    relaxation_steps = args.relaxation_steps if args.relaxation_steps is not None else 0
    relaxation_step = args.relaxation_step
    local_relaxation = bool(args.local_relaxation)
    strategy = args.strategy or 'random'
    max_relaxation_penalty = args.max_relaxation_penalty
    mobility_policy = args.mobility_policy or 'circle'
    patch_padding = args.patch_padding
    max_new_stones = args.patch_candidates
    restore_removed = args.restore_removed
    secondary_resolution = args.secondary_resolution
    secondary_broad_distance = args.secondary_broad_distance
    secondary_edge_strip_width = args.secondary_edge_strip if args.secondary_edge_strip is not None else 0
    if relaxation_steps < 0 or args.pulse_rounds < 0 or args.pulse_steps < 0:
        parser.error('step and pulse counts must be nonnegative')
    if batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.input:
        if args.gap is None or args.edge is None:
            parser.error('--input requires explicit --gap and --edge in the layout units')
        if args.input.resolve() in generated_paths:
            parser.error('input file would be overwritten by output; choose a different --output directory')
        layout = load_json(args.input, seed=args.seed)
        gap, edge = args.gap, args.edge
        patio_margin = args.patio_margin if args.patio_margin is not None else 0
        frozen_ids = args.freeze
        if args.stone_margins:
            stone_margins = json.loads(args.stone_margins.read_text())
            if not isinstance(stone_margins, dict):
                parser.error('--stone-margins must contain a JSON object')
    elif args.resume:
        if args.freeze or args.stone_margins or args.uncertainty_model or args.patio_margin is not None:
            parser.error('resume uses saved frozen stones and margins; start a new run to change them')
        layout, metadata = load_checkpoint(args.resume)
        run_id = metadata.get('run_id')
        uncertainty_model = metadata.get('uncertainty_model')
        durable_resume = metadata.get('kind') == 'rebuild-run-v1'
        insertion_resume = metadata.get('kind') == 'insertion-run-v1'
        explore_resume = metadata.get('kind') == 'exploration-run-v1'
        iterative_resume = metadata.get('kind') == 'iterative-insertion-v1'
        if explore_resume and args.max_coverage_drop is not None:
            parser.error('resume uses the saved exploration coverage drop')
        if explore_resume and any(v is not None for v in (args.nudge_probability, args.nudge_translation, args.nudge_angle)):
            parser.error('resume uses the saved nudge settings')
        saved_config = metadata.get('config', metadata)
        saved_strategy = saved_config.get('strategy', 'random')
        if (durable_resume or insertion_resume or explore_resume or iterative_resume) and args.strategy is not None and args.strategy != saved_strategy:
            parser.error('resume uses the saved proposal strategy')
        if args.strategy is None:
            strategy = saved_strategy
        if durable_resume or explore_resume:
            saved_padding = saved_config.get('patch_padding')
            saved_limit = saved_config.get('max_new_stones')
            if (args.patch_padding is not None and args.patch_padding != saved_padding) or (args.patch_candidates is not None and args.patch_candidates != saved_limit):
                parser.error('resume uses saved local-patch settings')
            patch_padding, max_new_stones = saved_padding, saved_limit
        if durable_resume:
            saved_restore_removed = saved_config.get('restore_removed', False)
            if args.restore_removed and not saved_restore_removed:
                parser.error('resume uses the saved removed-stone restoration setting')
            restore_removed = saved_restore_removed
            secondary_settings = (saved_config.get('secondary_resolution'), saved_config.get('secondary_broad_distance'), saved_config.get('secondary_edge_strip_width', 0))
            requested_settings = (args.secondary_resolution, args.secondary_broad_distance, args.secondary_edge_strip)
            if any(requested is not None and requested != saved for requested, saved in zip(requested_settings, secondary_settings)):
                parser.error('resume uses saved secondary-score settings')
            secondary_resolution, secondary_broad_distance, secondary_edge_strip_width = secondary_settings
        saved_steps = saved_config.get('relaxation_steps', 0)
        saved_step = saved_config.get('relaxation_step', .02)
        saved_local = saved_config.get('local_relaxation', False)
        saved_penalty = saved_config.get('max_relaxation_penalty')
        saved_policy = saved_config.get('mobility_policy', 'circle')
        if (durable_resume or insertion_resume or explore_resume or iterative_resume) and args.mobility_policy is not None and args.mobility_policy != saved_policy:
            parser.error('resume uses the saved mobility policy')
        if args.mobility_policy is None:
            mobility_policy = saved_policy
        if (durable_resume or insertion_resume or explore_resume or iterative_resume) and args.max_relaxation_penalty is not None and args.max_relaxation_penalty != saved_penalty:
            parser.error('resume uses the saved relaxation penalty cutoff')
        if args.max_relaxation_penalty is None:
            max_relaxation_penalty = saved_penalty
        if (durable_resume or insertion_resume or explore_resume or iterative_resume) and args.local_relaxation is not None and args.local_relaxation != saved_local:
            parser.error('resume uses the saved local-relaxation setting')
        if args.local_relaxation is None:
            local_relaxation = saved_local
        if (durable_resume or insertion_resume or explore_resume or iterative_resume) and ((args.relaxation_steps is not None and args.relaxation_steps != saved_steps) or (args.relaxation_step is not None and args.relaxation_step != saved_step)):
            parser.error('resume uses the saved relaxation settings')
        if args.relaxation_steps is None:
            relaxation_steps = saved_steps
        if args.relaxation_step is None:
            relaxation_step = saved_step
        saved_batch_size = saved_config.get('batch_size', 1)
        if insertion_resume and args.batch_size is not None and args.batch_size != saved_batch_size:
            parser.error('resume uses the saved insertion batch size')
        if insertion_resume:
            batch_size = saved_batch_size
        gap = saved_config.get('gap')
        edge = saved_config.get('edge')
        saved_tolerance = saved_config.get('tolerance', 1e-9)
        if args.tolerance is not None and args.tolerance != saved_tolerance:
            parser.error('resume uses the tolerance saved in the checkpoint')
        tolerance = saved_tolerance
        if gap is None or edge is None:
            parser.error('checkpoint must include gap and edge clearances')
        if (args.gap is not None and args.gap != gap) or (args.edge is not None and args.edge != edge):
            parser.error('resume uses the gap and edge saved in the checkpoint')
        frozen_ids = saved_config.get('frozen_ids', [])
        stone_margins = saved_config.get('stone_margins')
        patio_margin = saved_config.get('patio_margin', 0)
    elif args.case == 'grid':
        layout = synthetic(args.seed)
    elif args.case == 'witness':
        layout = cut_witness(args.seed, clearance=.011)
    else:
        layout = inventory(args.seed, counts=(3, 6, 12))
    if not args.input and not args.resume:
        gap = args.gap if args.gap is not None else gap
        edge = args.edge if args.edge is not None else edge
        patio_margin = args.patio_margin if args.patio_margin is not None else 0
        frozen_ids = args.freeze
        if args.stone_margins:
            stone_margins = json.loads(args.stone_margins.read_text())
            if not isinstance(stone_margins, dict):
                parser.error('--stone-margins must contain a JSON object')
    iterative_mode = iterative_resume or args.passes is not None
    if iterative_resume and args.passes is None:
        parser.error('resuming iterative insertion requires --passes with the total target')
    if args.passes is not None and args.passes < 0:
        parser.error('--passes must be nonnegative')
    if args.max_seconds is not None and not iterative_mode:
        parser.error('--max-seconds requires iterative insertion')
    if iterative_mode and (args.durable_insertion or batch_size > 1):
        parser.error('iterative insertion has its own checkpoints and uses scalar proposals')
    if iterative_mode and (args.pulse_rounds or args.rebuild_rounds or args.durable_rebuild or args.explore_rounds):
        parser.error('run pulse, rebuild, or exploration in a separate invocation from iterative insertion')
    if args.passes is not None and (durable_resume or insertion_resume or explore_resume):
        parser.error('resume this checkpoint using its original schedule before starting iterative insertion')
    if args.uncertainty_model:
        uncertainty_model = json.loads(args.uncertainty_model.read_text())
        stone_margins, patio_margin = parse_uncertainty_model(layout, uncertainty_model)
    verify_uncertainty_model(layout, uncertainty_model, stone_margins, patio_margin)
    if relaxation_step is None:
        if layout.units != 'normalized' and (relaxation_steps or args.pulse_rounds):
            parser.error('physical-unit relaxation requires an explicit --relaxation-step in layout units')
        relaxation_step = .02
    if args.mobility_policy == 'clearance' and not relaxation_steps:
        parser.error('--mobility-policy clearance requires --relaxation-steps')
    if max_relaxation_penalty is not None and not relaxation_steps:
        parser.error('--max-relaxation-penalty requires --relaxation-steps')
    if args.local_relaxation and not relaxation_steps:
        parser.error('--local-relaxation requires --relaxation-steps')
    if batch_size > 1 and relaxation_steps:
        parser.error('batched insertion does not support relaxation')
    if any(v is not None for v in (args.secondary_resolution, args.secondary_broad_distance, args.secondary_edge_strip)) and not (args.rebuild_rounds or args.durable_rebuild or durable_resume):
        parser.error('secondary scoring options require a rebuild phase')
    if (args.patch_padding is not None or args.patch_candidates is not None) and not (args.rebuild_rounds or args.durable_rebuild or durable_resume or args.explore_rounds or explore_resume):
        parser.error('local-patch options require a rebuild phase')
    if any(v is not None for v in (args.nudge_probability, args.nudge_translation, args.nudge_angle)) and not args.explore_rounds:
        parser.error('nudge options require exploration rounds')
    if args.restore_removed and not (args.rebuild_rounds or args.durable_rebuild or durable_resume):
        parser.error('--restore-removed requires a greedy rebuild phase')
    if args.pulse_rounds and (durable_resume or explore_resume):
        parser.error('cannot add a pulse while resuming a rebuild or exploration schedule')
    if iterative_resume:
        layout, search_result = continue_iterative(args.resume, args.passes, max_seconds=args.max_seconds)
    elif iterative_mode:
        iterative_path = args.output / 'iterative_checkpoint.json'
        start_iterative(layout, iterative_path, gap=gap, edge=edge, tolerance=tolerance, proposals=args.proposals,
                        strategy=strategy, relaxation_steps=relaxation_steps, relaxation_step=relaxation_step,
                        local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty,
                        mobility_policy=mobility_policy, stone_margins=stone_margins, patio_margin=patio_margin,
                        frozen_ids=frozen_ids, uncertainty_model=uncertainty_model)
        layout, search_result = continue_iterative(iterative_path, args.passes, max_seconds=args.max_seconds)
    elif explore_resume:
        search_result = None
    elif insertion_resume:
        layout, search_result = continue_insertion(args.resume)
    elif args.durable_insertion and not durable_resume:
        insertion_path = args.output / 'insertion_checkpoint.json'
        start_insertion(layout, insertion_path, proposals=args.proposals, strategy=strategy, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids, batch_size=batch_size, relaxation_steps=relaxation_steps, relaxation_step=relaxation_step, uncertainty_model=uncertainty_model, local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy)
        layout, search_result = continue_insertion(insertion_path)
    else:
        should_insert = (args.case == 'search' or args.resume or args.input) and not durable_resume
        if should_insert and batch_size > 1:
            if strategy != 'random':
                parser.error('batched insertion requires --strategy random')
            result = batched_random_insert(layout, proposals=args.proposals, batch_size=batch_size, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids)
        else:
            result = greedy_insert(layout, proposals=args.proposals, strategy=strategy, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids, relaxation_steps=relaxation_steps, relaxation_step=relaxation_step, local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy) if should_insert else None
        search_result = asdict(result) if result else None
    pulse_result = None
    if args.pulse_rounds:
        pulse_result = asdict(pulse(layout, rounds=args.pulse_rounds,
                                    translation_noise=args.translation_noise, angle_noise=args.angle_noise,
                                    pressure=args.pressure, relaxation_steps=args.pulse_steps, step=relaxation_step,
                                    gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins,
                                    patio_margin=patio_margin, frozen_ids=frozen_ids))
    if explore_resume:
        rebuilt = None
    elif durable_resume:
        layout, rebuilt = continue_rebuild(args.resume, args.rebuild_rounds)
    elif args.durable_rebuild:
        durable_path = args.output / 'rebuild_checkpoint.json'
        start_rebuild(layout, durable_path, patch_size=args.patch_size, proposals=args.proposals, strategy=strategy, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids, relaxation_steps=relaxation_steps, relaxation_step=relaxation_step, uncertainty_model=uncertainty_model, patch_padding=patch_padding, max_new_stones=max_new_stones, local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy, secondary_resolution=secondary_resolution, secondary_broad_distance=secondary_broad_distance, secondary_edge_strip_width=secondary_edge_strip_width, restore_removed=restore_removed)
        layout, rebuilt = continue_rebuild(durable_path, args.rebuild_rounds)
    else:
        rebuilt_result = rebuild_search(layout, rounds=args.rebuild_rounds, patch_size=args.patch_size, proposals=args.proposals, strategy=strategy, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids, relaxation_steps=relaxation_steps, relaxation_step=relaxation_step, patch_padding=patch_padding, max_new_stones=max_new_stones, local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy, secondary_resolution=secondary_resolution, secondary_broad_distance=secondary_broad_distance, secondary_edge_strip_width=secondary_edge_strip_width, restore_removed=restore_removed) if args.rebuild_rounds else None
        rebuilt = asdict(rebuilt_result) if rebuilt_result else None
    exploration_state = None
    if explore_resume:
        _, layout, exploration_state = continue_exploration(args.resume, args.explore_rounds)
    elif args.explore_rounds:
        exploration_path = args.output / 'exploration_checkpoint.json'
        start_exploration(layout, exploration_path, gap=gap, edge=edge, tolerance=tolerance, patch_size=args.patch_size,
                          proposals=args.proposals, strategy=strategy, stone_margins=stone_margins,
                          patio_margin=patio_margin, frozen_ids=frozen_ids,
                          max_coverage_drop=args.max_coverage_drop if args.max_coverage_drop is not None else .1,
                          relaxation_steps=relaxation_steps, relaxation_step=relaxation_step, uncertainty_model=uncertainty_model, local_relaxation=local_relaxation, max_relaxation_penalty=max_relaxation_penalty, mobility_policy=mobility_policy,
                          patch_padding=patch_padding, max_new_stones=max_new_stones,
                          nudge_probability=args.nudge_probability if args.nudge_probability is not None else 0.,
                          nudge_translation=args.nudge_translation if args.nudge_translation is not None else 0.,
                          nudge_angle=args.nudge_angle if args.nudge_angle is not None else 0.)
        _, layout, exploration_state = continue_exploration(exploration_path, args.explore_rounds)
    for phase in (search_result, rebuilt, exploration_state):
        if isinstance(phase, dict) and phase.get('run_id'):
            run_id = phase['run_id']
    save_placement_manifest(layout, args.output / 'placement.json', gap=gap, edge=edge, tolerance=tolerance,
                            stone_margins=stone_margins, patio_margin=patio_margin, frozen_ids=frozen_ids, uncertainty_model=uncertainty_model)
    plot_placement(layout, args.output / 'placement_map.png', gap=gap, edge=edge, tolerance=tolerance,
                   stone_margins=stone_margins, patio_margin=patio_margin,
                   frozen_ids=frozen_ids, uncertainty_model=uncertainty_model)
    save_json(layout, args.output / 'layout.json')
    plot(layout, args.output / 'layout.png', clearance=gap)
    output_checkpoint = args.output / 'checkpoint.json'
    # Never overwrite a durable schedule when its path happens to be checkpoint.json.
    if not (args.resume and (durable_resume or insertion_resume or explore_resume or iterative_resume) and args.resume.resolve() == output_checkpoint.resolve()):
        output_metadata = {'gap': gap, 'edge': edge, 'tolerance': tolerance, 'strategy': strategy, 'proposals': args.proposals, 'rebuild_rounds': args.rebuild_rounds, 'patch_size': args.patch_size, 'batch_size': batch_size, 'local_relaxation': local_relaxation, 'max_relaxation_penalty': max_relaxation_penalty, 'mobility_policy': mobility_policy, 'patch_padding': patch_padding, 'max_new_stones': max_new_stones, 'restore_removed': restore_removed, 'secondary_resolution': secondary_resolution, 'secondary_broad_distance': secondary_broad_distance, 'secondary_edge_strip_width': secondary_edge_strip_width, 'relaxation_steps': relaxation_steps, 'relaxation_step': relaxation_step, 'stone_margins': stone_margins, 'patio_margin': patio_margin, 'frozen_ids': frozen_ids, 'uncertainty_model': uncertainty_model}
        if run_id is not None:
            output_metadata['run_id'] = run_id
        save_checkpoint(layout, output_checkpoint, output_metadata)
    metrics = {'layout': layout_metrics(layout), 'violations': validate(layout, gap=gap, edge=edge, tolerance=tolerance, stone_margins=stone_margins, patio_margin=patio_margin), 'search': search_result, 'pulse': pulse_result, 'rebuild': rebuilt,
               'exploration': {k: v for k, v in exploration_state.items() if k not in ('best_poses', 'best_active')} if exploration_state else None}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
