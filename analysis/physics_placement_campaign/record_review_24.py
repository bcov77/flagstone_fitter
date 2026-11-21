"""Attach the user's numbered review to frozen trial identities.
Human judgments are separate from original proxy metrics; no model is fitted.
"""
import json,hashlib
from pathlib import Path
DEST=Path(__file__).with_name('campaign_v1')/'completed_analysis'/'review_24'
placements=json.loads((DEST/'placements.json').read_text())
reviews=[
(1,'risky',['excessive_expected_movement','orientation_not_preoptimized'],"Risky, expected movement farr too great. Had to turn 24 degrees which implies it didn't look like it should fit."),
(2,'reject',['initial_silhouette_mismatch'],"Don't do this. The piece is not remotely fitting prior to minimizaing"),
(3,'strong_approve',['plausible_initial_fit'],"Great"),
(4,'conditional',['orientation_not_preoptimized','corner_overhang_against_straight_edge'],"Somewhat acceptable, the piece had to rotate. Overhanging corners to straight edges is generally not a good idea"),
(5,'reject',['excessive_initial_overlap'],"Not good. Too much overlap prior to minimization"),
(6,'reject',['corner_overhang_against_straight_edge','edge_alignment_mismatch','orientation_not_preoptimized'],"Excessively overhanging corners. None of the edges look like they will match prior to minimization which matches how far it had to rotate"),
(7,'reject',['excessive_initial_overlap','successful_settling_does_not_validate_proposal'],"Too much overlap. I don't care if this ended up working you can't have that much overlap prior to minimizing"),
(8,'conditional_multistep',['weak_final_fit','predictable_residual_gap','flat_edge_in_right_angle_pocket','possible_sequence_value'],"An acceptable idea. But the piece really doesn't fit that well. It left a gap on top which we could predict prior to minimizaing because it's flat surface doesn't match the right angle gap. That said, this could be a valid move in a larger set of steps"),
(9,'approve_with_pose_correction',['good_piece_choice','orientation_not_preoptimized'],"Good placement but it should have been possible to see that there was a better rotation prior to placing"),
(10,'reject',['excessive_initial_overlap','initial_silhouette_mismatch'],"Way too much overlap. This piece isn't even the right shape for the gap"),
(11,'reject',['original_piece_better'],"The previous piece fit better"),
(12,'conditional_risky',['implausible_initial_fit','disruptive_settling'],"Sort of acceptable. But the piece really doesn't look like it's going to fit. The mayhem afterwards confirms this"),
(13,'reject',['initial_silhouette_mismatch','small_overlap_is_not_enough'],"Piece doesn't look like it's going to fit despite not overlapping by much"),
(14,'marginal',['underfill','missed_opportunity'],"Kind of ok I guess but it leaves a lot on the table"),
(15,'reject',['excessive_initial_overlap','initial_silhouette_mismatch'],"Too much overlap prior to minimizing. Doesn't look like it should fit"),
(16,'reject',['initial_silhouette_mismatch','better_alternatives'],"Incredibly bad. This piece doesn't remotely fit here. There are far better choices"),
(17,'marginal',['easy_scenario'],"Ok I guess but this wasn't a hard scenario"),
(18,'reject',['initial_silhouette_mismatch'],"No, this piece is not remotely the right shape for the gap"),
(19,'acceptable_with_better_alternative',['flat_bottom_preferred','better_alternatives'],"Acceptable, though, a piece with a flat bottom would have fit better"),
(20,'reject',['initial_silhouette_mismatch','lucky_relaxation'],"This piece got lucky with where it minimized to. It does not look at all like the gap it is replacing"),
(21,'reject',['initial_silhouette_mismatch'],"Piece does not look remotely like the region it is replacing. Not good"),
(22,'reject',['underfill','better_alternatives'],"Certainly there is a better piece for this gap. It doesn't fill the space at all"),
(23,'marginal',['original_piece_better','underfill','better_alternatives'],"Semi good. A different piece (like the original) could fill this space better but this is ok. Not good though"),
(24,'reject',['implausible_initial_fit'],"This piece doesn't remotely fit to start with")]
overall="On the whole these are all quite bad except #3. They are exactly the kinds of things you'd expect to see from an algorithm that doesn't know what it's doing and isn't thinking like a human. The pieces we're inserting don't look like the gap they are trying to fill. This is the most important part, they should be trying to fill the gap leaving no extra space, rather than just hoping and praying the situation explodes and comes back looking ok after minimization"
assert [r[0] for r in reviews]==list(range(1,25))
by_number={p['number']:p for p in placements};records=[]
for number,verdict,tags,comment in reviews:
 p=by_number[number]
 records.append({'review_number':number,'task_id':p['id'],'case':p['case'],'target':p['target'],
                 'candidate_stone':p['stone'],'removed_stone':p['removed_stone'],'family':p['family'],
                 'reviewer':'user','user_comment_verbatim':comment,'interpreted_verdict':verdict,
                 'interpreted_tags':tags,'interpretation_is_assistant_coding':True,
                 'original_proxy_useful':p['original_provisional_useful'],
                 'measured_turn_degrees':p['max_saved_turn_degrees'],
                 'measured_neighbor_shift_inches':p['max_neighbor_translation_inches']})
assert len({r['task_id'] for r in records})==24
assert abs(records[0]['measured_turn_degrees']-24.1)<.2
(DEST/'human_feedback.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(DEST/'human_feedback_context.json').write_text(json.dumps({'overall_comment_verbatim':overall,
 'reviewed_placement_manifest_sha256':hashlib.sha256((DEST/'placements.json').read_bytes()).hexdigest(),
 'interpretation':'Initial gap-shape compatibility is a prerequisite. End-state proxy improvement cannot establish proposal quality.',
 'strong_positive_numbers':[3],'positive_with_orientation_correction':[9],
 'conditional_or_marginal_numbers':[4,8,12,14,17,19,23],
 'do_not_collapse_conditional_reviews_into_binary_positives':True,
 'selection_warning':'Purposive stratified review; counts are not an unbiased estimate of expert approval across the full campaign.',
 'orientation_warning':'Large rotation may mean wrong input orientation, not intrinsically wrong stone; bad pre-fit can also occur with little rotation.'},indent=2))
text='''# Expert feedback on the 24-placement sheet\n\nOriginal comments are preserved verbatim with exact task/case IDs in\n`human_feedback.jsonl`. Verdicts and tags below are assistant interpretations;\nthe user text is authoritative. The old area/void labels are preserved as\nhistorical proxy outcomes, not expert fit judgments.\n\nThe user emphasizes that a proposed piece must already resemble its gap, fill\nit with little residual space, and align edges/corners before simulation.\nMinimization cannot justify excessive initial intrusion or a lucky rearrangement.\nOnly #3 received unqualified strong approval. #9 has a good piece choice but\nneeds a better starting orientation. Conditional/marginal examples are kept\nseparate, including #8's possible multi-step value. This is a deliberately\nselected review set, not a population approval-rate estimate.\n\n| # | Candidate / removed | Interpreted verdict | User comment |\n|---|---|---|---|\n'''
for r in records:text+=f"| {r['review_number']:02d} | {r['candidate_stone']} / {r['removed_stone'] or '—'} | {r['interpreted_verdict']} | {r['user_comment_verbatim']} |\n"
text+='\nOverall user comment:\n\n> '+overall+'\n'
(DEST/'human_feedback.md').write_text(text)
print('Recorded all 24 comments against frozen trial IDs; only #3 coded strong approval.')
