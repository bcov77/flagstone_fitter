"""Preserve the user's exact 18-case feedback against immutable saved poses.

Verdicts/tags are explicitly assistant interpretations, never additional quotes.
No selector, geometry, frozen placement, or simulation is changed.
"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEST=HERE/'selection_v4_aggressive'/'review_18_v1'
REVIEWS=[
    (1,'improvement_with_underfill',['improvement','could_be_bigger'],"Great improvement, though, obviously it could be bigger"),
    (2,'strong_overall_approval',['strong_approval'],"Very very good"),
    (3,'approve_difficult_gap_closure',['closes_difficult_upper_gap','residual_space_right'],"Great job filling that area completely. Although we left space on the right. We really sealed off though tough gap above"),
    (4,'improvement_with_underfill',['improvement','could_be_bigger'],"Good improvement. Could be a bit bigger"),
    (5,'acceptable_with_dead_end_risk',['acceptable','point_toward_wall','predicted_gaps_above_below'],"Acceptable. That point straight into the left wall is going to cause a gap above and below we may never be able to fix"),
    (6,'strong_gap_finishing_approval',['finishes_right_side','residual_lower_left_acceptable','gap_finishing_priority'],"Very good. This finishes off the right half of this image. Yes there's a bit open down left. But finishing off gaps is the most important thing to work towards"),
    (7,'approve_choice_imperfect_shape',['good_piece_choice','shape_could_improve'],"Good choice. Obviously could be a slightly better shape but this is still good"),
    (8,'downgrade',['original_better','indent_creates_unfillable_right_gap'],"Downgrade. The original fit better. This new block causes an unfillable gap along the right edge with its indent"),
    (9,'downgrade_with_possible_sequence',['original_corner_reach_better','top_right_bottom_left_residual','possible_later_fill_not_established'],"Still a downgrade. The original did a better job of getting out to the corners. It's conceivable that we could fill the top right and bottom left with new stones, but that's the situation we're in"),
    (10,'space_fill_approval_with_corner_risk',['fills_space','top_right_bottom_right_dead_end_risk','possible_best_or_later_fill_unknown'],"Good job filling the space, however, the bottom right and top right may end up unfillable. It's conceivable this is the best we could do, or that other blocks could fill those, but that's my concern"),
    (11,'neutral_to_worse',['neutral_to_worse','new_lower_left_hole','likely_unfillable'],"Fairly neutral, somewhat worse. We created a big hole botto left that we'll probably never be able to fill"),
    (12,'neutral_with_immutable_corner_cost',['neutral','new_fixed_lower_left_hole','upper_left_flexible_neighbor_opportunity','residual_locations_have_different_cost'],"Neugral. We opened up a new hole bottom left but filled space top left. One thing to note is that bottom left is a corner that will never change, but top left is a free space that is adjacent to another block. We can in theory fix top left by fixing other blocks, but this change locks in a hole in the bottom left that can never be corrected"),
    (13,'larger_piece_approval_with_terminal_corner_risk',['larger_piece_approval','top_right_locked_hole','another_stone_cannot_fix_it'],"Excellent move to a larger stone, but same as the last one, this locks in the top right with a gap in a situation where another stone will never fix it"),
    (14,'improvement_with_repairable_neighbor_gaps',['upper_surface_alignment_improved','improvement','bottom_right_gaps','neighbor_changes_may_repair'],"Good job making it flat against the upper surface, and improvement, though, not perfect. The gap along the bottom and right will be hard to fill, but maybe all gaps could be fixed by changing the nearby stones"),
    (15,'neutral_wall_gap_corner_tradeoff',['neutral','locked_long_wall_gap','better_corner_reach'],"Neutral. This lockss in a long gap along the wall but reaches into the corners in a way the original doesn't"),
    (16,'slight_downgrade_future_alignment_lost',['slight_downgrade','original_future_flush_wall_option','new_upper_dead_end_gap'],"Slight downgrade. The original has a flat wall that, while not yet, could be flush against the right wall if everything else was modified. The new soloution introduces a tough gap up top that may never be filled"),
    (17,'long_term_fit_concern',['original_flat_bottom_future_alignment','more_area_not_enough','long_term_piece_choice_concern'],"Sme story as the previous. The original has a nice flat bottom that can next along the bottom. The new one is clever and fills more space, but may not be the best piece long term"),
    (18,'fairly_good_with_terminal_corner_risk',['fairly_good','locked_bottom_right_hole'],"Fairly good, but we're locking in a hole bottom right"),
]
GENERAL="""In general on the topic of "this locks in something that will never change", this happens most frequenly in corners, sometimes in walls, and sometimes against very, very well packed stones. It's conceivable that as the grid fills in, certain regions will be very good and never changed. If one locks in a hole against these guys, that hole can never be corrected unless we find a small stone to put in there

These look really good though. We're approaching a great hueristic. Im' trying to give suggestions for pieces that won't lead to dead ends farther down the line. That's kind of the main point of all of this, don't place a piece that locks in gaps that will be hard to fix later. It's best to make several perfect connections along the sides that are unlikely to change than it is to fill a big space on all sides with crappy connections"""


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths=[DEST/'placements.json',DEST/'review_18.png',*[DEST/f'case_{i:02}.png' for i in range(1,19)]]
    before={p.name:sha(p) for p in paths}
    placements=json.loads(paths[0].read_text());by_number={p['number']:p for p in placements}
    assert sorted(by_number)==list(range(1,19))
    records=[]
    for number,verdict,tags,comment in REVIEWS:
        p=by_number[number];t=p['target'];c=p['candidate']
        records.append(dict(review_number=number,reviewer='user',user_comment_verbatim=comment,
            target_id=t['id'],family=t['family'],context=t['context'],synthetic=bool(t['removed']),
            original_stone=t['old'],candidate_stone=c['stone'],candidate_pose=c['pose'],
            case_image=f'case_{number:02}.png',case_image_sha256=before[f'case_{number:02}.png'],
            placement_manifest_sha256=before['placements.json'],interpreted_verdict=verdict,
            interpreted_tags=tags,interpretation_is_assistant_coding=True))
    context=dict(general_comment_verbatim=GENERAL,placement_manifest_sha256=before['placements.json'],
        overview_sha256=before['review_18.png'],numbering='Expanded v4 review_18_v1; independent of earlier 24-case and six-case panels.',
        interpretation='Prioritize durable edge/corner closure and preserve future repair options; added area cannot buy new hard-to-repair residual gaps.',
        interpretation_is_assistant_coding=True,
        warning='Purposive related-layout review, not a population approval-rate estimate. Conditional praise is not an unconditional replacement label.')
    outputs={DEST/'human_feedback.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),
             DEST/'human_feedback_context.json':json.dumps(context,indent=2,ensure_ascii=False)+'\n'}
    md=['# Exact expert feedback on the expanded 18-case panel','',
        'User comments below are verbatim, with spelling retained. Verdicts and tags in the JSONL are assistant interpretations. '
        'Numbering is bound to the frozen placement manifest and individual image hashes. The panel and poses were not changed.', '',
        '| # | Action | User comment |','|---|---|---|']
    for r in records:
        action=f"{r['original_stone']} → {r['candidate_stone']}" if r['original_stone'] else f"Insert {r['candidate_stone']} (synthetic opening)"
        md.append(f"| {r['review_number']:02} | {action} | {r['user_comment_verbatim']} |")
    md+=['','## General feedback (verbatim)','',*['> '+line for line in GENERAL.splitlines()]]
    outputs[DEST/'human_feedback.md']='\n'.join(md)+'\n'
    for path,contents in outputs.items():
        if path.exists() and path.read_text()!=contents:
            raise RuntimeError(f'Refusing to overwrite different feedback: {path}')
    for path,contents in outputs.items():path.write_text(contents)
    assert before=={p.name:sha(p) for p in paths}
    print('Preserved all 18 comments and general feedback; exact poses/images unchanged.')


if __name__=='__main__':main()
