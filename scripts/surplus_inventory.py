"""Generate a fixed, synthetic 2.5-patio-area selection inventory."""
from pathlib import Path
from flagstone.generate import flagstone_inventory
from flagstone.phase1 import load_json, save_json

# Approximately preserve the proportions of the approved 156-stone sample.
# 18*.025 + 146*.008 + 294*.003 = 2.5 patio areas.
layout = flagstone_inventory(seed=17, counts=(18, 146, 294), areas=(.025, .008, .003))
layout.patio = load_json('examples/flagstone_reference/patio_inventory.json').patio
out = Path('examples/flagstone_reference/surplus_inventory.json')
save_json(layout, out)
print(f'{out}: {len(layout.ids)} stones, total area {sum(s.area for s in layout.shapes.values()):.6f}, patio {layout.patio.area:.6f}')
