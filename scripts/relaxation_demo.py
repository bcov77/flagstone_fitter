"""Generate a before/after controlled rotation example."""
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
from flagstone.phase1 import Layout, polygon, local_shape, plot
from flagstone.relaxation import relax

out = Path('examples/relaxation')
out.mkdir(parents=True, exist_ok=True)
shape = local_shape([(-.3, -.05), (.3, -.05), (.3, .05), (-.3, .05)])
x = Layout(polygon([(0, 0), (.4, 0), (.4, .8), (0, .8)]), {'s': shape}, [[.2, .4, .5]], [True], ('s',), np.random.default_rng(1))
plot(x, out / 'before.png', clearance=.01)
result = relax(x, step=.01)
plot(x, out / 'after.png', clearance=.01)
(out / 'metrics.json').write_text(json.dumps(asdict(result), indent=2))
