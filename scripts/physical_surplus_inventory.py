"""Scale the synthetic sketch and inventory to a 25-foot overall patio width."""
from shapely.affinity import scale
from flagstone.phase1 import Layout, load_json, local_shape, save_json

x = load_json('examples/flagstone_reference/surplus_inventory.json')
width = x.patio.bounds[2] - x.patio.bounds[0]
factor = 25 / width
y = Layout(scale(x.patio, xfact=factor, yfact=factor, origin=(0,0)),
           {sid:local_shape(scale(s, xfact=factor, yfact=factor, origin=(0,0)).exterior.coords)
            for sid,s in x.shapes.items()}, x.poses, x.active, x.ids, x.rng, units='ft')
save_json(y, 'examples/flagstone_reference/surplus_25ft.json')
print('width_ft', y.patio.bounds[2]-y.patio.bounds[0], 'area_sqft', y.patio.area,
      'inventory_sqft', sum(s.area for s in y.shapes.values()), 'factor',factor)
