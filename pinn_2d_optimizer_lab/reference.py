"""Independent P2 finite-element forward reference, force balance and mesh refinement."""
import json
from pathlib import Path
import numpy as np
import torch
from skfem import MeshTri, Basis, FacetBasis, ElementVector, ElementTriP2, LinearForm, asm, condense, solve
from skfem.models.elasticity import linear_elasticity
from physics import true_pressure


def solve_reference(nx=65, ny=33):
    # Coordinates are (X/W,Y/W), with W/H=2. Nondimensional E=1.
    mesh = MeshTri.init_tensor(np.linspace(0, 1, nx), np.linspace(0, 0.5, ny))
    basis = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    nu = 0.3
    lam, mu = nu/(1-nu**2), 1/(2*(1+nu))  # plane stress, not 3D lambda
    stiffness = asm(linear_elasticity(lam, mu), basis)
    top = mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.5))
    boundary = FacetBasis(mesh, basis.elem, facets=top, intorder=8)

    @LinearForm
    def load(v, w):
        return -true_pressure(w.x[0])*v[1]

    rhs = asm(load, boundary)
    fixed = basis.get_dofs(lambda x: np.isclose(x[1], 0))
    solution = solve(*condense(stiffness, rhs, D=fixed))
    reaction = stiffness@solution-rhs
    vertical = basis.nodal_dofs[1]  # pressure integral via assembled load, includes P2 facet DOFs below
    # Split vector DOF indices includes nodal and edge DOFs.
    vertical_all = basis.split_indices()[1]
    total_load = -rhs[vertical_all].sum()
    total_reaction = reaction[vertical_all].sum()
    balance = abs(total_reaction-total_load)/abs(total_load)

    def evaluate(points):
        coordinates = np.asarray(points).copy().T
        coordinates[1] *= 0.5
        # P2 vector interpolation at arbitrary points, independent of PINN grid.
        return np.asarray(basis.interpolator(solution)(coordinates)).T

    return evaluate, {'mesh_nodes': mesh.p.shape[1], 'elements': mesh.t.shape[1],
                      'dofs': int(basis.N), 'force_balance_relative': float(balance),
                      'assembled_force_n': float(total_load*40),
                      'bottom_reaction_n': float(total_reaction*40)}


def generate_reference(cfg, output):
    output = Path(output)
    sensors = (0.1+0.8*torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+37).draw(cfg.n_sensors).double()).numpy()
    axis = np.linspace(0, 1, cfg.grid)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.stack((xx.flatten(), yy.flatten()), 1)
    fine, fine_meta = solve_reference(65, 33)
    coarse, coarse_meta = solve_reference(33, 17)
    fine_grid, coarse_grid = fine(grid), coarse(grid)
    difference = np.linalg.norm(fine_grid-coarse_grid)/np.linalg.norm(fine_grid)
    result = {'sensor_displacement': fine(sensors).tolist(), 'grid_displacement': fine_grid.tolist(),
              'validation': {'fine': fine_meta, 'coarse': coarse_meta,
                             'coarse_fine_displacement_relative': float(difference),
                             'plane': 'plane_stress', 'units': {'width_mm': 40, 'height_mm': 20,
                             'thickness_mm': 10, 'young_mpa': 1, 'poisson': 0.3}}}
    assert difference < 0.01, f'FEM reference has insufficient mesh convergence: {difference}'
    assert fine_meta['force_balance_relative'] < 1e-8
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
