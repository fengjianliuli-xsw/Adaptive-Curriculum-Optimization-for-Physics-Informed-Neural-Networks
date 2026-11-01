"""Quarter half-cylinder, plane strain, frictionless unilateral nodal contact.

The reference solves an inequality-constrained elasticity problem. Hertz is only
an independent local asymptotic comparison, never an active-set input.
"""
import json
from pathlib import Path
import numpy as np
from skfem import MeshTri, Basis, FacetBasis, ElementTriP1, ElementVector, LinearForm, asm, condense, solve
from skfem.models.elasticity import linear_elasticity


def quarter_mesh(nr=48, nt=160):
    theta = np.linspace(0, 1, nt+1)**1.6*(np.pi/2)
    radii = 1-(1-np.linspace(0, 1, nr+1))**1.6
    points = [[0., 1.]]
    for radius in radii[1:]:
        points.extend(np.stack((radius*np.sin(theta), 1-radius*np.cos(theta)), 1))
    triangles = []
    for i in range(nt):
        triangles.append([0, 1+i, 2+i])
    for j in range(1, nr):
        inner, outer = 1+(j-1)*(nt+1), 1+j*(nt+1)
        for i in range(nt):
            triangles.extend([[inner+i, outer+i, outer+i+1], [inner+i, outer+i+1, inner+i+1]])
    mesh = MeshTri(np.array(points).T, np.array(triangles).T)
    arc = np.arange(1+(nr-1)*(nt+1), 1+nr*(nt+1))
    return mesh, arc


def solve_contact(cfg, nr=48, nt=160):
    mesh, arc = quarter_mesh(nr, nt)
    basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=3)
    nu, load = cfg.poisson, cfg.load_ratio
    lam, mu = nu/((1+nu)*(1-2*nu)), 1/(2*(1+nu))
    matrix = asm(linear_elasticity(lam, mu), basis).tocsr()
    top = mesh.facets_satisfying(lambda x: np.isclose(x[1], 1.0))
    fbasis = FacetBasis(mesh, basis.elem, facets=top, intorder=4)

    @LinearForm
    def applied(v, w):
        return -load*v[1]

    rhs = asm(applied, fbasis)
    symmetry = basis.get_dofs(lambda x: np.isclose(x[0], 0)).all('u^1')
    contact_dofs = basis.nodal_dofs[1, arc]
    gap0 = mesh.p[1, arc]
    active = np.zeros(len(arc), dtype=bool); active[0] = True
    diagonal = matrix.diagonal()[contact_dofs]
    history = []
    seen = set()
    for iteration in range(120):
        prescribed = basis.zeros()
        prescribed[contact_dofs[active]] = -gap0[active]
        fixed = np.unique(np.concatenate((symmetry, contact_dofs[active])))
        solution = solve(*condense(matrix, rhs, x=prescribed, D=fixed))
        reaction = matrix@solution-rhs
        gap = gap0+solution[contact_dofs]
        pressure_force = np.where(active, reaction[contact_dofs], 0)
        new_active = pressure_force-diagonal*gap > 1e-12
        violation = max(float(np.maximum(-gap, 0).max()), float(np.maximum(-pressure_force, 0).max()))
        history.append({'iteration': iteration, 'active_nodes': int(active.sum()), 'violation': violation})
        if np.array_equal(new_active, active):
            break
        signature = new_active.tobytes()
        if signature in seen:
            # Primal/dual tolerances suppress toggling on a zero-force edge node.
            new_active = ((gap < -1e-10) | (pressure_force > 1e-10))
            if np.array_equal(new_active, active):
                break
        seen.add(signature)
        if not new_active.any():
            raise RuntimeError('Contact active set became empty under a compressive load')
        active = new_active
    else:
        raise RuntimeError('Contact active-set iteration did not converge')
    if np.max(np.maximum(-gap, 0)) > 1e-8 or np.min(pressure_force) < -1e-8:
        raise RuntimeError('Reference violates contact feasibility')
    x = mesh.p[0, arc]
    weights = np.zeros_like(x)
    weights[:-1] += np.diff(x)/2; weights[1:] += np.diff(x)/2
    # Upward nodal reactions / projected contact measure -> projected pressure.
    pressure = pressure_force/weights
    force = float(pressure_force.sum())
    applied_force = float(-rhs[basis.nodal_dofs[1]].sum())
    balance = abs(force-applied_force)/applied_force
    contact_indices = np.flatnonzero(pressure_force > 1e-9*load)
    last = int(contact_indices.max())
    width_bracket = [float(x[last]), float(x[min(last+1, len(x)-1)])]

    def evaluate(points):
        return np.asarray(basis.interpolator(solution)(np.asarray(points).T)).T

    return {'x': x, 'pressure_over_e': pressure, 'gap_over_r': gap,
            'displacement_nodes_over_r': solution[basis.nodal_dofs].T,
            'nodes': mesh.p.T, 'triangles': mesh.t.T, 'active': active,
            'evaluate': evaluate, 'history': history,
            'validation': {'nr': nr, 'nt': nt, 'dofs': int(basis.N), 'active_nodes': int(active.sum()),
                           'force_balance_relative': balance, 'max_penetration_over_r': float(np.maximum(-gap, 0).max()),
                           'min_contact_nodal_reaction': float(pressure_force.min()),
                           'contact_half_width_bracket_over_r': width_bracket,
                           'reaction_over_er': force, 'applied_over_er': applied_force}}


def generate(cfg, output):
    output = Path(output)
    coarse = solve_contact(cfg, 32, 112)
    fine = solve_contact(cfg, 48, 160)
    x = np.linspace(0, 1, 401)
    # Grid strictly inside the quarter cylinder, avoid polygon/chord boundary mismatch.
    axis = np.linspace(0, 1, cfg.grid)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack((xx.flatten(), yy.flatten()), 1)
    mask = ((points[:, 0]**2+(points[:, 1]-1)**2 < 1-.003)
            & (points[:, 0] > 1e-8) & (points[:, 1] < 1-1e-8))
    valid = points[mask]
    ref_u, coarse_u = fine['evaluate'](valid), coarse['evaluate'](valid)
    pressure = np.interp(x, fine['x'], fine['pressure_over_e'])/cfg.load_ratio
    coarse_p = np.interp(x, coarse['x'], coarse['pressure_over_e'])/cfg.load_ratio
    width = cfg.hertz_half_width_over_r
    hertz = (2/(np.pi*width))*np.sqrt(np.maximum(0, 1-(x/width)**2))
    # pressure/q: full line load=2qR, so p0/q=4/(pi*a/R).
    hertz *= 2
    result = {'grid_points': points.tolist(), 'grid_mask': mask.tolist(),
              'valid_points': valid.tolist(), 'displacement_over_u0': (ref_u/cfg.load_ratio).tolist(),
              'pressure_x': x.tolist(), 'pressure_over_q': pressure.tolist(),
              'gap_over_u0': (np.interp(x, fine['x'], fine['gap_over_r'])/cfg.load_ratio).tolist(),
              'hertz_pressure_over_q': hertz.tolist(),
              'validation': {'fine': fine['validation'], 'coarse': coarse['validation'],
                             'coarse_fine_displacement_relative': float(np.linalg.norm(ref_u-coarse_u)/np.linalg.norm(ref_u)),
                             'coarse_fine_pressure_relative': float(np.linalg.norm(pressure-coarse_p)/np.linalg.norm(pressure)),
                             'fem_hertz_pressure_relative': float(np.linalg.norm(pressure-hertz)/np.linalg.norm(hertz)),
                             'hertz_half_width_over_r': width}, 'active_set_history': fine['history']}
    assert fine['validation']['force_balance_relative'] < 1e-8
    assert result['validation']['coarse_fine_displacement_relative'] < .02
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez(output.with_suffix('.npz'), nodes=fine['nodes'], triangles=fine['triangles'],
             displacement_over_r=fine['displacement_nodes_over_r'], arc_x=fine['x'],
             pressure_over_e=fine['pressure_over_e'], gap_over_r=fine['gap_over_r'], active=fine['active'])
    return result
