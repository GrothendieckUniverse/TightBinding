# TightBinding.jl

A Julia package for constructing real-space tight-binding models on Bravais lattices,
building uniform momentum-space grids, computing band structures and (many-body) Chern numbers.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TightBinding.jl                              │
│                      (main module entry)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐  │
│  │   Real_Space_Lattice     │    │   Real_Space_TightBinding    │  │
│  │   (real_space_lattice.jl)│───▶│   _Model                     │  │
│  │                          │    │   (real_space_tb_model.jl)   │  │
│  │  • bravais vectors       │    │                              │  │
│  │  • sublattice positions  │    │  • input_hopping_map         │  │
│  │  • sample size & PBC     │    │  • full_hopping_map          │  │
│  │  • twisted phases φ/2π   │    │  • graph-distance hoppings   │  │
│  │  • NN graph (Graphs.jl)  │    │  • Haldane NNN amplitude     │  │
│  │  • plotting (CairoMakie) │    │  • plotting (CairoMakie)     │  │
│  └────────────┬─────────────┘    └──────────────┬───────────────┘  │
│               │                                 │                  │
│               │   ┌──────────────────────────┐  │                  │
│               └──▶│    Uniform_Grids          │◀─┘                  │
│                   │    (uniform_grids.jl)     │                     │
│                   │                           │                     │
│                   │  • momentum grids (k-space)│                    │
│                   │  • flux-shifted grids      │                    │
│                   │  • generic uniform grids   │                    │
│                   └────────────┬──────────────┘                     │
│                                │                                    │
│                   ┌────────────┴──────────────┐                     │
│                   │        utils.jl            │                     │
│                   │                            │                     │
│                   │  • dual_basis_vec_mat/list │                     │
│                   │  • build_Hk_crys (H(k))    │                     │
│                   │  • build_Hamiltonian_matrix│                     │
│                   │  • Chern_number_FHS        │                     │
│                   │  • many_body_Chern_number  │                     │
│                   └────────────┬──────────────┘                     │
│                                │                                    │
│                   ┌────────────┴──────────────┐                     │
│                   │     band_plot.jl            │                     │
│                   │                            │                     │
│                   │  • plot_bands (1D path)    │                     │
│                   │  • plot_band_contour (2D)  │                     │
│                   │  • find_1st_BZ_k_cart_list │                     │
│                   └────────────────────────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Real_Space_Lattice ──▶ Real_Space_TightBinding_Model
        │                        │
        │   twisted_phase_over_2π│  input_hopping_map
        ▼                        ▼
  Uniform_Grids             build_Hk_crys() ──▶ H(k) Bloch Hamiltonian
   (k-space grid)                  │
        │                          ▼
        │              Chern_number_Fukui_Hatsugai_Suzuki
        │                   (single-particle Chern number)
        │
        ▼
  build_Hamiltonian_matrix(tb, θ)
        │
        ▼
  many_body_Chern_number_Fukui_Hatsugai_Suzuki
        (many-body Chern number on flux torus)
```

---

## Key Concepts

### Twisted Phases (Flux Insertion)

The field `twisted_phase_over_2π` in `Real_Space_Lattice` encodes magnetic fluxes
threaded through the periodic directions of the torus:

- **Cylinder geometry** (`pbc_indicator = [true, false]`): only `twisted_phase_over_2π[1]`
  may be non-zero. The second component must be zero.
- **Torus geometry** (`pbc_indicator = [true, true]`): both components may be non-zero.
- **Open boundaries** (`pbc_indicator = [false, false]`): both must be zero.

Inserting a flux φ is equivalent to shifting the crystal momentum grid by φ/L.
This is automatically handled when constructing `Uniform_Grids` from a lattice
with non-zero twisted phases.

### Many-Body Chern Number

For a non-interacting tight-binding model, the **many-body Chern number** computed
on the flux torus (θ₁, θ₂) ∈ [0,1]² equals the sum of single-particle Chern numbers
of all occupied bands. This package provides:

- `Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band, nk)` — single-particle
- `many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb_model; n_occ, nθ)` — many-body

Key properties verified by the tests:

| Filling | `n_occ` | Many-body C |
|---|---|---|
| Half-filling (lowest band) | `n_cell` | `C_lower` |
| Full filling (all bands) | `2·n_cell` | `0` (trivial insulator) |

---

## Usage Examples

### 1. Build a Haldane Honeycomb Model & Compute Chern Numbers

```julia
using TightBinding

# Build a 3×3 honeycomb lattice with PBC (torus)
lat = initialize_real_space_lattice(;
    lattice_name = "honeycomb",
    sample_size   = [3, 3],
    pbc_indicator = [true, true],
)

tb = initialize_real_space_tightbinding_model(lat; model_name = "Haldane")

# On-site staggered potential
add_hopping_term!(tb, (([0,0], 1), ([0,0], 1)) => 0.7;  is_hermitian=false)
add_hopping_term!(tb, (([0,0], 2), ([0,0], 2)) => -0.7; is_hermitian=false)

# NN hoppings t₁ = -1
add_hopping_term!(tb, (([0,0], 1), ([0,0], 2)) => -1.0; is_hermitian=true)
add_hopping_term!(tb, (([0,0], 2), ([1,0], 1)) => -1.0; is_hermitian=true)
add_hopping_term!(tb, (([0,0], 2), ([0,1], 1)) => -1.0; is_hermitian=true)

# Complex NNN hoppings (Haldane)
t2, φ = -0.24, π/2
add_hopping_term!(tb, (([0,0], 1), ([0,1], 1)) => t2 * exp(im*φ);  is_hermitian=true)
add_hopping_term!(tb, (([0,0], 1), ([1,0], 1)) => t2 * exp(-im*φ); is_hermitian=true)
add_hopping_term!(tb, (([1,0], 1), ([0,1], 1)) => t2 * exp(-im*φ); is_hermitian=true)
add_hopping_term!(tb, (([0,0], 2), ([0,1], 2)) => t2 * exp(-im*φ); is_hermitian=true)
add_hopping_term!(tb, (([0,0], 2), ([1,0], 2)) => t2 * exp(im*φ);  is_hermitian=true)
add_hopping_term!(tb, (([1,0], 2), ([0,1], 2)) => t2 * exp(im*φ);  is_hermitian=true)

# Build k-space Hamiltonian
Hk_crys = build_Hk_crys(tb)

# Single-particle Chern numbers
C1 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=1, nk=51)
C2 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=2, nk=51)
println("C_lower = $C1,  C_upper = $C2")  # C_lower = 1.0, C_upper = -1.0

# Many-body Chern numbers in flux space
n_b = lat.n_cell   # L² = number of k-points per band

# Half-filling: fill the lowest band → C = +1
C_mb_half = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=n_b, nθ=21)
println("C_mb (half-filling) = $C_mb_half")  # +1.0

# Full filling: all bands occupied → C = 0
C_mb_full = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=2*n_b, nθ=21)
println("C_mb (full filling) = $C_mb_full")   # 0.0
```

### 2. Flux Insertion & Shifted Momentum Grid

```julia
# Lattice with flux φ₁ = 0.8·2π threaded through direction 1
lat_flux = initialize_real_space_lattice(;
    lattice_name = "honeycomb",
    sample_size   = [3, 3],
    pbc_indicator = [true, true],
    twisted_phase_over_2π = [0.8, 0.0],
)

# The momentum grid is automatically shifted
k_grid = initialize_uniform_grids(lat_flux)
# k_grid.site_crys_list[1] ≈ [0.8/3, 0.0]  (shifted by φ/L)
```

### 3. Graph-Based Hopping Generation

```julia
lat = initialize_real_space_lattice(;
    lattice_name = "honeycomb",
    sample_size   = [3, 3],
    pbc_indicator = [true, true],
)
tb = initialize_real_space_tightbinding_model(lat; model_name="Haldane")

# On-site staggered chemical potential via graph distance 0
add_hoppings_by_graph_distance!(tb, 0, (i, j) -> begin
    i == j || return 0.0
    (_, sub) = tb.lattice.site_list[i]
    return sub == 1 ? 0.7 : -0.7
end; is_hermitian=false)

# Isotropic NN hopping via graph distance 1
add_hoppings_by_graph_distance!(tb, 1, -1.0)

# Haldane complex NNN via graph distance 2
add_hoppings_by_graph_distance!(tb, 2, (i, j) ->
    TightBinding.haldane_nnn_hopping_amplitude(i, j;
        t2=-0.24, φ=π/2, tb_model=tb
    ); is_hermitian=true)
```

### 4. Band Structure Plotting

```julia
k_data = initialize_uniform_grids(lat)

Hk_crys = build_Hk_crys(tb)

plot_bands(Hk_crys, k_data;
    k_path = [[0,0], [2/3, 1/3], [1/2, 1/2], [1/3, 2/3], [0,0]],
    k_path_name_list = ["Γ", "K", "M", "K'", "Γ"],
    nband_range = 1:lat.n_sub,
    nk = 30,
)
```

---

## Exported API

| Function | Description |
|---|---|
| `initialize_real_space_lattice` | Build a Bravais lattice with sublattices & PBC |
| `plot_real_space_lattice` | Visualise lattice with NN graph edges |
| `initialize_real_space_tightbinding_model` | Create a tight-binding model on a lattice |
| `add_hopping_term!` | Add a hopping term (template + translated copies) |
| `add_hoppings_by_graph_distance!` | Add hoppings by graph distance on the NN graph |
| `haldane_nnn_hopping_amplitude` | Compute Haldane complex NNN amplitude |
| `plot_real_space_tightbinding_model` | Visualise hoppings on the lattice |
| `initialize_uniform_grids` | Build uniform grids (momentum or flux space) |
| `build_Hk_crys` | Construct H(k) Bloch Hamiltonian |
| `build_Hamiltonian_matrix` | Build real-space H matrix with twisted BCs |
| `Chern_number_Fukui_Hatsugai_Suzuki` | Single-particle Chern number (FHS method) |
| `many_body_Chern_number_Fukui_Hatsugai_Suzuki` | Many-body Chern number on flux torus |
| `dual_basis_vec_mat` / `dual_basis_vec_list` | Compute dual (reciprocal) basis vectors |
| `plot_bands` | Plot 1D band structure along a k-path |
| `plot_band_contour` | Plot 2D band contour with BZ outline |
| `find_1st_BZ_k_cart_list` | Find 1st BZ vertices (Wigner-Seitz cell) |

---

## Tests

Run the test suite:

```bash
julia --project=. -e 'include("test/test.jl")'
```

The tests verify:
- Lattice construction for honeycomb, square, and checkerboard geometries
- Manual hopping term insertion and graph-based hopping generation
- Graph construction (nearest-neighbor by Euclidean distance, `allowed_bonds` filtering)
- k-space Hamiltonian construction in periodic gauge
- Single-particle Chern numbers via Fukui–Hatsugai–Suzuki (Haldane: C = ±1)
- **Many-body Chern numbers on the flux torus**: half-filling matches single-particle,
  half-filling matches single-particle, and integer filling gives zero.
- **Twisted phase validation** (non-zero only along periodic directions)
- **Flux-shifted momentum grids**

---

## Version History

- **0.3.0** — Added `twisted_phase_over_2π`, flux-shifted `Uniform_Grids`,
  `build_Hamiltonian_matrix`, `many_body_Chern_number_Fukui_Hatsugai_Suzuki`,
  and many-body Chern number tests.
- **0.2.0** — Graph-based hopping generation, `allowed_bonds` filtering,
  Haldane NNN helper, lattice & model plotting.
- **0.1.0** — Initial release: `Real_Space_Lattice`, `Real_Space_TightBinding_Model`,
  `Uniform_Grids`, `build_Hk_crys`, `Chern_number_Fukui_Hatsugai_Suzuki`,
  band plotting.
