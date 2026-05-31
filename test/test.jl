module Lattice_Test

using Revise
using LinearAlgebra
using CairoMakie
using MLStyle
using Test

using TightBinding

MLStyle.@data Predefined_Model begin
    Haldane_Honeycomb_via_Graph()
    Haldane_Honeycomb()
    Extented_Haldane_Honeycomb()
    CheckerBoard()
    Singular_Band_Dice()
    Anisotropic_Singular_Band_Dice()
end


function test(model::Haldane_Honeycomb_via_Graph)
    # --- 1. Build real-space honeycomb lattice -----------------------------------
    lat = initialize_real_space_lattice(;
        lattice_name="honeycomb",
        sample_size=[3, 3],
        pbc_indicator=[true, true],
    )

    # --- 2. Build tight-binding model (Haldane) ----------------------------------
    tb = initialize_real_space_tightbinding_model(lat; model_name="Haldane")
    # plot_real_space_lattice(tb.lattice)

    # On-site: staggered chemical potential ±μ
    function stagger_chemical_potential(i, j; tb_model=tb, μ=0.7)
        i == j || return 0.0

        (_, sub_i) = tb_model.lattice.site_list[i]

        return sub_i == 1 ? μ : -μ
    end
    add_hoppings_by_graph_distance!(tb, 0, stagger_chemical_potential; is_hermitian=false)

    # NN hopping t₁ = -1.0 (isotropic, via graph distance 1)
    add_hoppings_by_graph_distance!(tb, 1, -1.0)


    # NNN hopping: Haldane complex t₂ exp(i φ ν)
    # is_hermitian=true: conj(amp(i,j)) = amp(j,i) for Haldane, so correct
    haldane_amp(i, j) = TightBinding.haldane_nnn_hopping_amplitude(i, j;
        t2=-0.24, φ=π / 2, tb_model=tb,
    )
    add_hoppings_by_graph_distance!(tb, 2, haldane_amp; is_hermitian=true)

    println("Model built: $(length(tb.full_hopping_map)) hoppings in full_hopping_map.")


    # --- 3. Visualise real-space lattice + hoppings -----------------------------
    plot_real_space_tightbinding_model(tb)

    # --- 4. Build k-space grid (reciprocal lattice) ----------------------------
    # Uniform_Grids from the real-space lattice with PBC in all directions
    k_data = initialize_uniform_grids(lat; twisted_phases_over_2π=[0.0, 0.0])

    println("k-grid: $(k_data.nsite) points, cell volume = $(round(k_data.cell_volume, digits=4))")

    # --- 5. Construct k-space Hamiltonian ---------------------------------------
    Hk_crys = build_Hk_crys(tb)

    # --- 6. Plot band structure -------------------------------------------------
    # k_path = [[0, 0], [1 / 2, 0], [2 / 3, 1 / 3], [0, 0]]          # Γ → M → K → Γ
    # k_path_names = ["Γ", "M", "K", "Γ"]

    k_path_names = ["Γ", "K", "M", "K'", "Γ"]
    k_path = [
        [0.0, 0.0],  # Γ
        [2 / 3, 1 / 3],  # K
        [1 / 2, 1 / 2],  # M
        [1 / 3, 2 / 3],  # K'
        [0.0, 0.0],  # Γ
    ]

    fig_bands = plot_bands(Hk_crys, k_data;
        k_path=k_path,
        k_path_name_list=k_path_names,
        nband_range=1:lat.n_sub,
        nk=30,
    )
    display(fig_bands)

    # sanity check
    println("C_lower = ", Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=1, nk=51))
    println("C_upper = ", Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=2, nk=51))
end


function test(model::Haldane_Honeycomb)
    # --- 1. Build real-space honeycomb lattice -----------------------------------
    lat = initialize_real_space_lattice(;
        lattice_name="honeycomb",
        sample_size=[3, 3],
        pbc_indicator=[true, true],
    )

    # --- 2. Build tight-binding model (Haldane) ----------------------------------
    tb = initialize_real_space_tightbinding_model(lat; model_name="Haldane")
    # plot_real_space_lattice(tb.lattice)

    μ = 0.7
    t1 = -1.0

    t2 = -0.24
    φ = π / 2

    # chemical potential
    add_hopping_term!(tb, (([0, 0], 1), ([0, 0], 1)) => μ; is_hermitian=false)
    add_hopping_term!(tb, (([0, 0], 2), ([0, 0], 2)) => -μ; is_hermitian=false)

    # nn hoppings 
    add_hopping_term!(tb, (([0, 0], 1), ([0, 0], 2)) => t1; is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([1, 0], 1)) => t1; is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([0, 1], 1)) => t1; is_hermitian=true)


    # complex nnn hoppings 
    add_hopping_term!(tb, (([0, 0], 1), ([0, 1], 1)) => t2 * exp(im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 1), ([1, 0], 1)) => t2 * exp(-im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([1, 0], 1), ([0, 1], 1)) => t2 * exp(-im * φ); is_hermitian=true)

    add_hopping_term!(tb, (([0, 0], 2), ([0, 1], 2)) => t2 * exp(-im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([1, 0], 2)) => t2 * exp(im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([1, 0], 2), ([0, 1], 2)) => t2 * exp(im * φ); is_hermitian=true)

    println("Model built: $(length(tb.full_hopping_map)) hoppings in full_hopping_map.")


    # --- 3. Visualise real-space lattice + hoppings -----------------------------
    plot_real_space_tightbinding_model(tb)

    # --- 4. Build k-space grid (reciprocal lattice) ----------------------------
    # Uniform_Grids from the real-space lattice with PBC in all directions
    k_data = initialize_uniform_grids(lat; twisted_phases_over_2π=[0.0, 0.0])

    println("k-grid: $(k_data.nsite) points, cell volume = $(round(k_data.cell_volume, digits=4))")

    # --- 5. Construct k-space Hamiltonian ---------------------------------------
    Hk_crys = build_Hk_crys(tb)

    # --- 6. Plot band structure -------------------------------------------------
    k_path_names = ["Γ", "K", "M", "K'", "Γ"]
    k_path = [
        [0.0, 0.0],  # Γ
        [2 / 3, 1 / 3],  # K
        [1 / 2, 1 / 2],  # M
        [1 / 3, 2 / 3],  # K'
        [0.0, 0.0],  # Γ
    ]

    fig_bands = plot_bands(Hk_crys, k_data;
        k_path=k_path,
        k_path_name_list=k_path_names,
        nband_range=1:lat.n_sub,
        nk=30,
    )
    display(fig_bands)

    # sanity check — single-particle Chern numbers
    C_sp_lower = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=1, nk=51)
    C_sp_upper = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=2, nk=51)
    println("C_lower (sp) = ", round(C_sp_lower, digits=6))
    println("C_upper (sp) = ", round(C_sp_upper, digits=6))

    # --- 7. Many-body Chern numbers in flux space -----------------------------
    n_occ = lat.n_cell  # half-filling: fill the lowest L² states

    # 7a. Half-filling: C_mb should match the occupied band's single-particle C
    C_mb_half = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=n_occ, nθ=21)
    println("C_mb (half-filling)  = ", round(C_mb_half, digits=6))
    @test round(C_mb_half, digits=2) ≈ round(C_sp_lower, digits=2) atol = 0.1

    # 7b. Full filling: all bands occupied → total Chern number must be zero
    C_mb_full = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=2 * n_occ, nθ=21)
    println("C_mb (full filling)  = ", round(C_mb_full, digits=6))
    @test round(C_mb_full, digits=2) ≈ 0.0 atol = 0.1

    println("✓ All many-body Chern number checks passed.\n")
end


function test(model::CheckerBoard)
    # --- 1. Build real-space lattice -----------------------------------
    lat = initialize_real_space_lattice(;
        lattice_name="checkerboard",
        sample_size=[3, 3],
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sub_crys_list=[[0.5, 0], [0.0, 0.5]],
        allowed_bonds=[(1, 2)],
        pbc_indicator=[true, true],
    )

    # --- 2. Build tight-binding model & visualize ----------------------
    tb = initialize_real_space_tightbinding_model(lat; model_name="Dice")
    # plot_real_space_lattice(tb.lattice)


    # add complex nn hoppings (graph distance = 1)
    t = 1.0
    φ = π / 4
    add_hopping_term!(tb, (([0, 0], 1), ([0, 0], 2)) => -t * exp(-im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 1), ([1, 0], 2)) => -t * exp(im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([0, 1], 1)) => -t * exp(-im * φ); is_hermitian=true)
    add_hopping_term!(tb, (([1, 0], 2), ([0, 1], 1)) => -t * exp(im * φ); is_hermitian=true)

    # add nnn hoppings (graph distance = 2)
    t1′ = 1 / (2 + sqrt(2))
    t2′ = -1 / (2 + sqrt(2))
    # A: x-direction uses t1′, y-direction uses t2′.
    add_hopping_term!(tb, (([0, 0], 1), ([1, 0], 1)) => -t1′; is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 1), ([0, 1], 1)) => -t2′; is_hermitian=true)
    # B: x-direction uses t2′, y-direction uses t1′.
    add_hopping_term!(tb, (([0, 0], 2), ([1, 0], 2)) => -t2′; is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([0, 1], 2)) => -t1′; is_hermitian=true)

    # add nnnn hoppings (still graph distance = 2)
    t′′ = 1 / (2 + 2 * sqrt(2))
    add_hopping_term!(tb, (([0, 0], 1), ([1, 1], 1)) => -t′′; is_hermitian=true)
    add_hopping_term!(tb, (([0, 0], 2), ([1, 1], 2)) => -t′′; is_hermitian=true)
    add_hopping_term!(tb, (([1, 0], 2), ([0, 1], 2)) => -t′′; is_hermitian=true)
    add_hopping_term!(tb, (([0, 1], 1), ([1, 0], 1)) => -t′′; is_hermitian=true)

    println("Model built: $(length(tb.full_hopping_map)) hoppings in full_hopping_map.")


    # # --- 3. Visualise real-space lattice + hoppings -----------------------------
    plot_real_space_tightbinding_model(tb)

    # --- 4. Build k-space grid (reciprocal lattice) ----------------------------
    # Uniform_Grids from the real-space lattice with PBC in all directions
    k_data = initialize_uniform_grids(lat; twisted_phases_over_2π=[0.0, 0.0])

    println("k-grid: $(k_data.nsite) points, cell volume = $(round(k_data.cell_volume, digits=4))")

    # --- 5. Construct k-space Hamiltonian ---------------------------------------
    Hk_crys = build_Hk_crys(tb)

    # --- 6. Plot band structure -------------------------------------------------
    k_path = [[0, 0], [1 / 2, 0], [1 / 2, 1 / 2], [0, 0]]
    k_path_names = ["Γ", "X", "K", "Γ"]

    fig_bands = plot_bands(Hk_crys, k_data;
        k_path=k_path,
        k_path_name_list=k_path_names,
        nband_range=1:lat.n_sub,
        nk=30,
    )
    display(fig_bands)

    # sanity check — single-particle Chern numbers
    C_sp_lower = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=1, nk=51)
    C_sp_upper = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys; band=2, nk=51)
    println("C_lower (sp) = ", round(C_sp_lower, digits=6))
    println("C_upper (sp) = ", round(C_sp_upper, digits=6))

    # --- 7. Many-body Chern numbers in flux space -----------------------------
    n_occ = lat.n_cell  # half-filling: fill the lowest L² states

    # 7a. Half-filling: C_mb should match the occupied band's single-particle C
    C_mb_half = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=n_occ, nθ=21)
    println("C_mb (half-filling)  = ", round(C_mb_half, digits=6))
    @test round(C_mb_half, digits=2) ≈ round(C_sp_lower, digits=2) atol = 0.1

    # 7b. Full filling: all bands occupied → total Chern number must be zero
    C_mb_full = many_body_Chern_number_Fukui_Hatsugai_Suzuki(tb; n_occ=2 * n_occ, nθ=21)
    println("C_mb (full filling)  = ", round(C_mb_full, digits=6))
    @test round(C_mb_full, digits=2) ≈ 0.0 atol = 0.1

    println("✓ All many-body Chern number checks passed.\n")
end


test(Haldane_Honeycomb())
test(CheckerBoard())


end