const test = 1


"""
Struct `Real_Space_TightBinding_Model{T,U}`
---
for real-space hoppings.
- Fields:
    - `lattice::Real_Space_Lattice{T}`: the underlying real-space lattice
    - `model_name::String`: name of the tight-binding model
    - `input_hopping_map::Dict{Tuple{Site,Site},U}`: hashmap `(cell_int, i_sub) -> t`. This includes hoppings within and across unit cells. Hermicity is already implemented when building the `input_hopping_map`
    - `full_hopping_map::Dict{Tuple{Site,Site},U}`: hashmap `(cell_int, i_sub) -> t`. This includes and expands ALL hoppings of the model. Translation symmetry is already implemented for bulk hopping terms
    - `H_hop::Function`: real-space hopping Hamiltonian (for edge-mode calculation)
"""
mutable struct Real_Space_TightBinding_Model{T,U}
    lattice::Real_Space_Lattice{T}
    model_name::String

    input_hopping_map::Dict{Tuple{Site,Site},U} # hashmap `(cell_int, i_sub) -> t`. This includes hoppings within and across unit cells. Hermicity is already implemented when building the `input_hopping_map`
    full_hopping_map::Dict{Tuple{Site,Site},U} # hashmap `(cell_int, i_sub) -> t`. This includes and expands ALL hoppings of the model. Translation symmetry is already implemented for bulk hopping terms

    H_hop::Function # real-space hopping Hamiltonian (for edge-mode calculation)
end

"""
Constructor for `Real_Space_TightBinding_Model`
---
- Args:
    - `lattice::Real_Space_Lattice{T}`: the underlying real-space lattice
- Named Args:
    - `model_name::String`: name of the tight-binding model
"""
function initialize_real_space_tightbinding_model(lattice::Real_Space_Lattice{T};
    model_name::String="",
)::Real_Space_TightBinding_Model where T
    @assert length(lattice.pbc_indicator) == lattice.dim

    input_hopping_map = Dict{Tuple{Site,Site},Number}()
    full_hopping_map = Dict{Tuple{Site,Site},Number}()

    function H_hop end

    return Real_Space_TightBinding_Model(
        lattice,
        model_name,
        input_hopping_map,
        full_hopping_map,
        H_hop
    )
end




"""
Manual Adding of Hopping Terms to `Real_Space_TightBinding_Model`
---
- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the real-space tight-binding model to which the hopping term will be added
    - `input_hopping_term::Pair{Tuple{Site,Site},T}`: the input hopping term in the form of `((cell_from, sub_from), (cell_to, sub_to)) => hopping_strength`. Note: it also applies to chemical potentials, when `cell_from == cell_to` and `sub_from == sub_to`.
- Named Args:
    - `is_hermitian::Bool=true`: whether to add the Hermitian conjugate of the input hopping term to the model
"""
function add_hopping_term!(
    tb_model::Real_Space_TightBinding_Model,
    input_hopping_term::Pair{Tuple{Site,Site},T};
    is_hermitian::Bool=true,
) where T
    n_sub = tb_model.lattice.n_sub

    # update the `input_hopping_map` (if is hermitian, also add the hermitian conjugate)
    let ((site_from, site_to), hopping_strength) = input_hopping_term
        (cell_from, sub_from) = site_from
        (cell_to, sub_to) = site_to

        # check the validity of input hopping term
        @assert sub_from in 1:n_sub && sub_to in 1:n_sub "The input sublattice indices for `input_hopping_map`=$(input_hopping_term) is invalid for `sample_size`=$(tb_model.lattice.sample_size)!"
        # @assert all(0 .<= cell_from .<= (tb_model.lattice.sample_size .- 1)) && all(0 .<= cell_to .<= (tb_model.lattice.sample_size .- 1)) "The input cell indices for `input_hopping_map`=$(input_hopping_term) is invalid for `sample_size`=$(tb_model.lattice.sample_size)!"

        if haskey(tb_model.input_hopping_map, (site_from, site_to))
            @error "The input hopping term `$(input_hopping_term)` already exists in `input_hopping_map`!\n --- The old hopping term will be overwritten, rather than being added to the existing value!"
        end
        tb_model.input_hopping_map[(site_from, site_to)] = hopping_strength

        if is_hermitian
            if haskey(tb_model.input_hopping_map, (site_to, site_from))
                tb_model.input_hopping_map[(site_to, site_from)] += conj(hopping_strength)
            else
                tb_model.input_hopping_map[(site_to, site_from)] = conj(hopping_strength)
            end
        end
    end

    # rebuild the `full_hopping_map` using translation symmetry
    let ((site_from, site_to), hopping_strength) = input_hopping_term
        (cell_from, sub_from) = site_from
        (cell_to, sub_to) = site_to

        cell_shift = cell_to - cell_from
        for new_cell_from in tb_model.lattice.cell_int_list
            new_cell_to = new_cell_from + cell_shift
            @inbounds for i in eachindex(tb_model.lattice.pbc_indicator)
                if tb_model.lattice.pbc_indicator[i] # handle periodic boundary condition
                    new_cell_to[i] = mod(new_cell_to[i], tb_model.lattice.sample_size[i])
                end
            end

            new_site_from = (new_cell_from, sub_from)
            new_site_to = (new_cell_to, sub_to)

            if haskey(tb_model.full_hopping_map, (new_site_from, new_site_to))
                tb_model.full_hopping_map[(new_site_from, new_site_to)] += hopping_strength
            else
                tb_model.full_hopping_map[(new_site_from, new_site_to)] = hopping_strength
            end

            if is_hermitian
                if haskey(tb_model.full_hopping_map, (new_site_to, new_site_from))
                    tb_model.full_hopping_map[(new_site_to, new_site_from)] += conj(hopping_strength)
                else
                    tb_model.full_hopping_map[(new_site_to, new_site_from)] = conj(hopping_strength)
                end
            end
        end
    end
    return nothing
end



"""
Add One Hopping Term by Updating `tb_model.full_hopping_map`
---
- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the real-space tight-binding model to which the hopping term will be added
    - `hopping_term::Pair{Tuple{Site,Site},T}`: the hopping term in the form of `((cell_from, sub_from), (cell_to, sub_to)) => hopping_strength`. Note: it also applies to chemical potentials, when `cell_from == cell_to` and `sub_from == sub_to`.
- Named Args:
    - `is_hermitian::Bool=true`: whether to add the Hermitian conjugate of the input hopping term to the model
"""
function add_hopping_term_to_full_hopping_map!(
    tb_model::Real_Space_TightBinding_Model,
    hopping_term::Pair{Tuple{Site,Site},T};
    is_hermitian::Bool=true,
) where T
    ((site_from, site_to), hopping_strength) = hopping_term

    if haskey(tb_model.full_hopping_map, (site_from, site_to))
        @info "The hopping term `$(hopping_term)` already exists in `full_hopping_map`\n --- The new hopping term will be added to the existing value!"
        tb_model.full_hopping_map[(site_from, site_to)] += hopping_strength
    else
        tb_model.full_hopping_map[(site_from, site_to)] = hopping_strength
    end

    if is_hermitian
        if haskey(tb_model.full_hopping_map, (site_to, site_from))
            tb_model.full_hopping_map[(site_to, site_from)] += conj(hopping_strength)
        else
            tb_model.full_hopping_map[(site_to, site_from)] = conj(hopping_strength)
        end
    end

    return nothing
end







# ------------------------------------------------------------------------
# Graph-distance-based hopping generation
# ------------------------------------------------------------------------
"""
    add_hoppings_by_graph_distance!(tb_model, distance, amplitude; is_hermitian=true)

Add hoppings between ALL pairs of sites separated by graph distance `distance` on the underlying lattice graph, with a uniform complex `amplitude`.

- `distance = 0`: on-site terms (chemical potential).
- `distance = 1`: nearest-neighbor hoppings (one graph edge).
- `distance ≥ 2`: hoppings between sites connected by `distance` graph edges.

The graph must be available (`tb_model.lattice.graph` is not `nothing`).

# Examples
```julia
# Isotropic NN hopping on any lattice
add_hoppings_by_graph_distance!(tb_model, 1, -1.0)

# On-site chemical potential
add_hoppings_by_graph_distance!(tb_model, 0, 0.5; is_hermitian=false)

# Isotropic NNN hopping
add_hoppings_by_graph_distance!(tb_model, 2, -0.3)
```
"""
function add_hoppings_by_graph_distance!(
    tb_model::Real_Space_TightBinding_Model,
    graph_distance::Int,
    amplitude::Number;
    is_hermitian::Bool=true,
)
    g = tb_model.lattice.graph
    if isnothing(g)
        error("The lattice graph is `nothing` — cannot traverse by graph distance. " *
              "Ensure the lattice was built with numerical bravais vectors.")
    end
    @assert graph_distance >= 0 "Graph distance must be ≥ 0, got $graph_distance."


    n_site = tb_model.lattice.n_site

    if graph_distance == 0
        # on-site: each site to itself
        for i_site in 1:n_site
            site = tb_model.lattice.site_list[i_site]
            add_hopping_term_to_full_hopping_map!(tb_model, (site, site) => amplitude; is_hermitian=false)
        end
        return nothing
    end

    if graph_distance ≥ 1
        for i_site in 1:n_site
            site_i = tb_model.lattice.site_list[i_site]
            dists = Graphs.gdistances(g, i_site)
            j_iter = (i_site+1):n_site
            for j_site in j_iter
                j_site == i_site && continue
                if dists[j_site] == graph_distance
                    site_j = tb_model.lattice.site_list[j_site]
                    add_hopping_term_to_full_hopping_map!(tb_model, (site_i, site_j) => amplitude; is_hermitian=is_hermitian)
                end
            end
        end
    end

    return nothing
end


"""
    add_hoppings_by_graph_distance!(tb_model, distance, amplitude_func; is_hermitian=true)

Add hoppings between ALL pairs of sites separated by graph distance `distance`, where the complex amplitude is computed by `amplitude_func(i_site, j_site)`. Here `amplitude_func` receives two linear site indices and must return a `Number`.

This enables **direction-dependent** hoppings such as the complex NNN hoppings in the Haldane model.
"""
function add_hoppings_by_graph_distance!(
    tb_model::Real_Space_TightBinding_Model,
    graph_distance::Int,
    amplitude_func::Function;
    is_hermitian::Bool=true,
)
    g = tb_model.lattice.graph
    if isnothing(g)
        error("The lattice graph is `nothing` — cannot traverse by graph distance. " *
              "Ensure the lattice was built with numerical bravais vectors.")
    end
    @assert graph_distance >= 0 "Graph distance must be ≥ 0, got $graph_distance."

    n_site = tb_model.lattice.n_site

    if graph_distance == 0
        # on-site: each site to itself
        for i_site in 1:n_site
            site = tb_model.lattice.site_list[i_site]
            amp = amplitude_func(i_site, i_site)
            if amp != 0
                add_hopping_term_to_full_hopping_map!(tb_model, (site, site) => amp; is_hermitian=false)
            end
        end
        return nothing
    end

    if graph_distance ≥ 1
        for i_site in 1:n_site
            site_i = tb_model.lattice.site_list[i_site]
            dists = Graphs.gdistances(g, i_site)
            j_iter = (i_site+1):n_site
            for j_site in j_iter
                j_site == i_site && continue
                if dists[j_site] == graph_distance
                    amp = amplitude_func(i_site, j_site)
                    if amp != 0
                        site_j = tb_model.lattice.site_list[j_site]
                        add_hopping_term_to_full_hopping_map!(tb_model, (site_i, site_j) => amp; is_hermitian=is_hermitian)
                    end
                end
            end
        end
    end

    return nothing
end

"""
Helper Method to Construct Haldane Complex NNN Hoppings for Graph as a Honeycomb Lattice
---
The strategy is to first search for the common nearest neighbor (which must be single vertex for honeycomb lattice), and then compute the chirality `ν = sign((𝐫_𝐤 -  𝐫_𝐢) × ( 𝐫_𝐣 - 𝐫_𝐤)) = {±1}`.
- Args:
    - `i_site::Int`: linear index of site-i
    - `j_site::Int`: linear index of site-j
- Named Args:
    - `t2::Number=0.3`: NNN hopping amplitude
    - `φ::Number=π/2`: Haldane phase
    - `tb_model::Real_Space_TightBinding_Model`
- Returns:
    - `ComplexF64`: the complex NNN hopping amplitude `t2 * exp(i * φ * ν)`
"""
function haldane_nnn_hopping_amplitude(
    i_site::Int,
    j_site::Int;
    t2::Number=0.3,
    φ::Number=π / 2,
    tb_model::Real_Space_TightBinding_Model,
)::ComplexF64
    l = tb_model.lattice
    @assert l.lattice_name == "honeycomb" "Haldane NNN hopping is defined for the honeycomb lattice only."
    g = l.graph
    @assert g !== nothing "The lattice graph is `nothing` — cannot compute Haldane NNN hopping."

    # find the unique intermediate site connecting i and j (distance 2)
    i_site_nn_sites = Set(Graphs.neighbors(g, i_site))
    j_site_nn_sites = Set(Graphs.neighbors(g, j_site))
    common = intersect(i_site_nn_sites, j_site_nn_sites)
    @assert length(common) == 1 "Sites $i_site and $j_site do not share exactly one common neighbor (graph distance ≠ 2?)."
    k_site = first(common)

    # cross product of the two NN hops determines the chirality
    site_i_crys = l.site_crys_list[i_site]
    site_j_crys = l.site_crys_list[j_site]
    site_k_crys = l.site_crys_list[k_site]

    Δ_ik_wrapped = _wrap_Δ_crys!(site_k_crys - site_i_crys; sample_size=l.sample_size, pbc_indicator=l.pbc_indicator)
    Δ_kj_wrapped = _wrap_Δ_crys!(site_j_crys - site_k_crys; sample_size=l.sample_size, pbc_indicator=l.pbc_indicator)

    Δ_ik_cart = sum(l.brav_vec_list .* Δ_ik_wrapped)
    Δ_kj_cart = sum(l.brav_vec_list .* Δ_kj_wrapped)

    cross_2D = Δ_ik_cart[1] * Δ_kj_cart[2] - Δ_ik_cart[2] * Δ_kj_cart[1]
    chirality = cross_2D > 0 ? 1 : -1

    return t2 * exp(chirality * im * φ)
end


"""
Plot Hoppings and Sites of the `Real_Space_TightBinding_Model`
---
- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the real-space tight-binding model to be visualized
---
Draws the lattice background (sites + faint graph edges) and overlays hoppings as **curved annotated arcs** so they do not overlap with the background.

- Bulk hoppings: solid steelblue arcs with amplitude labels.
- Wrapped hoppings (across the PBC boundary): dashed tomato arcs with faded labels.
- On-site terms are annotated as text next to the site marker.

Only supports 2D lattices with numeric `site_cart_list`.
"""
function plot_real_space_tightbinding_model(
    tb_model::Real_Space_TightBinding_Model;
)::CairoMakie.Figure
    l = tb_model.lattice
    dim = l.dim
    @assert dim == 2 "Only 2D lattices are supported for plotting."
    @assert eltype(first(l.site_cart_list)) <: Real "Cartesian coordinates must be numeric for plotting."

    # --- convert to Float64 ---------------------------------------------------
    brav_vec = [Float64.(collect(v)) for v in l.brav_vec_list]
    site_crys = [Float64.(c) for c in l.site_crys_list]
    site_cart = [Float64.(c) for c in l.site_cart_list]

    to_cart(c) = sum(brav_vec .* c)

    # --- figure & axis --------------------------------------------------------
    default_fig_size = [1200, 1200]
    scaled_fig_size = sqrt(reduce(*, l.sample_size)) / 6 * default_fig_size |> Tuple
    @info "scaled_fig_size: $scaled_fig_size"

    fig = CairoMakie.Figure(size=scaled_fig_size, backgroundcolor=:transparent)
    ax = CairoMakie.Axis(fig[1, 1];
        aspect=CairoMakie.DataAspect()
    )

    # --- global plot scales ---------------------------------------------------
    site_markersize = 20
    ghost_markersize = 20
    site_fontsize = 10
    ghost_fontsize = 10

    # Registry for ghost sites.
    #
    # Key = (physical site index, rounded ghost crystal coordinate).
    # This distinguishes, for example, site j translated to different periodic images,
    # but prevents repeatedly drawing the same ghost image.
    ghost_site_registry = Dict{Any,NamedTuple}()

    # --- precompute graph distances for color-by-graph-distance ---------------
    dist_cache = Dict{Int,Vector{Int}}()
    get_dists(i_site) =
        if haskey(dist_cache, i_site)
            dist_cache[i_site]
        else
            d = isnothing(l.graph) ? zeros(Int, l.n_site) : Graphs.gdistances(l.graph, i_site)
            dist_cache[i_site] = d
            d
        end

    # -------------------------------------------------------------------------
    # 1. Draw lattice background edges.
    #    Do NOT draw ghost sites here. Only register them.
    # -------------------------------------------------------------------------
    if !isnothing(l.graph)
        for e in Graphs.edges(l.graph)
            i = Graphs.src(e)
            j = Graphs.dst(e)
            i < j || continue

            c_i = site_crys[i]
            c_j = site_crys[j]
            Δc_raw = c_j - c_i
            Δc_wrapped = _wrap_Δ_crys!(copy(Δc_raw);
                sample_size=l.sample_size,
                pbc_indicator=l.pbc_indicator,
            )
            is_wrapped = norm(Δc_wrapped - Δc_raw) > 1e-10

            if !is_wrapped
                CairoMakie.lines!(ax,
                    [site_cart[i][1], site_cart[j][1]],
                    [site_cart[i][2], site_cart[j][2]];
                    color=(:black, 1.0),
                    linewidth=2,
                )
            else
                ghost_j_crys = c_i + Δc_wrapped
                ghost_i_crys = c_j - Δc_wrapped
                ghost_j_cart = to_cart(ghost_j_crys)
                ghost_i_cart = to_cart(ghost_i_crys)

                (_, i_sub_i) = l.site_list[i]
                (_, i_sub_j) = l.site_list[j]

                CairoMakie.lines!(ax,
                    [site_cart[i][1], ghost_j_cart[1]],
                    [site_cart[i][2], ghost_j_cart[2]];
                    color=Makie.Cycled(i_sub_j),
                    alpha=0.28,
                    linewidth=2,
                    linestyle=:dash,
                )

                CairoMakie.lines!(ax,
                    [site_cart[j][1], ghost_i_cart[1]],
                    [site_cart[j][2], ghost_i_cart[2]];
                    color=Makie.Cycled(i_sub_i),
                    alpha=0.28,
                    linewidth=2,
                    linestyle=:dash,
                )

                key_j = (j, Tuple(round.(ghost_j_crys; digits=10)))
                key_i = (i, Tuple(round.(ghost_i_crys; digits=10)))

                ghost_site_registry[key_j] = (
                    i_site=j,
                    i_sub=i_sub_j,
                    cart=ghost_j_cart,
                    alpha=0.28,
                )

                ghost_site_registry[key_i] = (
                    i_site=i,
                    i_sub=i_sub_i,
                    cart=ghost_i_cart,
                    alpha=0.28,
                )
            end
        end
    end

    # -------------------------------------------------------------------------
    # 2. Draw hopping arcs.
    #    Again: do NOT draw ghost sites here. Only register wrapped targets.
    # -------------------------------------------------------------------------

    hopping_line_style(amplitude) = imag(amplitude) == 0 ? :solid : :dot

    for ((site_from, site_to), amplitude) in tb_model.full_hopping_map
        i = l.site_to_index_map[site_from]
        j = l.site_to_index_map[site_to]

        p_i = Makie.Point2f(site_cart[i][1], site_cart[i][2])
        p_j = Makie.Point2f(site_cart[j][1], site_cart[j][2])

        if i == j
            # on-site: large-angle self-loop via cubic Bézier
            scale = sqrt(brav_vec[1][1]^2 + brav_vec[1][2]^2)
            offset1 = Makie.Point2f(scale * 0.5, scale * 0.5)
            ctrl1 = Makie.Point2f(p_i[1] + offset1[1], p_i[2] + offset1[2])
            ctrl2 = Makie.Point2f(p_i[1] + offset1[2], p_i[2] - offset1[1])

            bp_self = Makie.BezierPath([
                Makie.MoveTo(p_i),
                Makie.CurveTo(ctrl1, ctrl2, p_i),
            ])

            CairoMakie.lines!(ax, bp_self;
                color=(:darkgreen, 0.7),
                linewidth=1,
                linestyle=hopping_line_style(amplitude),
            )

            adir = Makie.Point2f(p_i[1] - ctrl2[1], p_i[2] - ctrl2[2])
            anrm = sqrt(adir[1]^2 + adir[2]^2)
            anrm > 0 || continue

            ud = Makie.Point2f(adir[1] / anrm, adir[2] / anrm)
            alen = scale * 0.12
            astart = Makie.Point2f(p_i[1] - ud[1] * alen, p_i[2] - ud[2] * alen)

            CairoMakie.arrows2d!(ax,
                [astart[1]], [astart[2]],
                [ud[1] * alen], [ud[2] * alen];
                color=(:darkgreen, 0.7),
                shaftlength=0.0,
                shaftwidth=1,
                tipwidth=7.2,
                tiplength=16,
                normalize=false,
            )
        else
            # classify bulk vs wrapped
            c_i = site_crys[i]
            c_j = site_crys[j]
            Δc_raw = c_j - c_i
            Δc_wrapped = _wrap_Δ_crys!(copy(Δc_raw);
                sample_size=l.sample_size,
                pbc_indicator=l.pbc_indicator,
            )
            is_wrapped = norm(Δc_wrapped - Δc_raw) > 1e-10

            graph_d = get_dists(i)[j]
            palette = Makie.wong_colors()
            hop_color_plain = palette[mod1(graph_d, length(palette))]
            alpha = is_wrapped ? 0.28 : 0.64

            if is_wrapped
                ghost_crys = c_i + Δc_wrapped
                ghost_cart = to_cart(ghost_crys)
                p_target = Makie.Point2f(ghost_cart[1], ghost_cart[2])

                (_, i_sub_j) = l.site_list[j]
                key_j = (j, Tuple(round.(ghost_crys; digits=10)))

                ghost_site_registry[key_j] = (
                    i_site=j,
                    i_sub=i_sub_j,
                    cart=ghost_cart,
                    alpha=0.28,
                )
            else
                p_target = p_j
            end

            # cubic Bézier curve with midpoint perpendicular offset
            dir = p_target - p_i
            nrm = sqrt(dir[1]^2 + dir[2]^2)
            nrm == 0 && continue

            perp = Makie.Point2f(-dir[2] / nrm, dir[1] / nrm)
            bend = 0.1 * nrm
            ctrl = Makie.Point2f(
                (p_i[1] + p_target[1]) / 2 + bend * perp[1],
                (p_i[2] + p_target[2]) / 2 + bend * perp[2],
            )

            bp = Makie.BezierPath([
                Makie.MoveTo(p_i),
                Makie.CurveTo(ctrl, ctrl, p_target),
            ])

            CairoMakie.lines!(ax, bp;
                color=hop_color_plain,
                alpha=alpha,
                linewidth=1,
                linestyle=hopping_line_style(amplitude),
            )

            # arrowhead at target
            arrow_dir = Makie.Point2f(p_target[1] - ctrl[1], p_target[2] - ctrl[2])
            anrm = sqrt(arrow_dir[1]^2 + arrow_dir[2]^2)
            anrm > 0 || continue

            unit_dir = Makie.Point2f(arrow_dir[1] / anrm, arrow_dir[2] / anrm)
            arrow_len = 0.12
            arrow_start = Makie.Point2f(
                p_target[1] - unit_dir[1] * arrow_len,
                p_target[2] - unit_dir[2] * arrow_len,
            )

            CairoMakie.arrows2d!(ax,
                [arrow_start[1]], [arrow_start[2]],
                [unit_dir[1] * arrow_len], [unit_dir[2] * arrow_len];
                color=hop_color_plain,
                alpha=alpha,
                shaftlength=0.0,
                shaftwidth=1.0,
                tipwidth=7.2,
                tiplength=16,
                normalize=false,
            )
        end
    end

    # -------------------------------------------------------------------------
    # 3. Draw all ghost sites exactly once.
    # -------------------------------------------------------------------------
    for (_, ghost_data) in ghost_site_registry
        i_site = ghost_data.i_site
        i_sub = ghost_data.i_sub
        ghost_cart = ghost_data.cart
        alpha = ghost_data.alpha

        CairoMakie.scatter!(ax,
            ghost_cart[1], ghost_cart[2];
            color=Makie.Cycled(i_sub),
            alpha=alpha,
            markersize=ghost_markersize,
        )

        CairoMakie.text!(ax,
            ghost_cart[1], ghost_cart[2];
            text="$(i_site)",
            color=(:white, 0.75),
            fontsize=ghost_fontsize,
            align=(:center, :center),
        )
    end

    # -------------------------------------------------------------------------
    # 4. Draw physical bulk sites exactly once, on top of all edges/hoppings.
    # -------------------------------------------------------------------------
    for i_site in 1:l.n_site
        (_, i_sub) = l.site_list[i_site]
        x, y = site_cart[i_site]

        CairoMakie.scatter!(ax, x, y;
            color=Makie.Cycled(i_sub),
            alpha=0.82,
            markersize=site_markersize,
        )

        CairoMakie.text!(ax, x, y;
            text="$(i_site)",
            color=(:white, 1.0),
            fontsize=site_fontsize,
            align=(:center, :center),
        )
    end

    # --- unit cell & bravais arrows ------------------------------------------
    a1 = brav_vec[1]
    a2 = brav_vec[2]
    origin = [0.0, 0.0]

    cell_cx = [origin[1], a1[1], a1[1] + a2[1], a2[1], origin[1]]
    cell_cy = [origin[2], a1[2], a1[2] + a2[2], a2[2], origin[2]]

    CairoMakie.lines!(ax, cell_cx, cell_cy;
        color=(:black, 0.5),
        linewidth=2,
        linestyle=:dashdot,
    )

    CairoMakie.arrows2d!(ax,
        [origin[1]], [origin[2]], [a1[1]], [a1[2]];
        color=(:tomato, 0.64),
        shaftwidth=2.4,
        tipwidth=12,
        tiplength=18,
    )

    CairoMakie.arrows2d!(ax,
        [origin[1]], [origin[2]], [a2[1]], [a2[2]];
        color=(:tomato, 0.64),
        shaftwidth=2.4,
        tipwidth=12,
        tiplength=18,
    )

    CairoMakie.display(fig)
    return fig
end
