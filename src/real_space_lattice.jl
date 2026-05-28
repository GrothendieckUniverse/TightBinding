"Type alias: `Site:=Tuple{Vector{Int}, Int}`"
const Site = Tuple{Vector{Int},Int}

"""
Struct `Real_Space_Lattice{T}`
---
- fields:
    - `lattice_name::String`: name of lattice
    - `dim::Int`: dimension of lattice
    - `sample_size::Vector{Int}`
    - `cell_int_list::Vector{<:Vector}`: list of integer cell indices
    - `n_cell::Int`: number of unit cells
    - `brav_vec_list::Vector{<:Vector}`: list of bravais vectors for real-space lattice (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `cell_volume::T`: volume of the unit cell in real space
    - `n_sub::Int`: number of sublattices in each unit cell
    - `sub_crys_list::Vector{<:Vector}`: list of sublattice positions _in crystal coordinates_ (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `sub_name_list::Vector{String}`: list of sublattice names
    - `pbc_indicator::Vector{Bool}`: whether to apply periodic boundary condition in direction-i
    - `n_site::Int`: total number of sites in the lattice
    - `site_list::Vector{Site}`: list of site positions in each cell as `(cell_int, i_sub)`
    - `site_cart_list::Vector{<:Vector}`: list of site positions in cartesian coordinates (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `site_to_index_map::Dict{Site,Int}`: hashmap `(cell_int, i_sub) -> i_site`
    - `graph::Union{Nothing,Graphs.SimpleGraph}`: undirected _nearest-neighbor graph_ on the finite sample, built by comparing minimal Euclidean distances with PBC. `nothing` if symbolic entries prevent numerical construction. Use `Graphs.neighbors`, `Graphs.neighborhood`, `Graphs.gdistances`, `Graphs.bfs_tree` etc. to query.
"""
mutable struct Real_Space_Lattice{T}
    lattice_name::String
    dim::Int
    sample_size::Vector{Int}

    cell_int_list::Vector{Vector{Int}}
    n_cell::Int

    brav_vec_list::Vector{<:Vector} # bravais vectors for real-space lattice
    cell_volume::T

    n_sub::Int
    sub_crys_list::Vector{<:Vector} # sublattice positions in crystal coordinates
    sub_name_list::Vector{String}

    pbc_indicator::Vector{Bool} # whether to apply periodic boundary condition in direction-i

    n_site::Int
    site_list::Vector{Site} # site positions in each cell as `(cell_int, i_sub)`
    site_crys_list::Vector{<:Vector} # site positions in crystal coordinates
    site_cart_list::Vector{<:Vector} # site positions in cartesian coordinates
    site_to_index_map::Dict{Site,Int} # hashmap `(cell_int, i_sub) -> i_site`

    graph::Union{Nothing,Graphs.SimpleGraph}  # NN graph on the finite sample (minimal-Euclidean distance with PBC)
end


"""
Constructor for `Real_Space_Lattice`
---
- Named Args:
    - `brav_vec_list::Vector{<:Vector}`: list of bravais vectors for real-space lattice
    - `sample_size::Vector{Int}`: number of unit cells in each direction
    - `sub_crys_list::Vector{<:Vector}`: list of sublattice positions _in crystal coordinates_
    - `lattice_name::String`: name of lattice. If this is set to be `"square"`, `"honeycomb"`, `"kagome"`, `"Lieb"`, or `"dice"`, it will override the above three arguments with the corresponding default values.)
    - `pbc_indicator::Vector{Bool}`: whether to apply periodic boundary condition in direction-i
    - `allowed_bonds::Union{Nothing, Vector{Tuple{Int,Int}}}`: optional list of _allowed_ sublattice pairs `(sub_i, sub_j)` for graph construction. When `nothing` (default), all pairs are allowed (original Euclidean-distance algorithm). When provided, only edges between the specified sublattice pairs are filtered out. Indices are 1-based sublattice indices matching `sub_crys_list`.
"""
function initialize_real_space_lattice(;
    brav_vec_list::Vector{<:Vector}=[[1.0, 0.0], [0.0, 1.0]],
    sample_size::Vector{Int}=[2, 2],
    sub_crys_list::Vector{<:Vector}=[[0.0, 0.0]],
    lattice_name::String="",
    pbc_indicator::Vector{Bool}=[true, true],
    allowed_bonds::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
)::Real_Space_Lattice
    (brav_vec_list, sub_crys_list, allowed_bonds) = @match lattice_name begin
        "square" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], allowed_bonds)
        "honeycomb" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3]], allowed_bonds)
        "kagome" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]], allowed_bonds)
        "Lieb" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]], allowed_bonds)
        "dice" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3], [2 / 3, 2 / 3]], [(1, 2), (2, 3)])
        _ => (brav_vec_list, sub_crys_list, allowed_bonds)
    end
    dim = length(brav_vec_list)

    brav_vec_mat = reduce(hcat, brav_vec_list) # `hcat()` forces `brav_vec` to be stored in columns in `brav_vec_mat`
    cell_volume = abs(det(brav_vec_mat))

    @assert dim == 2 || dim == 3
    @assert length(brav_vec_list) == length(sample_size)
    n_sub = length(sub_crys_list)
    @assert n_sub >= 1 # at least one sublattice

    cell_int_list = Iterators.product([0:(Ni-1) for Ni in sample_size]...) .|> collect |> vec
    n_cell = length(cell_int_list)

    @assert all(length.(sub_crys_list) == [dim for _ in 1:n_sub])
    sub_name_list::Vector{String} = [string("A", i) for i in 1:n_sub] # the default name 

    site_list = [(cell_int, i_sub) for cell_int in cell_int_list for i_sub in 1:n_sub]
    n_site = length(site_list)
    site_crys_list = [cell_int + sub_crys_list[i_sub] for (cell_int, i_sub) in site_list]
    site_cart_list = [sum(brav_vec_list .* site_crys) for site_crys in site_crys_list]

    site_to_index_map = Dict(zip(site_list, 1:n_site))

    # Build nearest-neighbor graph by comparing minimal Euclidean distances.
    # Falls back to `nothing` for symbolic bravais vectors / sublattice positions.
    graph::Union{Nothing,Graphs.SimpleGraph} =
        try
            brav_vec_list_num = convert.(Vector{Float64}, brav_vec_list)
            site_crys_list = convert.(Vector{Float64}, site_crys_list)

            _build_nearest_neighbor_graph_by_Euclidean_distance(;
                site_crys_list=site_crys_list, site_list=site_list,
                brav_vec_list=brav_vec_list_num, sample_size=sample_size,
                pbc_indicator=pbc_indicator,
                allowed_bonds=allowed_bonds,
            )
        catch
            nothing
        end

    return Real_Space_Lattice(
        lattice_name,
        dim,
        sample_size,
        cell_int_list,
        n_cell,
        brav_vec_list,
        cell_volume,
        n_sub,
        sub_crys_list,
        sub_name_list,
        pbc_indicator,
        n_site,
        site_list,
        site_crys_list,
        site_cart_list,
        site_to_index_map,
        graph,
    )
end


"""
_In-place_ Wrap of Crystal Displacement `Δ_crys` with Respect to Boundary Conditions
---
The strategy is to apply periodic boundary conditions by wrapping the crystal displacement into the range `[-L/2, L/2]` for each direction where `pbc_indicator` is `true`.
- Args:
    - `Δ_crys::Vector{Float64}`: displacement in crystal coordinates
- Named Args:
    - `sample_size::Vector{Int}`: vector of sample sizes
    - `pbc_indicator::Vector{Bool}`: boundary conditions indicator
"""
@inline function _wrap_Δ_crys!(Δ_crys::Vector{Float64};
    sample_size::Vector{Int}=sample_size,
    pbc_indicator::Vector{Bool}=pbc_indicator
)
    for d in eachindex(Δ_crys) # loop over dimensions
        if pbc_indicator[d]
            # wrap Δc to [-L/2, L/2] where L = sample_size[d]
            Δ_crys[d] -= round(Δ_crys[d] / sample_size[d]) * sample_size[d]
        end
    end
    return Δ_crys
end

"""
Build the Nearest-Neighbor Graph by Minimal Euclidean Distance
---
For each site, we first search for the minimum distance between any two distinct sites, then connect pairs whose distance falls within that value.
- Named Args:
    - `site_crys_list::Vector{Vector{Float64}}`: site crystal coordinate list
    - `site_list::Vector{Site}`: site list with `(cell_int, i_sub)` for each site
    - `brav_vec_list::Vector{Vector{Float64}}`: bravais vectors
    - `sample_size::Vector{Int}`: sample size
    - `pbc_indicator::Vector{Bool}`: boundary conditions indicator
    - `allowed_bonds::Union{Nothing, Vector{Tuple{Int,Int}}}`: optional list of allowed sublattice pairs `(sub_i, sub_j)`. When provided, only edges between the specified sublattice pairs are considered. The pair is checked symmetrically: `(a,b)` allows both `a→b` and `b→a`. When `nothing`, all pairs are allowed.
"""
function _build_nearest_neighbor_graph_by_Euclidean_distance(;
    site_crys_list::Vector{Vector{Float64}},
    site_list::Vector{Site},
    brav_vec_list::Vector{Vector{Float64}},
    sample_size::Vector{Int},
    pbc_indicator::Vector{Bool},
    allowed_bonds::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
)::Graphs.SimpleGraph
    n_site = length(site_crys_list)

    # Build a set of allowed sublattice pairs (symmetric) for fast lookup
    allowed_pairs::Union{Nothing,Set{Tuple{Int,Int}}} =
        if isnothing(allowed_bonds)
            nothing
        else
            s = Set{Tuple{Int,Int}}()
            for (a, b) in allowed_bonds
                push!(s, (a, b))
                push!(s, (b, a))
            end
            s
        end

    @inline function _is_pair_allowed(i::Int, j::Int)::Bool
        isnothing(allowed_pairs) && return true
        sub_i = site_list[i][2]
        sub_j = site_list[j][2]
        return (sub_i, sub_j) in allowed_pairs
    end

    # Pass 1: find the nearest-neighbour distance (only among allowed pairs)
    nn_dist = Inf
    for i in 1:n_site
        for j in (i+1):n_site
            _is_pair_allowed(i, j) || continue

            Δ_crys = site_crys_list[j] - site_crys_list[i]
            _wrap_Δ_crys!(Δ_crys;
                sample_size=sample_size, pbc_indicator=pbc_indicator
            )

            d = sum(brav_vec_list .* Δ_crys) |> norm
            if d != 0 && d < nn_dist
                nn_dist = d
            end
        end
    end

    # Pass 2: build the graph
    g = Graphs.SimpleGraph(n_site)
    if isinf(nn_dist)
        return g  # no edges (e.g. single-site lattice)
    end

    threshold = nn_dist * (1.0 + 1.0E-10)
    for i in 1:n_site
        for j in (i+1):n_site
            _is_pair_allowed(i, j) || continue

            Δ_crys = site_crys_list[j] - site_crys_list[i]
            _wrap_Δ_crys!(Δ_crys;
                sample_size=sample_size, pbc_indicator=pbc_indicator
            )

            d = sum(brav_vec_list .* Δ_crys) |> norm
            if d <= threshold
                Graphs.add_edge!(g, i, j)
            end
        end
    end
    return g
end



"""
    plot_real_space_lattice(lattice::Real_Space_Lattice; kwargs...) -> Figure

Plot the 2D real-space lattice from its nearest-neighbor graph.

- **Bulk edges** (minimum-image displacement ≡ raw displacement): solid black.
- **Wrapped edges** (PBC wrapping shortens the bond): dashed, 42% transparency, with ghost sites
  plotted at the *unwrapped* positions (outside the sample region) at 42% transparency.
- Every site (bulk and ghost) is labelled with its linear index from `site_list`.
- The unit cell (parallelogram spanned by the bravais vectors) is outlined, with
  arrowed bravais basis vectors.

Only supports 2D lattices.
"""
function plot_real_space_lattice(
    lattice::Real_Space_Lattice;
)::CairoMakie.Figure
    dim = lattice.dim
    @assert dim == 2 "plot_real_space_lattice currently only supports 2D lattices"

    # --- convert everything to Float64 ----------------------------------------
    brav_vec = [Float64.(collect(v)) for v in lattice.brav_vec_list]
    site_crys = [Float64.(c) for c in lattice.site_crys_list]
    n_site = lattice.n_site
    L = Float64.(lattice.sample_size)

    # crystal → Cartesian
    to_cart(c) = sum(brav_vec .* c)
    site_cart = to_cart.(site_crys)

    # --- figure & axis --------------------------------------------------------
    default_fig_size = [1200, 1200]
    scaled_fig_size = sqrt(reduce(*, lattice.sample_size)) / 6 * default_fig_size |> Tuple
    @info "scaled_fig_size: $scaled_fig_size"
    fig = CairoMakie.Figure(size=scaled_fig_size, backgroundcolor=:transparent)
    ax = CairoMakie.Axis(fig[1, 1];
        aspect=CairoMakie.DataAspect()
    )

    # --- plot edges -----------------------------------------------------------
    scaled_marker_size = 20
    # scaled_marker_size *= sqrt(reduce(*, lattice.sample_size)) / 8
    @info "scaled_marker_size: $scaled_marker_size"
    scaled_font_size = 10
    # scaled_font_size *= sqrt(reduce(*, lattice.sample_size)) / 10
    @info "scaled_font_size: $scaled_font_size"
    if !isnothing(lattice.graph)
        g = lattice.graph
        alpha_wrap = 0.42

        for e in Graphs.edges(g)
            i = Graphs.src(e)
            j = Graphs.dst(e)
            i < j || continue   # each undirected edge once

            Δc_raw = site_crys[j] - site_crys[i]
            Δc_min = copy(Δc_raw)
            Δc_min = _wrap_Δ_crys!(Δc_min;
                sample_size=lattice.sample_size, pbc_indicator=lattice.pbc_indicator
            )

            xi = [site_cart[i][1], site_cart[j][1]]
            yi = [site_cart[i][2], site_cart[j][2]]

            if norm(Δc_min - Δc_raw) < 1e-10
                # ---- bulk edge -------------------------------------------------
                CairoMakie.lines!(ax, xi, yi;
                    color=(:black, 1.0), linewidth=2)
            else
                # ---- wrapped edge: bulk ↔ ghost (not bulk ↔ bulk) -------------
                # ghost positions (unwrapped, outside sample region)
                ghost_j_crys = site_crys[i] + Δc_min
                ghost_i_crys = site_crys[j] - Δc_min
                ghost_j_cart = to_cart(ghost_j_crys)
                ghost_i_cart = to_cart(ghost_i_crys)

                # sublattice indices for ghost colors
                i_sub_i = lattice.site_list[i][2]
                i_sub_j = lattice.site_list[j][2]
                color_i = CairoMakie.Cycled(i_sub_i)
                color_j = CairoMakie.Cycled(i_sub_j)

                # faded line from bulk i → ghost j
                CairoMakie.lines!(ax,
                    [site_cart[i][1], ghost_j_cart[1]],
                    [site_cart[i][2], ghost_j_cart[2]];
                    color=color_j, alpha=alpha_wrap, linewidth=2, linestyle=:dash)

                # faded line from bulk j → ghost i
                CairoMakie.lines!(ax,
                    [site_cart[j][1], ghost_i_cart[1]],
                    [site_cart[j][2], ghost_i_cart[2]];
                    color=color_i, alpha=alpha_wrap, linewidth=2, linestyle=:dash)

                # ghost sites
                CairoMakie.scatter!(ax, ghost_j_cart[1], ghost_j_cart[2];
                    color=color_j, alpha=alpha_wrap, markersize=scaled_marker_size)
                CairoMakie.scatter!(ax, ghost_i_cart[1], ghost_i_cart[2];
                    color=color_i, alpha=alpha_wrap, markersize=scaled_marker_size)

                # ghost labels (same index as the corresponding bulk site)
                CairoMakie.text!(ax, ghost_j_cart[1], ghost_j_cart[2];
                    text="$(j)", color=(:white, 1.0),
                    fontsize=scaled_font_size, align=(:center, :center))
                CairoMakie.text!(ax, ghost_i_cart[1], ghost_i_cart[2];
                    text="$(i)", color=(:white, 1.0),
                    fontsize=scaled_font_size, align=(:center, :center))
            end
        end
    end

    # --- plot bulk sites & labels ---------------------------------------------
    for i_site in 1:n_site
        (cell_int, i_sub) = lattice.site_list[i_site]
        x = site_cart[i_site][1]
        y = site_cart[i_site][2]
        CairoMakie.scatter!(ax, x, y;
            color=CairoMakie.Cycled(i_sub), markersize=scaled_marker_size)
        CairoMakie.text!(ax, x, y;
            text="$(i_site)", color=(:white, 1.0),
            fontsize=scaled_font_size, align=(:center, :center))
    end


    # --- unit cell (1st BZ in real space) & bravais arrows -------------------
    a1 = brav_vec[1]
    a2 = brav_vec[2]
    origin = [0.0, 0.0]

    # unit-cell parallelogram: origin → a1 → a1+a2 → a2 → origin
    cell_cx = [origin[1], a1[1], a1[1] + a2[1], a2[1], origin[1]]
    cell_cy = [origin[2], a1[2], a1[2] + a2[2], a2[2], origin[2]]
    CairoMakie.lines!(ax, cell_cx, cell_cy;
        color=(:black, 0.5), linewidth=2, linestyle=:dashdot
    )

    # bravais basis vectors
    CairoMakie.arrows2d!(ax,
        [origin[1]], [origin[2]], [a1[1]], [a1[2]];
        color=(:tomato, 0.64), shaftwidth=2.4, tipwidth=12, tiplength=18
    )
    CairoMakie.arrows2d!(ax,
        [origin[1]], [origin[2]], [a2[1]], [a2[2]];
        color=(:tomato, 0.64), shaftwidth=2.4, tipwidth=12, tiplength=18
    )



    CairoMakie.display(fig)
    return fig
end