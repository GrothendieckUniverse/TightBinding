"""
Plot Band Structure
---
with automatic scaling of `k_point_xs` based on each cartesian length of the k-path segments
- Args:
    - `Hk_crys::Function`: the k-space Hamiltonian function with `k_crys` input
    - `k_data::Uniform_Grids`: the k-space uniform grids
- Named Args:
    - `k_path::Vector{Vector{Float64}}`: a list turning k-points in crystal coordinates
    - `k_path_name_list::Vector{String}`: the names for turning points in `k_path`
    - `nband_range::Union{UnitRange{Int},Vector{Int}}`: the range or list of band indices to plot
    - `nk::Int`: number of k-points per path for plot
"""
function plot_bands(
    Hk_crys, k_data::Uniform_Grids;
    k_path::Vector{<:Vector{<:Real}},
    k_path_name_list::Vector{String}=Vector{String}(),
    nband_range::Union{UnitRange{Int},Vector{Int}}=1:1,
    nk::Int=30
)::CairoMakie.Figure
    dim = k_data.dim
    @assert length(k_path) >= 2 "The input `k_path` must contain at least two k-points to form a path!"
    @assert all(length(k_crys) == dim for k_crys in k_path) "Every k-point in `k_path` must have the same dimension as the k-space lattice!"
    if !isempty(k_path_name_list)
        @assert length(k_path_name_list) == length(k_path) "Every k-point in `k_path` must have a corresponding name in `k_path_name_list`!"
    end

    # inline function to prepare k-point list for band plot
    function prepare_band_plot_data(;
        k_path::Vector{Vector{Float64}}, nk::Int=20
    )
        k_crys_list = Vector{Vector{Float64}}()
        vline_pos_list = Vector{Float64}() # the vertical line positions for plot
        k_point_xs = Vector{Float64}() # the x-coordinates of k-points for plot

        push!(vline_pos_list, 0.0) # the first vline
        push!(k_point_xs, 0.0) # the first k-point x-coordinate

        for k_path_id in 1:(length(k_path)-1)
            k_head_crys = k_path[k_path_id]
            k_tail_crys = k_path[k_path_id+1]

            k_head_cart = sum(k_head_crys .* k_data.basis_vec_list)
            k_tail_cart = sum(k_tail_crys .* k_data.basis_vec_list)
            δk_cart = norm(k_tail_cart - k_head_cart) / nk
            for i in 0:(nk-1)
                push!(k_crys_list, k_head_crys .+ (k_tail_crys .- k_head_crys) .* (i / nk))
                push!(k_point_xs, k_point_xs[end] + δk_cart)
            end
            push!(vline_pos_list, k_point_xs[end]) # the last vline
        end
        return (k_crys_list, vline_pos_list, k_point_xs)
    end
    (k_crys_list, vline_pos_list, k_point_xs) = prepare_band_plot_data(; k_path=k_path, nk=nk)

    fig = CairoMakie.Figure(size=(300, 300), backgroundcolor=:transparent)

    vline_ticks = if !isempty(k_path_name_list)
        (vline_pos_list, k_path_name_list)
    else
        (vline_pos_list, ["A_$(i)" for i in eachindex(vline_pos_list)])
    end

    ax = CairoMakie.Axis(fig[1, 1],
        backgroundcolor=:transparent,
        aspect=1.6,
        xticks=vline_ticks,
    )

    for (k_crys, k_point_x) in zip(k_crys_list, k_point_xs)
        Hk_mat = Hk_crys(k_crys)
        @assert all(band_index <= size(Hk_mat, 1) for band_index in nband_range)
        @assert norm(Hk_mat - 1 / 2 * (Hk_mat + Hk_mat')) < 1.0E-8 "The k-space Hamiltonian is not Hermitian at `k_crys=$(k_crys)`!"
        eig_vals = eigen(Hermitian(Hk_mat)).values
        for i_band in nband_range
            CairoMakie.scatter!(ax, k_point_x, eig_vals[i_band]; color=CairoMakie.Cycled(i_band))
        end
    end

    for vline_pos in vline_pos_list
        CairoMakie.vlines!(ax, vline_pos; color=(:black, 0.3), linewidth=4)
    end

    display(fig)
    return fig
end


"Helper to find all cartesian coordinates of 1st-BZ vertices by intersecting half-planes"
function find_1st_BZ_k_cart_list(reciprocal_vec_list::Vector{<:Vector}; max_shell::Int=3)::Vector{Vector{Float64}}
    (b1, b2) = reciprocal_vec_list
    err_tol = 1.0E-8

    # Try growing shells until we can form a valid polygon
    for shell in 1:max_shell
        # Collect reciprocal lattice points in the given shell
        Gs = Vector{eltype(b1)}[]
        for n1 in -shell:shell, n2 in -shell:shell
            if n1 == 0 && n2 == 0
                continue
            end
            push!(Gs, n1 .* b1 .+ n2 .* b2)
        end

        # Keep only the shortest non-zero vectors (those that define the nearest-neighbor bisectors)
        norms = map(norm, Gs)
        minnorm = minimum(norms)
        nn_Gs = [G for (G, n) in zip(Gs, norms) if n <= minnorm * (1 + 1e-8)]

        # Lines: n ⋅ k = d, where n = G, d = |G|^2 / 2
        ns = nn_Gs
        ds = [dot(n, n) / 2 for n in ns]

        # Find all pairwise intersections that satisfy all half-plane constraints
        vertices_k_cart_list = Vector{Vector{Float64}}()
        for i in 1:length(ns)-1
            n1v = ns[i]
            d1 = ds[i]
            for j in i+1:length(ns)
                n2v = ns[j]
                d2 = ds[j]
                A = @inbounds [n1v[1] n1v[2]; n2v[1] n2v[2]]
                if abs(det(A)) < err_tol
                    continue
                end
                x = A \ [d1; d2]
                # Must lie within all half-planes
                if all(dot(nv, x) <= dv + err_tol for (nv, dv) in zip(ns, ds))
                    # Uniqueness filter
                    if all(norm(x .- v) > err_tol for v in vertices_k_cart_list)
                        push!(vertices_k_cart_list, Vector{Float64}(x))
                    end
                end
            end
        end

        # If we have a polygon, sort vertices counter-clockwise and return
        if length(vertices_k_cart_list) >= 3
            angles = map(v -> atan(v[2], v[1]), vertices_k_cart_list)
            order = sortperm(angles)
            return vertices_k_cart_list[order]
        end
    end
    return Vector{Vector{Float64}}() # fallback if not found
end


"""
Plot Band Contour for 2D Systems
---
- Args:
    - `hk_cart::Function`: the k-space Hamiltonian function with `k_cart` input
    - `k_data::Uniform_Grids`: the k-space uniform grids
- Named Args:
    - `k_cart_ranges::Vector{<:StepRangeLen}`: the kx and ky ranges for contour plot
    - `levels::Int=10`: number of contour levels
    - `band_idx::Int=1`: the band index to plot
    - `show_BZ::Bool=true`: whether to show the 1st Brillouin zone boundary
"""
function plot_band_contour(hk_cart::Function, k_data::Uniform_Grids;
    k_cart_ranges::Vector{<:StepRangeLen}=[-1.5π:0.05:1.5π, -1.5π:0.05:1.5π],
    levels::Int=10,
    band_idx::Int=1,
    show_BZ::Bool=true
)::CairoMakie.Figure
    @assert k_data.dim == 2 "Only 2D systems are supported for band contour plot!"
    @assert length(k_cart_ranges) == 2 "The length of `k_cart_ranges` must be 2 for 2D systems!"

    (kx_range, ky_range) = k_cart_ranges

    n_kx = length(kx_range)
    n_ky = length(ky_range)
    energy_spec = zeros(Float64, n_kx, n_ky)
    for (i_kx, kx) in enumerate(kx_range)
        for (i_ky, ky) in enumerate(ky_range)
            k_cart = [kx, ky]
            Hk_mat = hk_cart(k_cart)
            @assert norm(Hk_mat - Hk_mat') < 1.0E-10 "The Hamiltonian matrix is not Hermitian at k-point $(k_cart)!"

            energy_spec[i_kx, i_ky] = eigen(Hermitian(Hk_mat)).values[band_idx]
        end
    end

    fig = CairoMakie.Figure(size=(600, 400), backgroundcolor=:transparent)
    ax = fig[1, 1] = CairoMakie.Axis(fig;
        backgroundcolor=:transparent,
        xlabel="kx", ylabel="ky",
    )
    p = CairoMakie.contourf!(ax, kx_range, ky_range, energy_spec; colormap=:viridis, levels=levels)
    CairoMakie.Colorbar(fig[1, 2], p)
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    # resize_to_layout!(fig)


    if show_BZ
        # Compute the vertices of the 1st Brillouin zone (Wigner–Seitz cell in k-space) using the perpendicular bisectors of the shortest non-zero reciprocal lattice vectors
        b1, b2 = k_data.basis_vec_list
        @assert length(b1) == 2 && length(b2) == 2 "Only 2D BZ plotting is supported."

        bz_vertices = find_1st_BZ_k_cart_list(k_data.basis_vec_list; max_shell=2) # bz_vertices now holds the list of vertices of the 1st BZ (in k-cartesian coordinates)

        @assert !isempty(bz_vertices) "Failed to compute BZ vertices! Check the reciprocal lattice vectors."

        poly_x = [v[1] for v in bz_vertices]
        poly_y = [v[2] for v in bz_vertices]
        # close polygon
        push!(poly_x, poly_x[1])
        push!(poly_y, poly_y[1])
        CairoMakie.lines!(ax, poly_x, poly_y; color=:black, linewidth=2)
    end
    return fig
end