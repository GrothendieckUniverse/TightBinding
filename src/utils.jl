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