
"""
Struct `Real_Space_TightBinding_Model{T,U}`
---
for real-space hoppings.
- Fields:
    - `lattice::Real_Space_Lattice{T}`: the underlying real-space lattice
    - `model_name::String`: name of the tight-binding model
    - `pbc_indicator::Vector{Bool}`: whether to apply periodic boundary condition in direction-i
    - `input_hopping_map::Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},U}`: hashmap `(cell_int, i_sub) -> t`. This includes hoppings within and across unit cells. Hermicity is already implemented when building the `input_hopping_map`
    - `full_hopping_map::Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},U}`: hashmap `(cell_int, i_sub) -> t`. This includes and expands ALL hoppings of the model. Translation symmetry is already implemented for bulk hopping terms
    - `H_hop::Function`: real-space hopping Hamiltonian (for edge-mode calculation)
"""
mutable struct Real_Space_TightBinding_Model{T,U}
    lattice::Real_Space_Lattice{T}
    model_name::String

    pbc_indicator::Vector{Bool} # whether to apply periodic boundary condition in direction-i

    input_hopping_map::Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},U} # hashmap `(cell_int, i_sub) -> t`. This includes hoppings within and across unit cells. Hermicity is already implemented when building the `input_hopping_map`
    full_hopping_map::Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},U} # hashmap `(cell_int, i_sub) -> t`. This includes and expands ALL hoppings of the model. Translation symmetry is already implemented for bulk hopping terms

    H_hop::Function # real-space hopping Hamiltonian (for edge-mode calculation)
end

"""
Constructor for `Real_Space_TightBinding_Model`
---
- Args:
    - `lattice::Real_Space_Lattice{T}`: the underlying real-space lattice
- Named Args:
    - `model_name::String`: name of the tight-binding model
    - `pbc_indicator::Vector{Bool}`: whether to apply periodic boundary condition in direction-i
"""
function initialize_real_space_tightbinding_model(lattice::Real_Space_Lattice{T};
    model_name::String="",
    pbc_indicator::Vector{Bool},
)::Real_Space_TightBinding_Model where T
    @assert length(pbc_indicator) == lattice.dim

    input_hopping_map = Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},Number}()
    full_hopping_map = Dict{Tuple{Tuple{Vector{Int},Int},Tuple{Vector{Int},Int}},Number}()

    function H_hop end

    return Real_Space_TightBinding_Model(
        lattice,
        model_name,
        pbc_indicator,
        input_hopping_map,
        full_hopping_map,
        H_hop
    )
end




"""
Add One Hopping Term to `Real_Space_TightBinding_Model`
---
- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the real-space tight-binding model to which the hopping term will be added
    - `input_hopping_term::Pair{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},T}`: the input hopping term in the form of `((cell_from, sub_from), (cell_to, sub_to)) => hopping_strength`. Note: it also applies to chemical potentials, when `cell_from == cell_to` and `sub_from == sub_to`.
- Named Args:
    - `is_hermitian::Bool=true`: whether to add the Hermitian conjugate of the input hopping term to the model
"""
function add_hopping_term!(
    tb_model::Real_Space_TightBinding_Model,
    input_hopping_term::Pair{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},T};
    is_hermitian::Bool=true,
) where T
    nsub = tb_model.lattice.nsub

    let (((cell_from, sub_from), (cell_to, sub_to)), hopping_strength) = input_hopping_term
        # check the validity of input hopping term
        @assert sub_from in 1:nsub && sub_to in 1:nsub "The input sublattice indices for `input_hopping_map`=$(input_hopping_term) is invalid for `sample_size`=$(tb_model.lattice.sample_size)!"
        # @assert all(0 .<= cell_from .<= (tb_model.lattice.sample_size .- 1)) && all(0 .<= cell_to .<= (tb_model.lattice.sample_size .- 1)) "The input cell indices for `input_hopping_map`=$(input_hopping_term) is invalid for `sample_size`=$(tb_model.lattice.sample_size)!"

        if haskey(tb_model.input_hopping_map, ((cell_from, sub_from), (cell_to, sub_to)))
            @warn "The input hopping term `$(input_hopping_term)` already exists in `input_hopping_map`! The old hopping term will be overwritten!"
        end

        tb_model.input_hopping_map[((cell_from, sub_from), (cell_to, sub_to))] = hopping_strength
        if is_hermitian
            tb_model.input_hopping_map[((cell_to, sub_to), (cell_from, sub_from))] = conj(hopping_strength)
        end

        # generate all bulk hopping terms using translation symmetry
        cell_diff = cell_to - cell_from
        for cell_int in tb_model.lattice.cell_int_list
            new_cell_from = cell_int
            new_cell_to = cell_int + cell_diff
            for i in 1:tb_model.lattice.dim
                if tb_model.pbc_indicator[i] # handle periodic boundary condition
                    new_cell_to[i] = new_cell_to[i] % tb_model.lattice.sample_size[i]
                end
            end
            if all(0 .<= new_cell_to .<= (tb_model.lattice.sample_size .- 1))
                tb_model.full_hopping_map[((new_cell_from, sub_from), (new_cell_to, sub_to))] = hopping_strength
                if is_hermitian
                    tb_model.full_hopping_map[((new_cell_to, sub_to), (new_cell_from, sub_from))] = conj(hopping_strength)
                end
            end
        end
        return nothing
    end
end

"""
Plot Hoppings and Sites of the `Real_Space_TightBinding_Model`
---
- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the real-space tight-binding model to be visualized
---
Note: Hoppings can be overlapping if the model is complicated.
"""
function plot_real_space_tightbinding_model(tb_model::Real_Space_TightBinding_Model)::CairoMakie.Figure
    l = tb_model.lattice
    @assert eltype(first(l.site_cart_list)) <: Real "The cartesian coordinates of sites must be real numbers for plotting!"
    fig = CairoMakie.Figure(size=(300, 300), backgroundcolor=:transparent)
    ax = @match l.dim begin
        2 => CairoMakie.Axis(fig[1, 1], backgroundcolor=:transparent; aspect=CairoMakie.DataAspect())
        3 => CairoMakie.Axis3(fig[1, 1], backgroundcolor=:transparent)
        _ => error("The dimension of the lattice must be 2 or 3!")
    end

    # plot all sites
    for (((cell_int), i_sub), i_site) in l.site_pos_to_index_map
        site_cart = l.site_cart_list[i_site]

        # color-coded by sublattice
        CairoMakie.scatter!(ax, site_cart...; markersize=20, color=CairoMakie.Cycled(i_sub))
    end

    # plot all real-space hoppings
    for (((cell_from, sub_from), (cell_to, sub_to)), t) in tb_model.full_hopping_map
        hopping_from_site_cart = l.site_cart_list[l.site_pos_to_index_map[(cell_from, sub_from)]]
        hopping_to_site_cart = l.site_cart_list[l.site_pos_to_index_map[(cell_to, sub_to)]]

        hopping_site_components = [reduce(hcat, [hopping_from_site_cart, hopping_to_site_cart])[d, :] for d in 1:l.dim] |> collect
        CairoMakie.lines!(ax, hopping_site_components...; linewidth=4, color=(:black, 0.3))
    end

    CairoMakie.display(fig)
    return fig
end