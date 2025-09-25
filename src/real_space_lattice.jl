"""
Struct `Real_Space_Lattice{T}`
---
- fields:
    - `lattice_name::String`: name of lattice
    - `dim::Int`: dimension of lattice
    - `sample_size::Vector{Int}`
    - `cell_int_list::Vector{<:Vector}`: list of integer cell indices
    - `ncell::Int`: number of unit cells
    - `brav_vec_list::Vector{<:Vector}`: list of bravais vectors for real-space lattice (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `cell_volume::T`: volume of the unit cell in real space
    - `nsub::Int`: number of sublattices in each unit cell
    - `sub_crys_list::Vector{<:Vector}`: list of sublattice positions _in crystal coordinates_ (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `sub_name_list::Vector{String}`: list of sublattice names
    - `nsite::Int`: total number of sites in the lattice
    - `site_pos_list::Vector{Tuple{Vector{Int},Int}}`: list of site positions in each cell as `(cell_int, i_sub)`
    - `site_cart_list::Vector{<:Vector}`: list of site positions in cartesian coordinates (it can be _symbolic_ such as `MathExpr` from `YAN.jl`)
    - `site_pos_to_index_map::Dict{Tuple{Vector{Int},Int},Int}`: hashmap `(cell_int, i_sub) -> i_site`
"""
mutable struct Real_Space_Lattice{T}
    lattice_name::String
    dim::Int
    sample_size::Vector{Int}

    cell_int_list::Vector{<:Vector}
    ncell::Int

    brav_vec_list::Vector{<:Vector} # bravais vectors for real-space lattice
    cell_volume::T

    nsub::Int
    sub_crys_list::Vector{<:Vector} # sublattice positions in crystal coordinates
    sub_name_list::Vector{String}

    nsite::Int
    site_pos_list::Vector{Tuple{Vector{Int},Int}} # site positions in each cell as `(cell_int, i_sub)`
    site_cart_list::Vector{<:Vector} # site positions in cartesian coordinates
    site_pos_to_index_map::Dict{Tuple{Vector{Int},Int},Int} # hashmap `(cell_int, i_sub) -> i_site`
end


"""
Constructor for `Real_Space_Lattice`
---
- Named Args:
    - `brav_vec_list::Vector{<:Vector}`: list of bravais vectors for real-space lattice
    - `sample_size::Vector{Int}`: number of unit cells in each direction
    - `sub_crys_list::Vector{<:Vector}`: list of sublattice positions _in crystal coordinates_
    - `lattice_name::String`: name of lattice. If this is set to be `"square"`, `"honeycomb"`, `"kagome"`, and `"Lieb"`, it will override the above three arguments with the corresponding default values.)
"""
function initialize_real_space_lattice(;
    brav_vec_list::Vector{<:Vector}=[[1.0, 0.0], [0.0, 1.0]],
    sample_size::Vector{Int}=[2, 2],
    sub_crys_list::Vector{<:Vector}=[[0.0, 0.0]],
    lattice_name::String="",
)::Real_Space_Lattice
    (brav_vec_list, sub_crys_list) = @match lattice_name begin
        "square" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]])
        "honeycomb" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3]])
        "kagome" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        "Lieb" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        _ => (brav_vec_list, sub_crys_list)
    end
    dim = length(brav_vec_list)

    brav_vec_mat = reduce(hcat, brav_vec_list) # `hcat()` forces `brav_vec` to be stored in columns in `brav_vec_mat`
    cell_volume = abs(det(brav_vec_mat))

    @assert dim == 2 || dim == 3
    @assert length(brav_vec_list) == length(sample_size)
    nsub = length(sub_crys_list)
    @assert nsub >= 1 # at least one sublattice

    cell_int_list = Iterators.product([0:(Ni-1) for Ni in sample_size]...) .|> collect |> vec
    ncell = length(cell_int_list)

    @assert all(length.(sub_crys_list) == [dim for _ in 1:nsub])
    sub_name_list::Vector{String} = [string("A", i) for i in 1:nsub] # the default name 

    site_pos_list = [(cell_int, i_sub) for cell_int in cell_int_list for i_sub in 1:nsub]
    nsite = length(site_pos_list)
    site_cart_list = [sum(brav_vec_list .* (cell_int + sub_crys_list[i_sub])) for (cell_int, i_sub) in site_pos_list]

    site_pos_to_index_map = Dict(zip(site_pos_list, 1:nsite))

    return Real_Space_Lattice(
        lattice_name,
        dim,
        sample_size,
        cell_int_list,
        ncell,
        brav_vec_list,
        cell_volume,
        nsub,
        sub_crys_list,
        sub_name_list,
        nsite,
        site_pos_list,
        site_cart_list,
        site_pos_to_index_map,
    )
end
