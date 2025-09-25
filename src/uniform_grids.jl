"""
Struct `Uniform_Grids{T,U}`
---
mostly for k-space lattice. It can either be constructed independently, or from the real-space lattice with all-direction periodic boundary conditions. Note: for the latter case the real-space sublattice degrees of freedom are transferred to the dimension of the single-particle k-space Hamiltonian, which should be handled elsewhere.
- Fields: 
    - `name::String`: name of the uniform grid
    - `dim::Int`: dimension of the uniform grid
    - `sample_size::Vector{Int}`
    - `basis_vec_list::Vector{<:Vector}`: for k-space lattice, this is just the reciprocal vectors
    - `cell_volume::T`: volume of the unit cell for the uniform grid (please distinguish from the real space cell-volume when the uniform grid is for momentum space!)
    - `twisted_phases_over_2π::Vector{U}`: twisted phases (over 2π). This will shift the `site_crys_list` as well as `site_cart_list`
    - `site_int_list::Vector{Vector{Int}}`
    - `site_int_to_index_map::Dict{Vector{Int},Int}`: hashmap `site_int -> i_site`
    - `site_crys_list::Vector{<:Vector}`: positions of sites in crystal coordinates
    - `site_cart_list::Vector{<:Vector}`: positions of sites in cartesian coordinates
    - `nsite::Int`
"""
struct Uniform_Grids{T,U}
    name::String
    dim::Int
    sample_size::Vector{Int} # number of k-points in each direction

    basis_vec_list::Vector{<:Vector} # for k-space lattice, this is just the reciprocal vectors
    cell_volume::T # volume of the unit cell in momentum space (please distinguish it from the volume of the unit cell in real space!)

    twisted_phases_over_2π::Vector{U} # twisted phases (over 2π). This will shift the `site_crys_list` as well as `site_cart_list`
    site_int_list::Vector{Vector{Int}}
    site_int_to_index_map::Dict{Vector{Int},Int} # hashmap `site_int -> i_site`
    site_crys_list::Vector{<:Vector} # positions of sites in crystal coordinates
    site_cart_list::Vector{<:Vector} # positions of sites in cartesian coordinates

    nsite::Int
end

"""
Constructor of `Uniform_Grids`
---
- Named Args:
    - `basis_vec_list::Vector{<:Vector}`: list of basis vectors for the uniform grid
    - `sample_size::Vector{Int}`: number of k-points in each direction
    - `name::String=""`: name of the uniform grid
    - `twisted_phases_over_2π::Vector{U}`: twisted phases (over 2π). This will shift the `site_crys_list` as well as `site_cart_list`
"""
function initialize_uniform_grids(;
    basis_vec_list::Vector{<:Vector},
    sample_size::Vector{Int},
    name::String="",
    twisted_phases_over_2π::Vector{U}
)::Uniform_Grids where U
    dim = length(sample_size)
    @assert length(basis_vec_list) == dim "The length of `basis_vec_list` must be the same as the dimension of the uniform grid!"
    @assert all(length.(basis_vec_list) .== dim) "Every basis vector in `basis_vec_list` must have the same dimension as the uniform grid!"

    basis_vec_mat = reduce(hcat, basis_vec_list) # `hcat()` forces `basis_vec` to be stored in columns in `basis_vec_mat`
    cell_volume = abs(det(basis_vec_mat)) # volume of the unit cell in momentum space

    site_int_list = Iterators.product([0:(Ni-1) for Ni in sample_size]...) .|> collect |> vec
    nsite = length(site_int_list)
    site_int_to_index_map = Dict(zip(site_int_list, 1:nsite))

    site_crys_list = [((site_int + twisted_phases_over_2π) ./ sample_size) for site_int in site_int_list]
    site_cart_list = [sum(site_crys .* basis_vec_list) for site_crys in site_crys_list]

    return Uniform_Grids{typeof(cell_volume),U}(
        name,
        dim,
        sample_size,
        basis_vec_list,
        cell_volume,
        twisted_phases_over_2π,
        site_int_list,
        site_int_to_index_map,
        site_crys_list,
        site_cart_list,
        nsite
    )
end

"""
Get Dual Basis Vector Matrix from a Given Basis Vector Matrix
---
satisfying the relation `dual_basis_vec_mat' * basis_vec_mat = 2π * I`, where both matrix should be stored in _columns_, i.e., `basis_vec_mat = [v1 v2 ...]`. 

This method can be useful to tranform from real-space basis to momentum-space basis, or vice versa.
- Args:
    - `basis_vec_mat::Matrix{T}`
"""
@inline function dual_basis_vec_mat(basis_vec_mat::Matrix{T})::Matrix{T} where T<:Number
    return 2π * inv(basis_vec_mat)'
end

"""
Get Dual Basis Vector List from a Given Basis Vector List
---
using the method `dual_basis_vec_mat()` to satisfy the relation `dual_basis_vec_mat' * basis_vec_mat = 2π * I`, where both matrix should be stored in _columns_, i.e., `basis_vec_mat = [v1 v2 ...]`.

This method can be useful to tranform from real-space basis to momentum-space basis, or vice versa.
- Args:
    - `basis_vec_list::Vector{<:Vector}`: bravais vectors for real-space lattice, or reciprocal vectors for k-space lattice
"""
@inline function dual_basis_vec_list(basis_vec_list::Vector{Vector{T}})::Vector{Vector{T}} where T<:Number
    basis_vec_mat = reduce(hcat, basis_vec_list) # force `basis_vec` to be stored in columns in `basis_vec_mat`
    return eachcol(dual_basis_vec_mat(basis_vec_mat)) .|> collect # clone to create a `Vector{Vector{T}}`
end


"""
Constructor of `Uniform_Grids`
---
from `r_data::Real_Space_Lattice` satisfying PBC in ALL directions.
- Args:
    - `r_data::Real_Space_Lattice`: the underlying real-space lattice (with PBC in ALL directions)
- Named Args:
    - `twisted_phases_over_2π::Vector{U}`: twisted phases (over 2π). This will shift the `site_crys_list` as well as `site_cart_list`
"""
function initialize_uniform_grids(
    r_data::Real_Space_Lattice;
    twisted_phases_over_2π::Vector{U}
)::Uniform_Grids where U
    dim = r_data.dim
    sample_size = r_data.sample_size

    dual_basis_vec_mat = reduce(hcat, r_data.brav_vec_list) # for momentum-space lattice, the `dual_basis_vec_mat` is the real-space bravais vectors
    basis_vec_mat = 2π * inv(dual_basis_vec_mat)'

    cell_volume = abs(det(basis_vec_mat)) # volume of the unit cell in momentum space
    basis_vec_list = [basis_vec_mat[:, i] for i in 1:dim]

    site_int_list = Iterators.product([0:(Ni-1) for Ni in sample_size]...) .|> collect |> vec
    nsite = length(site_int_list)
    site_int_to_index_map = Dict(zip(site_int_list, 1:nsite))

    site_crys_list = [((site_int + twisted_phases_over_2π) ./ sample_size) for site_int in site_int_list]
    site_cart_list = [sum(site_crys .* basis_vec_list) for site_crys in site_crys_list]

    return Uniform_Grids{typeof(r_data.cell_volume),T}(
        r_data.lattice_name,
        dim,
        sample_size,
        basis_vec_list,
        cell_volume,
        twisted_phases_over_2π,
        site_int_list,
        site_int_to_index_map,
        site_crys_list,
        site_cart_list,
        nsite
    )
end

