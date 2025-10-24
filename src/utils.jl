"""
Get Dual Basis Vector Matrix from a Given Basis Vector Matrix
---
satisfying the relation `dual_basis_vec_mat' * basis_vec_mat = 2π * I`, where both matrix should be stored in _columns_, i.e., `basis_vec_mat = [v1 v2 ...]`. 

This method can be useful to tranform from real-space basis to momentum-space basis, or vice versa.
- Args:
    - `basis_vec_mat::Matrix{T}`
"""
@inline function dual_basis_vec_mat(basis_vec_mat::Matrix{T})::Matrix{T} where T<:Number
    res_vec_mat = 2π * inv(basis_vec_mat)'
    @assert norm(res_vec_mat' * basis_vec_mat - 2π * I) < 1.0E-10 "The computed `dual_basis_vec_mat` does not satisfy the relation `dual_basis_vec_mat' * basis_vec_mat = 2π * I`!"
    return res_vec_mat
end
"add a method to accept input of `basis_vec_list`"
@inline function dual_basis_vec_mat(basis_vec_list::Vector{Vector{T}})::Matrix{T} where T<:Number
    basis_vec_mat = reduce(hcat, basis_vec_list) # force `basis_vec` to be stored in columns in `basis_vec_mat`
    return dual_basis_vec_mat(basis_vec_mat)
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