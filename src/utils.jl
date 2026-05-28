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





"""
Construct Crystal-Momentum Hamiltonian `H(k)` in _Periodic Gauge_
---
strickly satisfying `H(k+G) = H(k)`.
> Note: if you input with `H^{αβ} = ∑_𝐑' t^{0,α; 𝐑',β} e^{i𝐤⋅[(𝐑'+𝛕_β)-(0+𝛕_α)]`, then you are in a gauge that is NOT periodic in the BZ.

- Args:
    - `tb_model::Real_Space_TightBinding_Model`
- Returns:
    - `Hk_crys::Function` sending the crystal k-vector to the `n_sub × n_sub` Hamiltonian matrix `H(k)`
"""
# function build_Hk_crys(tb_model::Real_Space_TightBinding_Model)::Function
#     l = tb_model.lattice
#     n_sub = l.n_sub
#     sub_crys = l.sub_crys_list

#     function Hk_crys(k_crys::Vector{Float64})
#         Hk = zeros(ComplexF64, n_sub, n_sub)

#         if length(tb_model.input_hopping_map) != 0
#             for ((site_from, site_to), amp) in tb_model.input_hopping_map
#                 (cell_from, sub_from) = site_from
#                 (cell_to, sub_to) = site_to

#                 Δ_crys = cell_to - cell_from # DO NOT use the gauge with `Δ_crys = (cell_to + sub_crys[sub_to]) - (cell_from + sub_crys[sub_from])` 

#                 Hk[sub_from, sub_to] += amp * cis(2π * dot(k_crys, Δ_crys)) # dot product direct in crystal space is OK
#             end
#         else
#             for ((site_from, site_to), amp) in tb_model.full_hopping_map
#                 (cell_from, sub_from) = site_from
#                 (cell_to, sub_to) = site_to

#                 Δ_crys = cell_to - cell_from # DO NOT use the gauge with `Δ_crys = (cell_to + sub_crys[sub_to]) - (cell_from + sub_crys[sub_from])` 

#                 TightBinding._wrap_Δ_crys!(Δ_crys;
#                     sample_size=l.sample_size,
#                     pbc_indicator=l.pbc_indicator,
#                 )

#                 Hk[sub_from, sub_to] += amp * cis(2π * dot(k_crys, Δ_crys)) # dot product direct in crystal space is OK
#             end
#         end

#         return Hk
#     end

#     return Hk_crys
# end


function build_Hk_crys(tb_model::Real_Space_TightBinding_Model)::Function
    l = tb_model.lattice
    n_sub = l.n_sub

    function Hk_crys(k_crys::Vector{Float64})
        Hk = zeros(ComplexF64, n_sub, n_sub)

        if !isempty(tb_model.input_hopping_map)
            # Infinite-system hopping templates: no finite-size wrapping.
            for ((site_from, site_to), amp) in tb_model.input_hopping_map
                (cell_from, sub_from) = site_from
                (cell_to, sub_to) = site_to

                Δ_cell = cell_to - cell_from

                Hk[sub_from, sub_to] += amp * cis(2π * dot(k_crys, Δ_cell))
            end
        else
            # Graph-generated finite torus hoppings:
            # compress translated copies into one hopping template.
            reduced = Dict{Any,ComplexF64}()

            for ((site_from, site_to), amp) in tb_model.full_hopping_map
                (cell_from, sub_from) = site_from
                (cell_to, sub_to) = site_to

                Δ_cell = Float64.(cell_to - cell_from)

                TightBinding._wrap_Δ_crys!(Δ_cell;
                    sample_size=l.sample_size,
                    pbc_indicator=l.pbc_indicator,
                )

                Δ_key = Tuple(Int.(round.(Δ_cell)))
                key = (sub_from, sub_to, Δ_key)

                reduced[key] = get(reduced, key, 0.0 + 0.0im) + amp
            end

            # Divide by number of unit cells because full_hopping_map contains
            # one translated copy per cell.
            for ((sub_from, sub_to, Δ_key), amp_sum) in reduced
                Δ_cell = collect(Δ_key)
                amp = amp_sum / l.n_cell

                Hk[sub_from, sub_to] += amp * cis(2π * dot(k_crys, Δ_cell))
            end
        end

        return Hk
    end

    return Hk_crys
end



"""
Compute Chern Number using Fukui–Hatsugai–Suzuki Method
---
We compute a single-band **Chern number** for an `n_sub`-band tight-binding Hamiltonian `H(k)` discretised on a *single ℤₙ×ℤₙ* k-grid in the crystal momentum BZ.
> Reference: T. Fukui, Y. Hatsugai, & H. Suzuki, J. Phys. Soc. Jpn. **74**, 1674–1677 (2005).
- Args:
    - `Hk_crys::Function`: function `Hk_crys(k_crys)` → Hamiltonian matrix `::AbstractMatrix{<:Complex}`
    - `band::Int`: 1-based band index (e.g. `1` = lowest, `2` = next)
    - `nk::Int=51`: number of k-points per direction for grid
- Returns:
    - `::Float64` the computed Chern number of the given `band`
"""
function Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys::Function; band::Int, nk::Int=51)
    # band is 1-based: band=1 lower, band=2 upper
    H0 = Hk_crys([0.0, 0.0])
    n_sub = size(H0, 1)

    vecs = Array{ComplexF64}(undef, n_sub, nk, nk)

    for i in 0:(nk-1), j in 0:(nk-1)
        k = [i / nk, j / nk]
        F = eigen(Hermitian(Hk_crys(k)))
        vecs[:, i+1, j+1] = F.vectors[:, band]
    end

    link(u, v) = begin
        z = dot(u, v)       # Julia dot conjugates the first argument
        z / abs(z)
    end

    total_flux = 0.0

    for i in 1:nk, j in 1:nk
        ip = mod1(i + 1, nk)
        jp = mod1(j + 1, nk)

        u = vecs[:, i, j]
        ux = vecs[:, ip, j]
        uy = vecs[:, i, jp]
        uxy = vecs[:, ip, jp]

        Ux = link(u, ux)
        Uy = link(u, uy)
        Ux_y = link(uy, uxy)
        Uy_x = link(ux, uxy)

        total_flux += angle(Ux * Uy_x / (Ux_y * Uy))
    end

    return total_flux / (2π)
end