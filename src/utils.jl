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



"""
Build Real-Space Hamiltonian Matrix with Twisted Boundary Conditions
---
Construct the `n_site × n_site` real-space Hamiltonian matrix for a tight-binding model
with twisted boundary conditions specified by `twisted_phase_over_2π`.

A hopping that crosses the periodic boundary in direction `d` with winding number `w_d`
acquires an extra phase factor `exp(i·2π·twisted_phase_over_2π[d]·w_d)`.

- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the tight-binding model
    - `twisted_phase_over_2π::Vector{Float64}`: twisted phases φ/(2π) along each direction
- Returns:
    - `::Matrix{ComplexF64}` the `n_site × n_site` real-space Hamiltonian matrix
"""
function build_Hamiltonian_matrix(
    tb_model::Real_Space_TightBinding_Model,
    twisted_phase_over_2π::Vector{Float64},
)::Matrix{ComplexF64}
    l = tb_model.lattice
    n_site = l.n_site
    dim = l.dim
    L = l.sample_size
    pbc = l.pbc_indicator

    H = zeros(ComplexF64, n_site, n_site)

    # Determine which hopping map to use.
    # `input_hopping_map` contains hopping templates (single copy per orbital pair).
    # `full_hopping_map` contains all translated copies (already wrapped).
    use_input = !isempty(tb_model.input_hopping_map)

    if use_input
        # --- Build from input_hopping_map (template hoppings) ---
        for ((site_from, site_to), amp) in tb_model.input_hopping_map
            (cell_from, sub_from) = site_from
            (cell_to, sub_to) = site_to

            Δ_cell_template = cell_to - cell_from

            for cell in l.cell_int_list
                cell_to_new = cell + Δ_cell_template

                # Compute winding numbers and wrap cell_to_new
                winding = zeros(Int, dim)
                skip = false
                for d in 1:dim
                    if pbc[d]
                        winding[d] = fld(cell_to_new[d], L[d])
                        cell_to_new[d] = mod(cell_to_new[d], L[d])
                    else
                        if cell_to_new[d] < 0 || cell_to_new[d] >= L[d]
                            skip = true
                            break
                        end
                    end
                end
                skip && continue

                # Phase factor from twisted boundary conditions
                phase = cis(2π * dot(twisted_phase_over_2π, winding))

                i_site = l.site_to_index_map[(cell, sub_from)]
                j_site = l.site_to_index_map[(cell_to_new, sub_to)]

                H[i_site, j_site] += amp * phase
            end
        end
    else
        # --- Build from full_hopping_map (already translated & wrapped) ---
        # Reconstruct templates: collect all hoppings that start from cell [0,0,...]
        template_map = Dict{Tuple{Int,Int,Vector{Int}},ComplexF64}()
        for ((site_from, site_to), amp) in tb_model.full_hopping_map
            (cell_from, sub_from) = site_from
            (cell_to, sub_to) = site_to
            if all(cell_from .== 0)
                Δ = collect(cell_to)
                key = (sub_from, sub_to, Δ)
                template_map[key] = get(template_map, key, zero(ComplexF64)) + amp
            end
        end

        for ((sub_from, sub_to, Δ_cell_template), amp) in template_map
            for cell in l.cell_int_list
                cell_to_new = cell + Δ_cell_template

                winding = zeros(Int, dim)
                for d in 1:dim
                    if pbc[d]
                        winding[d] = fld(cell_to_new[d], L[d])
                        cell_to_new[d] = mod(cell_to_new[d], L[d])
                    end
                end

                phase = cis(2π * dot(twisted_phase_over_2π, winding))

                i_site = l.site_to_index_map[(cell, sub_from)]
                j_site = l.site_to_index_map[(cell_to_new, sub_to)]

                H[i_site, j_site] += amp * phase
            end
        end
    end

    return H
end



"""
Compute Many-Body Chern Number using Fukui–Hatsugai–Suzuki Method on the Flux Torus
---
For a non-interacting tight-binding model, we compute the **many-body Chern number**
by discretising the flux-torus (θ₁, θ₂) ∈ [0,1]² and applying the Fukui–Hatsugai–Suzuki
method to the many-body ground-state Slater determinant.

The many-body Chern number of a non-interacting system equals the sum of the
single-particle Chern numbers of all occupied bands.

- Args:
    - `tb_model::Real_Space_TightBinding_Model`: the tight-binding model (must have PBC in ALL directions)
    - `n_occ::Int`: number of occupied single-particle states (filling)
    - `nθ::Int=21`: number of θ-points per direction on the flux grid
- Returns:
    - `::Float64` the many-body Chern number
"""
function many_body_Chern_number_Fukui_Hatsugai_Suzuki(
    tb_model::Real_Space_TightBinding_Model;
    n_occ::Int,
    nθ::Int=21,
)::Float64
    l = tb_model.lattice
    @assert all(l.pbc_indicator) "All directions must be periodic (torus) for many-body Chern number."

    n_site = l.n_site
    @assert 1 ≤ n_occ ≤ n_site "n_occ=$n_occ must be between 1 and n_site=$n_site."

    # Pre-allocate storage for occupied eigenvectors at each θ-point
    occ_vecs = Array{Matrix{ComplexF64}}(undef, nθ, nθ)  # each entry is (n_site × n_occ)

    for i in 0:(nθ-1)
        θ₁ = i / nθ
        for j in 0:(nθ-1)
            θ₂ = j / nθ
            θ = [θ₁, θ₂]  # crystal-coordinate flux phases

            H = build_Hamiltonian_matrix(tb_model, θ)
            F = eigen(Hermitian(H))
            # Store the n_occ lowest eigenvectors
            occ_vecs[i+1, j+1] = F.vectors[:, 1:n_occ]
        end
    end

    # Link variable: overlap of two Slater determinants
    function slater_link(V1::Matrix{ComplexF64}, V2::Matrix{ComplexF64})::ComplexF64
        # V1, V2 are (n_site × n_occ) matrices whose columns are the occupied single-particle states
        # Overlap matrix S_{ab} = ⟨ψ_a^(1)|ψ_b^(2)⟩ = (V1[:,a]† · V2[:,b])
        S = V1' * V2  # (n_occ × n_occ) overlap matrix
        z = det(S)
        return z / abs(z)
    end

    total_flux = 0.0

    for i in 1:nθ, j in 1:nθ
        ip = mod1(i + 1, nθ)
        jp = mod1(j + 1, nθ)

        V = occ_vecs[i, j]
        Vx = occ_vecs[ip, j]
        Vy = occ_vecs[i, jp]
        Vxy = occ_vecs[ip, jp]

        Ux = slater_link(V, Vx)
        Uy = slater_link(V, Vy)
        Ux_y = slater_link(Vy, Vxy)
        Uy_x = slater_link(Vx, Vxy)

        total_flux += angle(Ux * Uy_x / (Ux_y * Uy))
    end

    return total_flux / (2π)
end