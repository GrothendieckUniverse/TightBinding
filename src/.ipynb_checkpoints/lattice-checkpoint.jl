module LatticeModule

using LinearAlgebra, SparseArrays, KrylovKit
using LaTeXStrings
using MLStyle
using Plots
using SymEngine

const σ0 = [1 0; 0 1] # identity matrix
const σ = [
    [0 1; 1 0];
    [0 -im; im 0];
    [1 0; 0 -1]
] # Pauli matrices


export Lattice, LatticeModel
export initialize_lattice, initialize_lattice_model, show, add_hopping_term!, update_full_hopping_Hamiltonian!


struct Lattice
    lattice_name::String
    sample_size::Vector{Int64}
    brav_vecs::Vector{Vector{Float64}}
    dim::Int64
    sub_crys_vecs::Vector{Vector{Float64}}
    nsub::Int64
    sub_name_list::Vector{String}
    cell_volume::Float64
    cell_pos_list::Vector{Vector{Int64}} # `cell_pos` as `[i,j,k]`
    atom_pos_list::Vector{Tuple{Vector{Int64},Int64}} # `atom_pos` as `(cell_pos, atom_type)`
    atom_cart_list::Vector{Vector{Float64}} # `atom_cart_pos` in cartesian coordinates
end

mutable struct LatticeModel
    lattice::Lattice
    model_name::String
    pbc_indicator::Vector{Bool}
    twisted_phases_over_2π::Vector{Float64}

    # real-space data
    input_hopping_terms::Dict{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},SymEngine.Basic}
    full_hopping_terms::Dict{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},SymEngine.Basic} # updated with `add_hopping_term!()` in consideration of periodic boundary condition and phase twists
    full_hopping_Hamiltonian::SparseMatrixCSC{SymEngine.Basic,Int64} # real-space Hamiltonian

    # k-space data

    # k_data::KData
    # hk_crystal::Function
    # hk_cart::Function
end

struct KData
    G_vecs::Vector{Vector{Float64}} # reciprocal vectors
    k_int_list::Vector{Vector{Int64}}
    k_crys_list::Vector{Vector{Float64}}
    k_cart_list::Vector{Vector{Float64}}
end


"""
Struct Initialization of `Lattice`
---
input keywords:
- `brav_vecs`: real-space bravias vectors as `[a1,a2,...]`
- `sample_size`: sample size as `[N1,N2,...]`
- `sub_crys_vecs`: sublattice atom positions in *crystal* coordinates
- `lattice_name`: string of lattice name
"""
function initialize_lattice(; brav_vecs::Vector{Vector{Float64}}, sample_size::Vector{Int64}, sub_crys_vecs::Vector{Vector{Float64}}, lattice_name::String="")
    dim = length(brav_vecs)
    @assert dim == 2 || dim == 3
    @assert length(brav_vecs) == length(sample_size)
    nsub = length(sub_crys_vecs)
    @assert all(length.(sub_crys_vecs) == [dim for _ in 1:nsub])
    sub_name_list = [string("A", i) for i in 1:nsub]
    cell_pos_list = Iterators.map(x -> collect(x), Iterators.product([0:(N-1) for N in sample_size]...)) |> collect |> vec
    atom_pos_list = [(cell_pos, atom_type) for cell_pos in cell_pos_list for atom_type in 1:nsub]
    atom_cart_list = [sum(brav_vecs .* (cell_pos + sub_crys_vecs[atom_type])) for (cell_pos, atom_type) in atom_pos_list]

    cell_volume = if dim == 2
        a1 = push!(copy(brav_vecs[1]), 1.0)
        a2 = push!(copy(brav_vecs[2]), 1.0)
        a3 = [0.0, 0.0, 1.0]
        det([a1 a2 a3])
    elseif dim == 3
        det(hcat(brav_vecs...))
    end

    return Lattice(
        lattice_name,
        sample_size,
        brav_vecs,
        dim,
        sub_crys_vecs,
        nsub,
        sub_name_list,
        cell_volume,
        cell_pos_list,
        atom_pos_list,
        atom_cart_list
    )
end

"""
Example Struct Initialization of `Lattice`
---
add dispatch to include example lattices of `lattice_name = square, honeycomb, kagome, Lieb...`
"""
function initialize_lattice(lattice_name::String, sample_size::Vector{Int64})
    (brav_vecs, sub_crys_vecs) = @match lattice_name begin
        "square" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]])
        "honeycomb" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 3, 1 / 3]])
        "kagome" => ([[1.0, 0.0], [1 / 2, sqrt(3) / 2]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        "Lieb" => ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]])
        _ => error("`lattice_name` $lattice_name not defined!")
    end
    return initialize_lattice(; brav_vecs=brav_vecs, sample_size=sample_size, sub_crys_vecs=sub_crys_vecs, lattice_name=lattice_name)
end

"add plot of `l`'s unit-cell for figure `p`"
function plot_single_unit_cell!(l::Lattice, p::Plots.Plot)::Plots.Plot
    unit_cell_edges = [[[0, 0], [1, 0]], [[0, 0], [0, 1]], [[1, 0], [1, 1]], [[0, 1], [1, 1]]]
    for unit_cell_edge in unit_cell_edges
        unit_cell_corners = [sum(l.brav_vecs .* coef) for coef in unit_cell_edge] # coef is integer coefficients like [0,0], [1,0], etc.
        unit_cell_corner_cart_list_along_each_direction = [[unit_cell_corner[i] for unit_cell_corner in unit_cell_corners] for i in 1:l.dim]
        plot!(p, unit_cell_corner_cart_list_along_each_direction...;
            lc=:red,
            ls=:dot,
            # alpha=0.6,
            lw=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 50, # adjust markersize with the number of atoms in the plot
        )
    end
    return p
end

"add dispatch to `Base.show` to plot `Lattice`"
function show(l::Lattice; save_plot::Bool=true)::Plots.Plot
    fig = fig()
    # add atoms
    atom_cart_list_along_each_direction = [[atom_cart[i] for atom_cart in l.atom_cart_list] for i in 1:l.dim]
    plot!(fig, atom_cart_list_along_each_direction...;
        seriestype=:scatter,
        aspect_ratio=:equal,
        legend=false,
        framestyle=:none,
        markersize=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 100, # adjust markersize with the number of atoms in the plot
        ticks=false,
    )
    plot_single_unit_cell!(l, fig)  # plot single unit cell enclosed by bravias vectors

    if save_plot
        figure_dir = "figure/"
        if !ispath(figure_dir)
            mkpath(figure_dir)
        end
        savefig(fig, figure_dir * l.lattice_name * ".pdf")
    end
    return fig
end


"""
Initialize `KData` for Real-space `Lattice`
---
Note: we always shift `k_crys` by 0.5 grid if the sample size is even along that direction (to cover as much high symmetry points of BZ as possible)
"""
function initialize_k_data(l::Lattice)
    G_vecs = let a = l.brav_vecs
        @match l.dim begin
            3 => 2 * π * [cross(a[2], a[3]), cross(a[3], a[1]), cross(a[1], a[2])] / l.cell_volume
            2 => begin
                a = [append!(a[1], [0.0]), append!(a[2], [0.0]), [0.0, 0.0, 1.0]]
                b = 2 * π * [cross(a[2], a[3]), cross(a[3], a[1]), cross(a[1], a[2])] / l.cell_volume
                [b[1][1:2], b[2][1:2]]
            end
        end
    end
    k_int_list = Iterators.product(0:(l.sample_size[i]-1) for i in 1:l.dim) |> collect |> vec
    k_crys_list = [(k_int + (1 .- mod.(l.sample_size, 2)) * 0.5) ./ l.sample_size for k_int in k_int_list] # shift `k_crys` by 0.5 grid if the sample size is even along that direction (to cover as much high symmetry points of BZ as possible)
    k_cart_list = [sum(G_vecs .* k_crys) for k_crys in k_crys_list]

    return KData(
        G_vecs,
        k_int_list,
        k_crys_list,
        k_cart_list
    )
end


"""
Struct Initialization of `LatticeModel`
---
input keywords:
- `l`: `Lattice` struct
- `input_hopping_terms`: dict of hopping terms involving all atoms *within* one unit cell, the format is `((cell_from, atom_from), (cell_to, atom_to)) => hopping_strength::SymEngine.Basic`
- `pbc_indicator`: vector of boolean of periodic boundary conditions
- `twisted_phases_over_2π`: vector of twisted phases over 2π
"""
function initialize_lattice_model(l::Lattice; model_name::String="", pbc_indicator::Vector{Bool}, twisted_phases_over_2π::Vector{Float64})
    @assert length(twisted_phases_over_2π) == l.dim
    @assert length(pbc_indicator) == l.dim

    # initialize empty hopping terms and emtpy Hamiltonian
    input_hopping_terms = Dict{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},SymEngine.Basic}()
    full_hopping_terms = Dict{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},SymEngine.Basic}()
    full_hopping_Hamiltonian = spzeros(SymEngine.Basic, length(l.atom_cart_list), length(l.atom_cart_list))

    # reduced_atom_pos_list = [([0 for _ in l.dim], atom_type) for atom_type in 1:l.nsub]
    # for atom_pos in l.atom_pos_list
    #     if atom_pos in reduced_atom_pos_list
    #         continue
    #     else
    #         # if translation symmetry exists, we can reduce the number of independent atoms in the lattice
    #         for i in 1:l.dim
    #             if l.pbc_indicator[i]

    #             end
    #         end
    #     end
    # end


    return LatticeModel(
        l,
        model_name,
        pbc_indicator,
        twisted_phases_over_2π,
        input_hopping_terms,
        full_hopping_terms,
        full_hopping_Hamiltonian
    )
end

"""
Add One Hopping Term to `LatticeModel`
---
The input format is `((cell_from, atom_from), (cell_to, atom_to)) => hopping_strength::SymEngine.Basic`. 

Note:
1. It also applies to chemical potentials, where `cell_from == cell_to` and `atom_from == atom_to`.
2. The hopping term is set to be Hermitian by default
"""
function add_hopping_term!(lm::LatticeModel, input_hopping_term::Pair{Tuple{Tuple{Vector{Int64},Int64},Tuple{Vector{Int64},Int64}},SymEngine.Basic}; is_hermitian::Bool=true)
    let (((cell_from, atom_from), (cell_to, atom_to)), hopping_strength) = input_hopping_term
        # promote hopping to all cells using translation symmetries
        cell_diff = cell_to - cell_from
        for (current_cell_from, current_atom_from) in lm.lattice.atom_pos_list
            new_cell_to = current_cell_from + cell_diff
            for i in 1:lm.lattice.dim
                if lm.pbc_indicator[i] # handle periodic boundary condition
                    new_cell_to[i] = new_cell_to[i] % lm.lattice.sample_size[i]
                end
            end
            if all(0 .<= new_cell_to .<= lm.lattice.sample_size) && current_atom_from == atom_from
                if is_hermitian
                    lm.full_hopping_terms[((current_cell_from, current_atom_from), (new_cell_to, atom_to))] = hopping_strength
                    lm.full_hopping_terms[((new_cell_to, atom_to), (current_cell_from, current_atom_from))] = SymEngine.conj(hopping_strength)
                else
                    lm.full_hopping_terms[((current_cell_from, current_atom_from), (new_cell_to, atom_to))] = hopping_strength
                end
            end
        end

        # spread inserted twisted phases along each direction
        # for i in 1:lm.lattice.dim
        #   todo!
        # end
        return nothing
    end
end

"update the hopping Hamiltonian from `LatticeModel.full_hopping_terms`"
function update_full_hopping_Hamiltonian!(lm::LatticeModel)
    for (((cell_from, atom_from), (cell_to, atom_to)), hopping_strength) in lm.full_hopping_terms
        atom_ind_from = findfirst(x -> x == (cell_from, atom_from), lm.lattice.atom_pos_list)
        atom_ind_to = findfirst(x -> x == (cell_to, atom_to), lm.lattice.atom_pos_list)
        if isnothing(atom_ind_from) || isnothing(atom_ind_to) # skip if the atom is not in the lattice
            continue
        else
            lm.full_hopping_Hamiltonian[atom_ind_from, atom_ind_to] += hopping_strength
        end
    end
    return lm.full_hopping_Hamiltonian
end

"add dispatch to `Base.show` to plot `LatticeModel`"
function show(lm::LatticeModel; save_plot::Bool=true)::Plots.Plot
    l = lm.lattice
    fig = plot()

    # plot hopping terms
    for ((cell_from, atom_from), (cell_to, atom_to)) in keys(lm.full_hopping_terms)
        # skip plot if the hopping term is within the same atom, i.e., skip the chemical potential
        if cell_from == cell_to && atom_from == atom_to
            continue
        end

        atom_ind_from = findfirst(x -> x == (cell_from, atom_from), l.atom_pos_list)
        atom_ind_to = findfirst(x -> x == (cell_to, atom_to), l.atom_pos_list)
        if isnothing(atom_ind_from) || isnothing(atom_ind_to) # skip if the atom is not in the lattice
            continue
        else
            atom_cart_from = l.atom_cart_list[atom_ind_from]
            atom_cart_to = l.atom_cart_list[atom_ind_to]

            atom_cart_from_for_plot = [atom_cart_from[i] for i in 1:l.dim]
            atom_cart_to_for_plot = [atom_cart_to[i] for i in 1:l.dim]

            atom_cart_for_plot = [[atom_cart_from_for_plot[i], atom_cart_to_for_plot[i]] for i in 1:l.dim]
            plot!(fig, atom_cart_for_plot...;
                seriestype=:line,
                lc=:black,
                alpha=0.2, # add transparency to visualize overlapping lines
                lw=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 50, # adjust markersize with the number of atoms in the plot
            )
        end
    end

    # add atoms (latter than hopping terms to make atoms on top of hopping terms)
    atom_cart_list_along_each_direction = [[atom_cart[i] for atom_cart in l.atom_cart_list] for i in 1:l.dim]
    plot!(fig, atom_cart_list_along_each_direction...;
        seriestype=:scatter,
        aspect_ratio=:equal,
        legend=false,
        framestyle=:none,
        markersize=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 100, # adjust markersize with the number of atoms in the plot
        ticks=false,
    )
    plot_single_unit_cell!(l, fig)  # plot single unit cell enclosed by bravias vectors

    if save_plot
        figure_dir = "figure/"
        if !ispath(figure_dir)
            mkpath(figure_dir)
        end
        savefig(fig, figure_dir * l.lattice_name * ".pdf")
    end
    return fig
end

"""
Example Struct Initialization of `LatticeModel`
---
add dispatch to include example models of  `lattice_name = square, honeycomb, kagome, Lieb...` and `model_name = graphene, spinless_Haldane...`
"""
function initialize_lattice_model(; lattice_name::String, model_name::String="", sample_size::Vector{Int64}, pbc_indicator::Vector{Bool}, twisted_phases_over_2π::Vector{Float64}=[0.0, 0.0])
    l = initialize_lattice(lattice_name, sample_size)
    lm = initialize_lattice_model(l; pbc_indicator=pbc_indicator, twisted_phases_over_2π=twisted_phases_over_2π)

    (t, μ) = SymEngine.@vars t μ
    @match (lattice_name, model_name) begin
        ("honeycomb", "graphene") => begin
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 1)) => μ)
            add_hopping_term!(lm, (([0, 0], 2), ([0, 0], 2)) => μ)
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 2)) => t)
            add_hopping_term!(lm, (([0, 0], 2), ([1, 0], 1)) => t)
            add_hopping_term!(lm, (([0, 0], 2), ([0, 1], 1)) => t)
        end
        ("honeycomb", "spinless_Haldane") => begin
            # todo!()
        end
        ("kagome", _) => begin
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 1)) => μ)
            add_hopping_term!(lm, (([0, 0], 2), ([0, 0], 2)) => μ)
            add_hopping_term!(lm, (([0, 0], 3), ([0, 0], 3)) => μ)
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 2)) => t)
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 3)) => t)
            add_hopping_term!(lm, (([0, 0], 2), ([0, 0], 3)) => t)
            add_hopping_term!(lm, (([0, 0], 2), ([1, 0], 1)) => t)
            add_hopping_term!(lm, (([0, 0], 3), ([0, 1], 1)) => t)
            add_hopping_term!(lm, (([0, 0], 3), ([-1, 1], 2)) => t)
        end
        ("Lieb", _) => begin
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 1)) => μ)
            add_hopping_term!(lm, (([0, 0], 2), ([0, 0], 2)) => μ)
            add_hopping_term!(lm, (([0, 0], 3), ([0, 0], 3)) => μ)
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 2)) => t)
            add_hopping_term!(lm, (([0, 0], 1), ([0, 0], 3)) => t)
            # add_hopping_term!(lm, (([0, 0], 2), ([0, 0], 3)) => t)
            add_hopping_term!(lm, (([0, 0], 2), ([1, 0], 1)) => t)
            add_hopping_term!(lm, (([0, 0], 3), ([0, 1], 1)) => t)
            # add_hopping_term!(lm, (([0, 0], 3), ([-1, 1], 2)) => t)
        end
        _ => error("Model $model_name on $lattice_name lattice is NOT implemented!")
    end
    update_full_hopping_Hamiltonian!(lm)

    return lm
end


"scan plot of ALL eigenvalues"
function scan_plot_eigvals(eigvals::Vector{Float64}, lm::LatticeModel; save_plot::Bool=true)
    fig = plot(1:length(eigvals), eigvals;
        seriestype=:scatter,
        xlabel="State Index",
        ylabel="Eigenvalue",
        legend=false,
        markersize=5,
        framestyle=:box,
    )
    if save_plot
        figure_dir = "figure/"
        if !ispath(figure_dir)
            mkpath(figure_dir)
        end
        savefig(fig, figure_dir * lm.lattice.lattice_name * "_eigvals_scan.pdf")
    end
    return nothing
end

# plot real-space distribution of eigensates
function plot_eigvec(eigind::Int64, eigvecs::Matrix, lm::LatticeModel; save_plot::Bool=true)
    l = lm.lattice
    fig = show(lm; save_plot=false)

    # adjust each atom color with |ψ|^2
    atom_color = round.(abs2.(eigvecs[:, eigind]), digits=2)
    atom_color /= maximum(atom_color) # normalize to [0,1] for visualization
    for i in eachindex(lm.lattice.atom_cart_list)
        atom_cart = lm.lattice.atom_cart_list[i]
        scatter!(fig, [Tuple(atom_cart)];
            aspect_ratio=:equal,
            legend=false,
            framestyle=:none,
            marker_z=atom_color[i],
            cmap=cgrad(:thermal, rev=false),
            markersize=l.cell_volume * sqrt(reduce(*, l.sample_size)) / length(l.atom_cart_list) * 100, # adjust markersize with the number of atoms in the plot
            ticks=false,
            colorbar=true,
            title="Real-space Distribution of the $eigind-th Eigenstate",
        )
    end

    if save_plot
        figure_dir = "figure/"
        if !ispath(figure_dir)
            mkpath(figure_dir)
        end
        savefig(fig, figure_dir * l.lattice_name * "_state_distribution.pdf")
    end
    return nothing
end

lm = initialize_lattice_model(; lattice_name="Lieb", model_name="", sample_size=[10, 10], pbc_indicator=[true, true])
(t, μ) = SymEngine.@vars t μ
h = convert(Matrix{Float64}, SymEngine.subs.(lm.full_hopping_Hamiltonian, Ref(Dict(μ => 0.0, t => 1.0))))

(eigvals, eigvecs) = eigen(h) #KrylovKit.eigsolve(h, rand(eltype(h), size(h)[1]), 30, :SR)

scan_plot_eigvals(eigvals, lm)
plot_eigvec(102, eigvecs, lm)
# @show round.(eigvecs[:, 150], digits=3)



# spy(h;
#     markersize=5,
#     framestyle=:box,
# ) |> display


end # end module
