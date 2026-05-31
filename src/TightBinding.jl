module TightBinding

using LinearAlgebra
using MLStyle
using CairoMakie
using Test

using Graphs


include("real_space_lattice.jl")
export Real_Space_Lattice, initialize_real_space_lattice, plot_real_space_lattice

include("real_space_tb_model.jl")
export Real_Space_TightBinding_Model, initialize_real_space_tightbinding_model, add_hopping_term!, add_hoppings_by_graph_distance!, plot_real_space_tightbinding_model

include("uniform_grids.jl")
export Uniform_Grids, initialize_uniform_grids

include("utils.jl")
export dual_basis_vec_mat, dual_basis_vec_list, build_Hk_crys, Chern_number_Fukui_Hatsugai_Suzuki, build_Hamiltonian_matrix, many_body_Chern_number_Fukui_Hatsugai_Suzuki

include("band_plot.jl")
export plot_bands, plot_band_counter, find_1st_BZ_k_cart_list



end # module TightBinding
