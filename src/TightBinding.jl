module TightBinding

using LinearAlgebra
using MLStyle
using CairoMakie
using Test



include("real_space_lattice.jl")
export Real_Space_Lattice, initialize_real_space_lattice

include("real_space_tb_model.jl")
export Real_Space_TightBinding_Model, initialize_real_space_tightbinding_model, add_hopping_term!, plot_real_space_tightbinding_model

include("uniform_grids.jl")
export Uniform_Grids, initialize_uniform_grids

include("utils.jl")
export dual_basis_vec_mat, dual_basis_vec_list, plot_bands






end # module TightBinding
