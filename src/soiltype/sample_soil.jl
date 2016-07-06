using GaussianProcessConvolutions
using Distributions
using PDMats
using JLD

include("julia/import_CCHRC.jl")
include("julia/metrop_soil.jl")

core_select_mask = [Bool(cores[s, :Distance] ∈ 50:3:149) for s in 1:size(cores, 1)]

soil_assoc = Dict{AbstractString, Int}("GW" => 1,
                                       "ML" => 2,
                                       "Pt" => 3,
                                       "SM" => 4,
                                       "SP" => 5,
                                       "SW" => 6,
                                       "ICE" => 7)
cores[:soil_int] = [soil_assoc[cores[i, :USCS_code]] for i in 1:size(cores, 1)]
soil_fit_mask = cores[:soil_int] .≤ 6
soil_dat = cores[soil_fit_mask & core_select_mask, [:Distance, :samp_elev, :soil_int]]

knot_loc_trans = 45.0:5.0:155.0
knot_loc_elev = 135.0:-1.0:125.0
knot_locs = reduce(vcat, [hcat(t, d) for d in knot_loc_elev, t in knot_loc_trans])

kern = SquaredExponentialKernel([14.0, 1.0])

nproc = 6
nknots = size(knot_locs, 1)

# kwt = knot_wt(GaussianProcessConvolution(knot_locs),
#               kern,
#               Array{Float64}(soil_dat[:, 1:2]))
# 
# soil_mat = zeros(size(soil_dat, 1), nproc)
# for i in 1:size(soil_dat, 1)
#     soil_mat[i, soil_dat[i, :soil_int]] = 1.
# end
# 
# kv_init = (kwt' * kwt + I) \ kwt' * soil_mat

using HDF5
soil_file = h5open("results/soil.hdf5")
kv_init = soil_file["soil_1/knot_values"][:, end]
prop_width = soil_file["soil_1/prop_width"][:, end]
close(soil_file)

metrop_soil(knot_locs,
            kern,
            soil_dat,
            nproc,
            results_file = "results/soil2.hdf5",
            run_name = "soil_2",
            nknots = nknots,
            misclass = 0.001,
            iters = 150_000,
            thin = 10,
            warmup = 0,
            finish_adapt = 0,
            adapt_every = 0,
            prop_width = prop_width[:, 1],
            init = reshape(kv_init, 253, 6))
