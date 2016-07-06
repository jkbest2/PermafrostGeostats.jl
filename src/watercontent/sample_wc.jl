using GaussianProcessConvolutions
using Distributions
using PDMats
using JLD
using StatsBase
using HDF5
using DataFrames

include("julia/import_CCHRC.jl")
cores[:lwc] = @data([isna(cores[i, :wc]) || !cores[i, :res_avail] ? NA : logit(cores[i, :wc])
                    for i in 1:size(cores, 1)])

include("julia/metrop_wc.jl")

core_select_mask = [Bool(cores[s, :Distance] ∈ 50:3:149) for s in 1:size(cores, 1)]

soil_assoc = Dict{AbstractString, Int}("GW" => 1,
                                       "ML" => 2,
                                       "Pt" => 3,
                                       "SM" => 4,
                                       "SP" => 5,
                                       "SW" => 6,
                                       "ICE" => 7)
cores[:soil_int] = [soil_assoc[cores[i, :USCS_code]] for i in 1:size(cores, 1)]
cores[cores[:soil_int] .== 7, :soil_int] = NA

wc_dat = cores[[:Distance, :samp_elev, :lwc, :log_res]]
complete_cases!(wc_dat)

knot_loc_trans = 45.0:5.0:155.0
knot_loc_elev = 135.0:-1.0:125.0
knot_locs = reduce(vcat, [hcat(tr, el) for el in knot_loc_elev, tr in knot_loc_trans])

kern = SquaredExponentialKernel([14.0, 1.0])

soil_kwt = knot_wt(knot_locs,
                   kern,
                   Array{Float64, 2}(wc_dat[[:Distance, :samp_elev]]))

function interp_soil!(pred::Array{Float64, 2},
                      soil_knots::Array{Float64, 3},
                      kwt::Array{Float64, 2})
    p = Array{Float64, 2}(size(pred, 1), size(soil_knots, 2))
    @showprogress for i in 1:size(soil_knots, 3)
        for j in 1:size(soil_knots, 2)
            p[:, j] = kwt * soil_knots[:, j, i]
        end
        for j in 1:size(pred, 1)
            pred[j, indmax(p[j, :])] += 1.
        end
    end
    pred ./= size(soil_knots, 3)
end

pred = zeros(Float64, size(wc_dat, 1), 6)

kv = h5read("results/soil2.hdf5", "soil_2/knot_values")
interp_soil!(pred, kv, soil_kwt)

wc_dat[:soil_int] = [indmax(pred[i, :]) for i in 1:size(pred, 1)]

# soil_knots = h5read("results/soil.hdf5", "soil_1/knot_values", (:, 10550))
# soil_knots = reshape(soil_knots, 253, 6)
# soil_proc = Array{Float64, 2}(size(soil_kwt, 1), size(soil_knots, 2))
# soil_pred = Array{Int, 1}(size(soil_proc, 1))
# update_proc!(soil_proc, soil_knots, soil_kwt)
# predict_soil!(soil_pred, soil_proc)
# 
# soil_dat = cores[core_select_mask,
#                  [:Distance, :samp_elev, :soil_int, :lwc, :log_res]]

init_vals = Dict{Symbol, AbstractArray}()
init_vals[:β_res] = randn(2, 6)
init_vals[:σ] = randexp(1)

prop_w = Dict{Symbol, Array{Float64, 1}}()
prop_w[:β_res] = fill(0.5, 2 * 6)
prop_w[:σ] = [0.1]

include("julia/metrop_wc.jl")
metrop_wc(knot_locs,
          kern,
          wc_dat,
          results_file = "results/wc.hdf5",
          run_name = "wc_1",
          iters = 1_015_000,
          thin = 100,
          warmup = 15_000,
          finish_adapt = 10_000,
          adapt_every = 500,
          prop_width = prop_w,
          init = init_vals)
