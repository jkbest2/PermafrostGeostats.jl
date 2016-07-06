using GaussianProcessConvolutions
using DataFrames
using Distributions
using PyPlot
using PDMats
using JLD

# Include 1-process sampler functions
include("metrop_1proc_fast.jl")

cores = readtable("data/Cores.csv")
boreholes = readtable("data/Boreholes.csv")

cores = join(cores, boreholes, on = :Point)
cores[:Distance] = 0.0
cores[:samp_elev] = 0.0
for n in 1:nrow(cores)
    cores[n, :Distance] = parse(cores[n, :Point][4:6])
end
cores[:samp_elev] = cores[:Elevation] - 0.01 * cores[:Depth]

pf_status = cores[[:Distance, :samp_elev, :PF_code]]

knot_loc_trans = collect(45.0:5.0:155.0)
n_trans = length(knot_loc_trans)
knot_loc_elev = collect(135.0:-1.0:125.0)
n_elev = length(knot_loc_elev)

knot_locs = Array{Float64, 2}(n_trans * n_elev, 2)
knot_locs[:, 1] = repeat(knot_loc_trans, inner = [n_elev], outer = [1])
knot_locs[:, 2] = repeat(knot_loc_elev, inner = [1], outer = [n_trans])

kern = SquaredExponentialKernel([14.0, 1.0])

nknots = size(knot_locs, 1)

## Warmup iterations
k_1proc, lp_1proc, pw_1proc = metrop_oneproc(knot_locs,
                                    kern,
                                    pf_status,
                                    nknots = size(knot_locs, 1),
                                    misclass = 0.001,
                                    iters = 5_000,
                                    warmup = 5000,
                                    finish_adapt = 5000,
                                    adapt_every = 200,
                                    prop_width = 0.5 * ones(nknots))

save("results/one_proc/0_1proc.jld",
     "k_1proc", k_1proc,
     "lp_1proc", lp_1proc,
     "pw_1proc", pw_1proc)

k_init = k_1proc[:, end]

k_1proc = nothing
lp_1proc = nothing

# Good iterations
for i in 1:5
    k_1proc, lp_1proc, pw_1proc = metrop_oneproc(knot_locs,
                                        kern,
                                        pf_status,
                                        nknots = size(knot_locs, 1),
                                        misclass = 0.001,
                                        iters = 200_000,
                                        warmup = 0,
                                        finish_adapt = 1,
                                        adapt_every = 200_001,
                                        prop_width = pw_1proc,
                                        init = k_init)

    save(string("results/one_proc/", i, "_1proc.jld"),
         "k_1proc", k_1proc,
         "lp_1proc", lp_1proc,
         "pw_1proc", pw_1proc)

    k_init = k_1proc[:, end]

    k_1proc = nothing
    lp_1proc = nothing
    gc()
end

# Read and thin sequentially

k_1proc_thin = Array{Float64, 2}(nknots, 10_000)
lp_1proc_thin = Array{Float64, 1}(10_000)
thin_index = 1:100:200_000

for i in 1:5
    k_1proc = load(string("results/one_proc/", i, "_1proc.jld"), "k_1proc")
    k_1proc_thin[:, ((i - 1) * 2000 + 1):(i * 2000)] = k_1proc[:, thin_index]
    k_1proc = nothing
    lp_1proc = load(string("results/one_proc/", i, "_1proc.jld"), "lp_1proc")
    lp_1proc_thin[((i - 1) * 2000 + 1):(i * 2000)]= lp_1proc[thin_index]
    lp_1proc = nothing
    gc()
end

save("results/one_proc/thin_1proc.jld",
     "k_1proc", k_1proc_thin,
     "lp_1proc", lp_1proc_thin,
     "knot_locs", knot_locs)

include("prob_frozen.jl")
int_locs = interp_locs(knot_locs)
k_wt = knot_wt(GaussianProcessConvolution(knot_locs,
                                          ones(nknots)),
               kern,
               int_locs)

prob_pf_1proc = prob_frozen_1proc(k_1proc_thin,
                            k_wt)

save("results/one_proc/post_1proc.jld",
     "int_locs", int_locs,
     "prob_pf_1proc", prob_pf_1proc)

