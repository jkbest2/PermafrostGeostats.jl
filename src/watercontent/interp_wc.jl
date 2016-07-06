using GaussianProcessConvolutions
using HDF5
using JLD
using ProgressMeter

include("julia/log_res_funcs.jl")
interp_lres = create_interp_lres()
interp_elev = create_interp_elev()

int_dist = 45:0.1:155.
int_depth = -0.9:-0.1:-10.

t_elev = interp_elev(int_dist)

int_locs = reduce(vcat, [hcat(d, el) for el in int_depth, d in int_dist])
int_locs[:, 2] = [interp_elev(int_locs[t, 1])[1] + int_locs[t, 2] for
                    t in 1:size(int_locs, 1)]

knot_loc_trans = collect(45.0:5.0:155.0)
knot_loc_elev = collect(135.0:-1.0:125.0)
knot_locs = reduce(vcat, [hcat(t, d) for d in knot_loc_elev, t in knot_loc_trans])

kern = SquaredExponentialKernel([14., 1.])

soil_knots = h5read("results/soil2.hdf5", "soil_2/knot_values")
int_kwt = knot_wt(GaussianProcessConvolution(knot_locs),
                  kern,
                  int_locs)

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

pred = zeros(Float64, size(int_locs, 1), 6)
interp_soil!(pred, soil_knots, int_kwt)

soil_type = Int[indmax(pred[i, :]) for i in 1:size(pred, 1)]
@save "results/soil_for_wc.jld" pred soil_type int_locs

@load "results/soil_for_wc.jld"

β_res = h5read("results/wc.hdf5", "wc_1/β_res")

function interp_lwc!(pred::Array{Float64, 1},
                     β_res::Array{Float64, 3},
                     soil_type::Array{Int, 1},
                     log_res::Array{Float64, 2};
                     summary::Function = mean)
    β = permutedims(β_res, [1, 3, 2])

    @showprogress for l in 1:length(pred)
        pred[l] = summary(log_res[l, :] * β[:, :, soil_type[l]])[1]
    end
    nothing
end

lres = ones(int_locs)
@showprogress for l in 1:size(lres, 1)
    lres[l, 2] = interp_lres(int_locs[l, 1], int_locs[l, 2])[1]
end

lwc_mean = Array{Float64, 1}(size(int_locs, 1))
interp_lwc!(lwc_mean,
            β_res,
            soil_type,
            lres)
wc_mean = logistic(lwc_mean)

lwc_median = Array{Float64, 1}(size(int_locs, 1))
interp_lwc!(lwc_median,
            β_res,
            soil_type,
            lres,
            summary = median)
wc_median = logistic(lwc_median)

lwc_10p = Array{Float64, 1}(size(int_locs, 1))
interp_lwc!(lwc_10p,
            β_res,
            soil_type,
            lres,
            summary = (v) -> quantile(vec(v), [0.1]))
wc_10p = logistic(lwc_10p)

lwc_90p = Array{Float64, 1}(size(int_locs, 1))
interp_lwc!(lwc_90p,
            β_res,
            soil_type,
            lres,
            summary = (v) -> quantile(vec(v), [0.9]))
wc_90p = logistic(lwc_90p)

using PyPlot

figure()
contourf(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
         reshape(wc_mean, length(int_depth), length(int_dist)),
         20, cmap = :viridis)
colorbar(orientation = :horizontal)
contour(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
        reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
        reshape(wc_mean, length(int_depth), length(int_dist)),
        [0.9])
plot(int_dist, t_elev, "k-", lw = 2)
title("Mean water content")

figure()
contourf(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
         reshape(wc_median, length(int_depth), length(int_dist)),
         20, cmap = :viridis)
colorbar(orientation = :horizontal)
contour(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
        reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
        reshape(wc_median, length(int_depth), length(int_dist)),
        [0.9])
plot(int_dist, t_elev, "k-", lw = 2)
title("Median water content")

figure()
contourf(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
         reshape(wc_10p, length(int_depth), length(int_dist)),
         20, cmap = :viridis, hold = false)
colorbar(orientation = :horizontal)
contour(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
        reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
        reshape(wc_10p, length(int_depth), length(int_dist)),
        [0.9])
plot(int_dist, t_elev, "k-", lw = 2)
title("10th percentile water content")

figure()
contourf(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
         reshape(wc_90p, length(int_depth), length(int_dist)),
         20, cmap = :viridis, hold = false)
colorbar(orientation = :horizontal)
contour(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
        reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
        reshape(wc_90p, length(int_depth), length(int_dist)),
        [0.9])
plot(int_dist, t_elev, "k-", lw = 2)
title("90th percentile water content")

figure()
contourf(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
         reshape(wc_90p .- wc_10p, length(int_depth), length(int_dist)),
         20, cmap = :viridis, hold = false)
colorbar(orientation = :horizontal)
# contour(reshape(int_locs[:, 1], length(int_depth), length(int_dist)),
#         reshape(int_locs[:, 2], length(int_depth), length(int_dist)),
#         reshape(wc_90p, length(int_depth), length(int_dist)),
#         [0.9])
plot(int_dist, t_elev, "k-", lw = 2)
title("80% width")
