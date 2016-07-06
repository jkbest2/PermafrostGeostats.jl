using Interpolations
using DataFrames

function create_interp_elev()
    cores = readtable("data/Cores.csv")
    boreholes = readtable("data/Boreholes.csv")

    bh_loc = Array{Float64}(size(boreholes, 1), 2)
    for r in 1:size(boreholes, 1)
        bh_loc[r, 1] = parse(boreholes[r, :Point][4:6])
    end
    bh_loc[:, 2] = boreholes[:Elevation]

    transect_index(t) = ((t .- 50.) ./ 3.) + 1.

    elev_int = interpolate(bh_loc[:, 2], BSpline(Quadratic(Flat())), OnGrid())
    elev_extr = extrapolate(elev_int, Flat())

    min_t = findmin(bh_loc[:, 1])[1]
    max_t = findmax(bh_loc[:, 1])[1]

    function interp_elevation(t)
        elev = Array{Float64, 1}(length(t))
        for i in 1:length(t)
            if (t[i] .< min_t) | (t[i] .> max_t)
                elev[i] = elev_extr[transect_index(t[i])]
            else
                elev[i] = elev_int[transect_index(t[i])]
            end
        end
        elev
    end
    return interp_elevation
end

function create_interp_lres()
    resist = readtable("data/Resistivity_Values.csv", header = true)
    tsect_resist = resist[(resist[:Distance] .≥ 40.) & (resist[:Distance] .≤ 160.), :]
    res_dist = unique(Array{Float64}(tsect_resist[:Distance]))
    res_depth = unique(Array{Float64}(tsect_resist[:Depth]))
    lres = reshape(Array(log(tsect_resist[:Resistivity])),
                   length(res_depth), length(res_dist))
    lres_int = interpolate((res_depth, res_dist),
                           lres,
                           Gridded(Linear()))
    interp_elev = create_interp_elev()
    function interp_lres(dist, depth, elev = true)
        if elev
            depth = interp_elev(dist) .- depth
        end
        lres_int[depth, dist]
    end
    return interp_lres
end

