using Interpolations
using DataFrames

"""
    create_interp_elev(locs::DataFrame,
                       [t_col::Symbol = :Distance,
                        e_col::Symbol = :SurfaceElevation])

Returns a function that interpolates ground elevation quadratically in the
range of the observed elevations. Outside these observations the elevation
is constant from the nearest observation. The DataFrame needs columns for
transect distance (typically `:Distance`) and elevation (typically
`:SurfaceElevation`) at that transect location. The transect measurements
should be evenly spaced.
"""
function create_interp_elev(locs::DataFrame,
                            t_col::Symbol = :Distance,
                            e_col::Symbol = :SurfaceElevation)
    t_min, t_max = extrema(locs[t_col])
    t_diff = locs[2, t_col] - locs[1, t_col]
    if any(diff(locs[t_col]) .!= t_diff)
        throw("Elevation observations need to be evenly spaced.")
    end

    transect_index(t) = ((t .- t_min) ./ t_diff) + 1.

    elev_int = interpolate(locs[e_col], BSpline(Quadratic(Flat())), OnCell())
    elev_extr = extrapolate(elev_int, Flat())

    function interp_elevation(t)
        elev = Array{Float64, 1}(length(t))
        for i in 1:length(t)
            if t_min .≤ t[i] .≤ t_max
                elev[i] = elev_int[transect_index(t[i])]
            else
                elev[i] = elev_extr[transect_index(t[i])]
            end
        end
        elev
    end
    return interp_elevation
end

"""
    create_interp_lres(res_csv::AbstractString,
                       tsect_name::AbstractString,
                       interp_elev::Function;
                       [res_col = :Resistivit])

Import resistivity data and construct an interpolator for log resistivity.
Returns a function that performs a linear interpolation of log resistivity
within the grid of inverted points.
"""
function create_interp_lres(res_csv::AbstractString,
                            tsect_name::AbstractString,
                            interp_elev::Function;
                            res_col = :Resistivit)
    resist = readtable(res_csv,
                       header = true)
    resist = resist[resist[:Transect] .== tsect_name, :]
    resist[isnan.(resist[res_col]), res_col] = NA

    NA_dists = by(resist, :Distance,
                  df -> DataFrame(NoNA = !any(isna, df[res_col])))
    resist = join(resist, NA_dists, on = :Distance, kind = :left)
    resist = resist[resist[:NoNA], :]

    res_dist = unique(Array{Float64}(resist[:Distance]))
    res_depth = unique(Array{Float64}(resist[:Depth]))
    lres = reshape(Array(log(resist[res_col])),
                   length(res_depth), length(res_dist))

    min_depth = minimum(res_depth)

    lres_int = interpolate((res_depth, res_dist),
                           lres,
                           Gridded(Linear()))

    """
        interp_lres(dist, depth, elev = true)

    This function returns the interpolated log resistivity at the given
    locations as a DataArray. Locations above the minimum depth are returned
    as `NA`s.
    """
    function interp_lres(dist, depth, elev::Bool = true)
        @assert length(dist) == length(depth)
        lres = DataArray{Float64, 1}(zeros(dist))

        if elev
            depth = interp_elev(dist) .- depth
        end

        for (r, (tr, dp)) in enumerate(zip(dist, depth))
            if dp .> min_depth
                lres[r] = lres_int[dp, tr]
            else
                lres[r] = NA
            end
        end
        lres
    end
    return interp_lres
end

"""
    create_interp_lres(dsn::ODBC.DSN,
                       tsect_name::AbstractString,
                       interp_elev::Function;
                       [res_col = "Resistivit"])

Import resistivity data and construct an interpolator for log resistivity.
Returns a function that performs a linear interpolation of log resistivity
within the grid of inverted points.
"""
function create_interp_lres(dsn::ODBC.DSN,
                            tsect_name::AbstractString,
                            interp_elev::Function;
                            res_col = "Resistivit")
    resist = ODBC.query(dsn,
                        """
                        SELECT
                          "Resistivity"."Distance",
                          "Resistivity"."Depth",
                          "Resistivity"."$res_col"
                        FROM "Resistivity"
                        WHERE "Resistivity"."Transect" = '$tsect_name'
                        ORDER BY "Resistivity"."Distance", "Resistivity"."Depth"
                        """)
    resist = nulldf2dadf(resist)

    res_col = Symbol(res_col)
    resist[isnan.(resist[res_col]), res_col] = NA

    NA_dists = by(resist, :Distance,
                  df -> DataFrame(NoNA = !anyna(df[res_col])))
    resist = join(resist, NA_dists, on = :Distance, kind = :left)
    resist = resist[resist[:NoNA], :]

    res_dist = unique(Array{Float64}(resist[:Distance]))
    res_depth = unique(Array{Float64}(resist[:Depth]))
    lres = reshape(Array(log(resist[res_col])),
                   length(res_depth), length(res_dist))

    min_depth = minimum(res_depth)

    lres_int = interpolate((res_depth, res_dist),
                           lres,
                           Gridded(Linear()))

    """
        interp_lres(dist, depth, elev = true)

    This function returns the interpolated log resistivity at the given
    locations as a DataArray. Locations above the minimum depth are returned
    as `NA`s.
    """
    function interp_lres(dist, depth, elev::Bool = true)
        @assert length(dist) == length(depth)
        lres = DataArray{Float64, 1}(zeros(dist))

        if elev
            depth = interp_elev(dist) .- depth
        end

        for (r, (tr, dp)) in enumerate(zip(dist, depth))
            if dp .> min_depth
                lres[r] = lres_int[dp, tr]
            else
                lres[r] = NA
            end
        end
        lres
    end
    return interp_lres
end

