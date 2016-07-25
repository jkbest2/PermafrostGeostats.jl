logit(x::DataArray) = 1 ./ (1 .+ exp(x))
logistic(p::DataArray) = log(p ./ (1 .- p))

""""
    gmc2wc(gmc, [k = 1])

Convert gravimetric moisture content (0, ∞) to water content (0, 1).
"""
gmc2wc(gmc, k = 1) = k .* gmc ./ (k .* gmc .+ 1)

""""
    wc2gmc(wc, [k = 1])

Convert water content (0, 1) to gravimetric moisture content (0, ∞).
"""
wc2gmc(wc, k = 1) = k .* (wc ./ (1 .- wc))

"""
    import_transect(core_csv::AbstractString,
                    loc_csv::AbstractString)

Import transect from .csv of borehole locations and .csv of core obervations.
DataFrame of locations joined with observations is returned. `loc_csv` controls
which transect is returned. Returns tuple of cores DataFrame and locations
DataFrame.
"""
function import_transect(core_csv::AbstractString,
                         loc_csv::AbstractString)
    locs = readtable(loc_csv)
    if !(:Distance ∈ names(locs))
        locs[:Distance] = [parse(Float64, p[4:6])
                            for p in locs[:Point]]
    end
    locs[:Distance] = DataArray{Float64}(locs[:Distance])
    locs[:SurfaceElevation] = locs[:Elevation]
    locs = locs[[:Point, :Distance, :SurfaceElevation, :Northing, :Easting]]

    cores = readtable(core_csv,
                      eltypes = [AbstractString,
                                 Float64,
                                 AbstractString,
                                 Bool,
                                 Float64,
                                 AbstractString],
                      truestrings = ["T", "t", "TRUE", "true", "1"],
                      falsestrings = ["F", "f", "FALSE", "false", "0"],
                      nastrings = ["NA", "", "N/A"])

    cores = join(locs, cores, on = :Point, kind = :left)

    cores[:Depth] = 0.01 * cores[:Depth]

    cores[:Elevation] = cores[:SurfaceElevation] .- cores[:Depth]
    cores = cores[[:Point, :Distance, :Elevation, :Depth,
                   :PF_code, :USCS_code, :GMC]]

    cores, locs
end

"""
    nullable2dataarray{T <: Real}(nullarray::NullableArray{T, 1})

Converts a NullableArray to a DataArray, inserting an NA where necessary.
For subtypes of Real, also converts to Float64.
"""
function nullable2dataarray{T <: Real}(nullarray::NullableArray{T, 1})
    DataArray(Vector{Float64}(nullarray.values), nullarray.isnull)
end

"""
    nullable2dataarray{T <: WeakRefString}(nullarray::NullableArray{T, 1})
Convert NullableArray of WeakRefString elements to DataArray of UTF8String
elements.
"""
function nullable2dataarray{T <: WeakRefString}(nullarray::NullableArray{T, 1})
    DataArray(Vector{UTF8String}(nullarray.values), nullarray.isnull)
end

"""
    nullable2dataarray{T <: Any}(Nullarray::NullableArray{T, 1})

Converts NullableArray to DataArray, preserving types.
"""
function nullable2dataarray{T <: Any}(nullarray::NullableArray{T, 1})
    DataArray(nullarray.values, nullarrays.isnull)
end

"""
    nulldf2dadf(df::AbstractDataFrame)

Converts a DataFrame with Nullable elements to a DataFrame with DataArray
columns. Also converts numeric columns to Float64.
"""
function nulldf2dadf(nulldf::AbstractDataFrame)
    @assert all([elt <: Nullable for elt in eltypes(nulldf)])
    dadf = DataFrame()
    for col in names(nulldf)
        dadf[col] = nullable2dataarray(nulldf[col])
    end
    dadf
end

"""
    str2bool{T <: AbstractString}(da::AbstractDataArray{T})

Convert a DataArray of strings to Bool. Assumes parse |> Bool will give
reasonable output.
"""
function str2bool{T <: AbstractString}(da::AbstractDataArray{T})
    DataArray(Bool[parse(str) for str in da.data], da.na)
end

"""
    import_transect(dsn::ODBC.DSN,
                    transect_name::AbstractString,
                    use_cores::Tuple)

Import borehole locations and core information and preprocess for modelling.
Returns a DataFrame of all borehole locations and information for the cores at
distances in `use_cores`.
"""
function import_transect(dsn::ODBC.DSN,
                         transect_name::AbstractString,
                         use_boreholes::Tuple)
    loc_query = """
                SELECT
                  "Boreholes".point AS "Point",
                  "Boreholes".distance AS "Distance",
                  "Boreholes".elevation AS "SurfaceElevation",
                  "Boreholes".northing AS "Northing",
                  "Boreholes".easting AS "Easting"
                FROM "Boreholes"
                WHERE "Boreholes".transect = '$transect_name'
                ORDER BY "Boreholes".distance
                """
    locs = ODBC.query(dsn, loc_query)
    loc_df = nulldf2dadf(locs)

    core_query = """
                 SELECT
                   "Cores".point as "Point",
                   "Boreholes".distance AS "Distance",
                   "Boreholes".elevation - 0.01 * "Cores".depth AS "Elevation",
                   "Boreholes".elevation AS "SurfaceElevation",
                   "Cores".gmc AS "GMC",
                   "Cores".pf_code AS "PF_code",
                   "Cores".uscs_code AS "USCS_code",
                   0.01 * "Cores".depth AS "Depth"
                 FROM
                   public."Cores",
                   public."Boreholes"
                 WHERE
                   "Cores".point = "Boreholes".point
                   AND "Boreholes".transect = '$transect_name'
                   AND "Boreholes".distance IN $use_boreholes
                 ORDER BY "Boreholes".distance, "Cores".depth;
                 """
    cores = ODBC.query(dsn, core_query)
    core_df = nulldf2dadf(cores)
    core_df[:PF_code] = str2bool(core_df[:PF_code])
    core_df, loc_df
end

"""
    import_transect(dsn::ODBC.DSN,
                    transect_name::AbstractString)

Import borehole locations and core information and preprocess for modelling.
Returns a DataFrame of all borehole locations and cores in the transect.
"""
function import_transect(dsn::ODBC.DSN,
                         transect_name::AbstractString)
    loc_query = """
                SELECT
                  "Boreholes".point AS "Point",
                  "Boreholes".distance AS "Distance",
                  "Boreholes".elevation AS "SurfaceElevation",
                  "Boreholes".northing AS "Northing",
                  "Boreholes".easting AS "Easting"
                FROM "Boreholes"
                WHERE "Boreholes".transect = '$transect_name'
                ORDER BY "Boreholes".distance
                """
    locs = ODBC.query(dsn, loc_query)
    loc_df = nulldf2dadf(locs)

    core_query = """
                 SELECT
                   "Cores".point as "Point",
                   "Boreholes".distance AS "Distance",
                   "Boreholes".elevation - 0.01 * "Cores".depth AS "Elevation",
                   "Boreholes".elevation AS "SurfaceElevation",
                   "Cores".gmc AS "GMC",
                   "Cores".pf_code AS "PF_code",
                   "Cores".uscs_code AS "USCS_code",
                   0.01 * "Cores".depth AS "Depth"
                 FROM
                   public."Cores",
                   public."Boreholes"
                 WHERE
                   "Cores".point = "Boreholes".point
                   AND "Boreholes".transect = '$transect_name'
                 ORDER BY "Boreholes".distance, "Cores".depth;
                 """
    cores = ODBC.query(dsn, core_query)
    core_df = nulldf2dadf(cores)
    core_df[:PF_code] = str2bool(core_df[:PF_code])
    core_df, loc_df
end

"""
    transform_gmc(cores::DataFrame,
                  [k = 0.01,
                   col = :GMC,
                   soil_col = :USCS_code,
                   ice_gmc = 2e3])

Transforms `:GMC` column of passed DataFrame to logistic scale. Gravimetric
moisture content is on a [0, ∞) scale, as it is the ratio of water mass to dry
mass. The values we are importing are percentages, thus the `k = 0.01` factor.
This is first transformed to water content as a percent of total weight
(on a [0, 1] scale), an then to a logistic (on a (-∞, ∞) scale) for modeling
purposes.

Missing gravimetric moisture content for observations of massive ice will be
imputed to the value specified by `ice_gmc`. The default is 2,000% gravimetric
moisture content, equivalent to approximately 95% water content.
"""
function transform_gmc(cores::DataFrame, k = 0.01,
                       col = :GMC,
                       soil_col = :USCS_code,
                       ice_gmc = 2e3)
    imp_gmc = map(s-> ismatch(r"ICE", s), cores[soil_col])
    imp_gmc = BitArray(imp_gmc & isna(cores[col]))
    cores[imp_gmc, col] = ice_gmc

    cores[col] |> gmc -> gmc2wc(gmc, k) |> logistic
end

"""
    type SoilMapping# <: Associative

Contains fields that define the association between soil observations as strings
and the integers that are used to model them. Note that these are not always
1:1 associations.
"""
type SoilMapping# <: Associative
    soil_int::Dict{AbstractString, Integer}
    int_soil::Dict{Integer, AbstractString}
end

getindex(sm::SoilMapping, idx::AbstractString) = sm.soil_int[idx]
getindex(sm::SoilMapping, idx::Integer) = sm.int_soil[idx]
haskey(SM::SoilMapping, key) = haskey(SM.soil_int, key) || haskey(SM.int_soil, key)

"""
    soil2int(cores::DataFrame,
             [col::Symbol = :USCS_code,
              ice::AbstractString = "ICE"])

Converts soil type string identifiers to integers for modeling convenience.
Sorts so that "ICE" type is last. Combination classifications (split by '+')
are converted to the non-"ICE" type or the first listed if neither is "ICE".

Optional arguments specify the column where soil types are recorded, and the
string the identifies massive ice observations.

Also returns a SoilMapping object that can be used to convert between string
and integer representations of the soil types.
"""
function soil2int(cores::DataFrame,
                  col::Symbol = :USCS_code,
                  ice::AbstractString = "ICE")
    soil_str = unique(cores[col])
    s2 = Vector{AbstractString}(length(soil_str))
    for (i, s) in enumerate(soil_str)
        sp = split(s, '+')
        if length(sp) > 1 && any(sp .== ice)
            s2[i] = sp[sp .!= ice][1]
        elseif length(sp) > 1
            s2[i] = sp[1]
        else
            s2[i] = s
        end
    end

    simp_dict = Dict(zip(soil_str, s2))

    s2 = unique(s2)
    if ice ∈ s2
        splice!(s2, findin(s2, [ice])[1])
        append!(s2, [ice])
    end

    d2 = indexmap(s2)

    int_str = Dict{Integer, AbstractString}()
    for (k, v) in d2
        int_str[v] = k
    end
    str_int = Dict{AbstractString, Integer}()
    for (k, v) in simp_dict
        str_int[k] = d2[v]
    end
    soil_map = SoilMapping(str_int, int_str)

    soil_int = map(s -> soil_map[s], cores[col])

    soil_int, soil_map
end
