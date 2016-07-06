using DataFrames
using StatsFuns

# include("julia/log_res_funcs.jl")
include("log_res_funcs.jl")

function transform_gmc(gmc)
    (0.01 .* gmc) ./ (1.0 .+ 0.01 .* gmc)
end

interp_lres = create_interp_lres()
interp_elev = create_interp_elev()

cores = readtable("data/Cores.csv",
                  eltypes = [AbstractString,
                             Float64,
                             AbstractString,
                             Bool,
                             Float64,
                             AbstractString],
                  truestrings = ["T", "t", "TRUE", "true", "1"],
                  falsestrings = ["F", "f", "FALSE", "false", "0"],
                  nastrings = ["NA", "", "N/A"])

boreholes = readtable("data/Boreholes.csv")
resistivity = readtable("data/Resistivity_Values.csv")

min_res_depth = extrema(resistivity[:Depth])[1]

cores = join(cores, boreholes, on = :Point)

pool!(cores, :USCS_code)
cores[:Distance] = [parse(pt[4:6]) for pt in cores[:Point]]
cores[:samp_elev] = cores[:Elevation] - 0.01 * cores[:Depth]
cores[:res_avail] = (0.01 .* cores[:Depth]) .≥ min_res_depth
# All but one "ICE" has NA for moisture content. Setting all but the one
# actually measured to 5000 ≈ 98% water content.
cores[(cores[:USCS_code] .== "ICE") & isna(cores[:GMC]), :GMC] = 5e3
cores[:log_res] = map((dist, elev) -> interp_lres(dist, elev)[1],
                      cores[:Distance],
                      cores[:samp_elev])
cores[!cores[:res_avail], :log_res] = NA
cores[:wc] = map(transform_gmc, cores[:GMC])
