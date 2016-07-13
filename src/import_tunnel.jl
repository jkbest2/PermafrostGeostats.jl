import StatsFuns: logit, logistic
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
    import_transect(loc_csv::AbstractString,
                    core_csv::AbstractString)

Import transect from .csv of borehole locations and .csv of core obervations.
DataFrame of locations joined with observations is returned. `loc_csv`
controls which transect is returned.
"""
function import_transect(loc_csv::AbstractString,
                         core_csv::AbstractString)
    locs = readtable(loc_csv)

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

    join(locs, cores, on = :Point, kind = :left)
end

