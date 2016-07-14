module BoreholePaperPackage
    using DataFrames, Distributions
    using GaussianProcessConvolutions
    using StatsFuns
    using HDF5, JLD
    using ProgressMeter
    using PyPlot

    include("import_transect.jl")
    include("mcmc_adapt.jl")
    include("interpolators.jl")
    # include("permafrost/metrop_permafrost.jl")
    # include("soiltype/metrop_permafrost.jl")
    # include("watercontent/metrop_watercontent.jl")
end
