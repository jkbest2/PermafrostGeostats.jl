#__precompile__()

module PermafrostGeostats

    using DataFrames, Distributions
    using GaussianProcessConvolutions
    using StatsFuns
    using HDF5, JLD
    using ProgressMeter
    using Interpolations
    using PyPlot
    using ODBC
    using NullableArrays, WeakRefStrings

    import Base: getindex, haskey
    import StatsFuns: logit, logistic

    export
        # import_transect.jl
        logit,
        logistic,
        gmc2wc,
        wc2gmc,
        import_transect,
        transform_gmc,
        SoilMapping,
        getindex,
        haskey,
        soil2int,
        # interpolators.jl
        create_interp_elev,
        create_interp_lres,
        # mcmc_adapt.jl
        accept_rate,
        erf_rate_score,
        adapt_prop_width!,
        # permafrost/permafrost_metrop.jl
        permafrost_lp,
        permafrost_update!,
        permafrost_metrop,
        # permafrost/permafrost_interp.jl
        permafrost_interp,
        # soiltype/soiltype_metrop.jl
        soiltype_lp,
        soiltype_update!,
        soiltype_metrop,
        # soiltype/soiltype_interp.jl
        soiltype_interp,
        # watercontent/watercontent_metrop.jl
        watercontent_lp,
        watercontent_update!,
        watercontent_metrop,
        # watercontent/watercontent_interp.jl
        watercontent_interp,
        # BoreholePaperPackage.jl
        locgrid

    include("import_transect.jl")
    include("mcmc_adapt.jl")
    include("interpolators.jl")
    include("setup_logs.jl")
    include("permafrost/permafrost_metrop.jl")
    include("permafrost/permafrost_interp.jl")
    include("soiltype/soiltype_metrop.jl")
    include("soiltype/soiltype_interp.jl")
    include("watercontent/watercontent_metrop.jl")
    include("watercontent/watercontent_interp.jl")

    """
        locgrid(X::FloatRange, Y::FloatRange)

    Returns a 2-column Array of the Cartesian product of `X` and `Y`.
    """
    function locgrid(X::FloatRange, Y::FloatRange)
        reduce(vcat,
               [hcat(x, y) for y in Y, x in X])
    end
end
