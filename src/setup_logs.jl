"""
    setup_sample_logs{T <: AbstractArray}(
                      results_file::AbstractString,
                      run_name::AbstractString,
                      θ::Dict{Symbol, T},
                      iters::Integer,
                      thin::Integer,
                      warmup::Integer,
                      adapt_every::Integer,
                      finish_adapt::Integer)

Set up a dictionary of HDF5 data objects to write thinned samples to.
Also writes number of samples, and number of warmup samples, as well as
the proposal width log.
"""
function setup_sample_logs{T <: AbstractArray}(
                           run_results::HDF5.HDF5Group,
                           run_name::AbstractString,
                           θ::Dict{Symbol, T},
                           iters::Integer,
                           thin::Integer,
                           warmup::Integer)
    g_create(run_results, "meta")
    run_results["meta/n_warmup_samp"] = fld(warmup, thin)
    run_results["meta/n_samp"] = fld(iters, thin)

    nsave = fld(iters, thin)

    samples = Dict{Symbol, HDF5.HDF5Dataset}
    for (k, v) in θ
        dat_size = size(v)
        samples[k] = d_create(run_results, string(k),
                              datatype(Float64),
                              dataspace(dat_size..., nsave),
                              "chunk", (dat_size..., 1))
    end
    samples[:lp] = d_create(run_results, "lp",
                            datatype(Float64),
                            dataspace(1, nsave))

    samples
end

"""
    setup_pw_log{T <: AbstractArray}(
                 results::HDF5.HDF5Group,
                 run_name::AbstractString,
                 prop_width::Dict{Symbol, T},
                 finish_adapt::Integer,
                 adapt_every::Integer)

Set up proposal width logging in HDF5 file. Records initial proposal widths.
Returns a dictionary of parameter symbols and associated HDF5 dataset.
"""
function setup_pw_log{T <: AbstractArray}(
                      results::HDF5.HDF5Group,
                      run_name::AbstractString,
                      prop_width::Dict{Symbol, T},
                      finish_adapt::Integer,
                      adapt_every::Integer)
    g_create(results, "prop_width")
    pw_record = run_results["prop_width"]

    n_adapt = fld(finish_adapt, adapt_every)
    pw_log = Dict{Symbol, HDF5.HDF5Dataset}()
    for (k, v) in prop_width
        pw_log[k] = d_create(pw_record, string(k),
                             datatype(Float64),
                             dataspace(length(v), n_adapt + 1),
                             "chunk", (length(v), 1))
        pw_log[k][:, 1] = prop_width[k]
    end
    pw_log
end

"""
    setup_adapt_log{T <: AbstractArray}(
                    prop_width::Dict{Symbol, T},
                    finish_adapt::Integer,
                    adapt_every::Integer)

Set up a dictionary to record every sample within an adaptation window so that
acceptance rates can be calculated and proposal widths adjusted accordingly.
"""
function setup_adapt_log{T <: AbstractArray}(
                         prop_width::Dict{Symbol, T},
                         finish_adapt::Integer,
                         adapt_every::Integer)
    n_adapt = fld(finish_adapt, adapt_every)
    if n_adapt > 0
        adapt_log = Dict{Symbol, Array}()
        for (k, v) in prop_width
            adapt_log[k] = Array{Float64, 2}(length(v), adapt_every)
        end
        return adapt_log
    else
        return nothing
    end
end

