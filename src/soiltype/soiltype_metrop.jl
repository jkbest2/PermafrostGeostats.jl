"""
    soiltype_lp(kval::Array{Float64, 1},
                pred_soil::Array{Integer, 1},
                data_vals::Array{Integer, 1},
                misclass::Float64;
                prior = Normal(0, 1))

Log posterior of the permafrost model, allowing for misclassification.
"""
function soiltype_lp(kval::Array{Float64, 2},
                     pred_soil::Array{Int},
                     data_vals::Array{Int},
                     misclass::Float64;
                     prior = Normal(0, 1))

    # Prior
    lp = loglikelihood(prior, kval)

    corr_lik = log(1 - misclass)
    incorr_lik = log(misclass)

    n_corr = sum(pred_soil .== data_vals)

    lp += n_corr * corr_lik
    lp += (length(data_vals) - n_corr) * incorr_lik

    lp
end

"""
    soiltype_predict!(pred::Vector{Integer},
                      proc::Array{Float64, 2})

Predict soil type (as an integer) from the processes.
"""
function soiltype_predict!{T <: Integer}(pred::Vector{T},
                                         proc::Array{Float64, 2})
    pred[:] = T[indmax(proc[r, :]) for r in 1:size(proc, 1)]::Array{T, 1}
end

"""
    soiltype_update!(idx::Integer,
                     adj::Float64,
                     knots::Array{Float64, 2},
                     proc::Array{Float64, 2},
                     kwt::Array{Float64, 2})

Update knot value and process that depends on it.
"""
function soiltype_update!(idx::Int,
                          adj::Float64,
                          knots::Array{Float64, 2},
                          proc::Array{Float64, 2},
                          kwt::Array{Float64, 2})
    change_proc = cld(idx, size(knots, 1))
    knots[idx] += adj
    proc[:, change_proc] = kwt * knots[:, change_proc]
    nothing
end

"""
function soiltype_metrop(knot_locs::Array{Float64, 2},
                         kern::AbstractConvolutionKernel,
                         data,
                         nproc::Integer = length(unique(soil_data));
                         results_file::AbstractString = "NULL",
                         run_name::AbstractString = "run",
                         nknots::Integer = size(knot_locs, 1),
                         misclass::Float64 = 0.001,
                         iters::Integer = 2000,
                         thin::Integer = 1,
                         warmup::Integer = 1000,
                         finish_adapt::Integer = 800,
                         adapt_every::Integer = 100,
                         prop_width::Array{Float64, 1} =
                                     0.5 * ones(nknots * nproc),
                         init::Array{Float64, 2} =
                                     randn(nknots, nproc),
                         RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

Function for performing Markov Chain Monte Carlo using the Metropolis
algorithm to fit the soil type model. Samples are saved in HDF5 format in
`results_file.hdf5`. Specialized for the soil type model.
"""
function soiltype_metrop(knot_locs::Array{Float64, 2},
                         kern::AbstractConvolutionKernel,
                         data,
                         nproc::Integer = length(unique(soil_data));
                         results_file::AbstractString = "NULL",
                         run_name::AbstractString = "run",
                         nknots::Integer = size(knot_locs, 1),
                         misclass::Float64 = 0.001,
                         iters::Integer = 2000,
                         thin::Integer = 1,
                         warmup::Integer = 1000,
                         finish_adapt::Integer = 800,
                         adapt_every::Integer = 100,
                         prop_width::Array{Float64, 1} =
                                     0.5 * ones(nknots * nproc),
                         init::Array{Float64, 2} =
                                     randn(nknots, nproc),
                         RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

    # Checks
    @assert 0 < misclass < 1 "Misclass rate must be ∈ (0, 1)"
    @assert warmup ≤ iters "Too many warmup iterations"
    @assert size(init) == (nknots, nproc) "Need initial values for all parameters"
    @assert length(prop_width) == length(init) "Need proposal widths for all parameters"

    nparam = nknots * nproc

    # Data setup
    data_locs = Array{Float64, 2}(data[[:Distance, :Elevation]])
    data_soil = Array{Int, 1}(data[:Soil])
    data_kwt = knot_wt(knot_locs,
                       kern,
                       data_locs)
    ndat = size(data_locs, 1)

    # HDF5 setup
    if !isfile(results_file)
        m = "w"
    else
        m = "r+"
    end
    results = h5open(results_file, m)
    g_create(results, run_name)
    run_results = results[run_name]

    try
        # Initial set up
        # prop_dist = MvNormal(zeros(prop_width), PDiagMat(prop_width))
        knot_samp = d_create(run_results, "knot_values",
                             datatype(Float64), dataspace(nknots, nproc, fld(iters, thin)),
                             "chunk", (nknots, nproc, 1))
        curr_knots = init
        prop_knots = copy(init)
        curr_proc = data_kwt * curr_knots
        prop_proc = Array{Float64, 2}(ndat, nproc)
        curr_pred = Array{Int}(ndat)
        soiltype_predict!(curr_pred, curr_proc)
        prop_pred = Array{Int}(ndat)
        lp = d_create(run_results, "lp",
                      datatype(Float64), dataspace(fld(iters, thin), 1))
        curr_lp = soiltype_lp(curr_knots, curr_pred, data_soil, misclass)
        prop_lp = -Inf
        knot_adj = Array{Float64, 1}(nparam)
        knot_seq = Array{Int64, 1}(nparam)

        g_create(run_results, "prop_width")
        pw_log = run_results["prop_width"]
        if finish_adapt > 0
            adapt_log = Array{Float64, 2}(nparam, adapt_every)
            knot_pw_log = d_create(pw_log, "knots",
                          datatype(Float64),
                          dataspace(nparam, finish_adapt ÷ adapt_every + 1),
                          "chunk", (nparam, 1))
            knot_pw_log[:, 1] = prop_width
        else
            pw_log["prop_width/knots"] = prop_width
        end

        knot_idx = collect(1:nparam)

        @showprogress "Sampling..." for i in 1:iters
            knot_adj[:] = randn(RNG, nparam) .* prop_width
            knot_seq[:] = shuffle(RNG, knot_idx)

            for k in knot_seq
                soiltype_update!(k, knot_adj[k],
                                 prop_knots,
                                 prop_proc,
                                 data_kwt)
                soiltype_predict!(prop_pred, prop_proc)
                prop_lp = soiltype_lp(prop_knots,
                                      prop_pred, data_soil,
                                      misclass)

                if (prop_lp ≥ curr_lp) || (prop_lp - curr_lp) > log(rand(RNG))
                    # Accept
                    curr_lp = prop_lp
                    curr_knots[:, :] = prop_knots
                    curr_pred[:] = prop_pred
                else
                    # Reject lower-prob sample
                    prop_knots[:, :] = curr_knots
                end
            end

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                adapt_idx = i % adapt_every
                if adapt_idx != 0
                    adapt_log[:, adapt_idx] = vec(curr_knots)
                else
                    adapt_log[:, end] = vec(curr_knots)
                    adapt_prop_width!(prop_width, adapt_log)
                    knot_pw_log[:, (i ÷ adapt_every) + 1] = prop_width
                end
            end

            # Record thinned samples when appropriate
            if i % thin == 0
                idx = i ÷ thin
                lp[idx, 1] = curr_lp
                knot_samp[:, :, idx] = curr_knots
            end
        end
    finally
        close(results)
    end
    nothing
end
