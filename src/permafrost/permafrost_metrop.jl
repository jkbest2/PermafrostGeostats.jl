"""
    permafrost_lp(kval::Array{Float64, 1},
                  pf_pred::BitArray{1},
                  pf_data::BitArray{1},
                  misclass::Float64;
                  prior = Normal(0, 1))

Log posterior of the permafrost model, allowing for misclassification.
"""
function permafrost_lp(kval::Array{Float64, 1},
                       pf_pred::BitArray{1},
                       pf_data::BitArray{1},
                       misclass::Float64;
                       prior = Normal(0, 1))

    # Prior
    lp = loglikelihood(prior, kval)

    corr_lik = log(1 - misclass)
    incorr_lik = log(misclass)

    n_corr = sum(pf_pred .== pf_data)
    lp += n_corr * corr_lik
    lp += (length(pf_data) - n_corr) * incorr_lik

    lp
end

"""
    permafrost_update!(idx::Integer,
                       adj::Float64,
                       kval::Vector{Float64},
                       pred::BitArray{1},
                       kwt::Array{Float64, 2})

Updates predictions of permafrost in-place.
"""
function permafrost_update!(idx::Integer,
                            adj::Float64,
                            kval::Vector{Float64},
                            pred::BitArray{1},
                            kwt::Array{Float64, 2})
    kval[idx] += adj
    pred[:] = (kwt * kval) .> 0
end

"""
    permafrost_metrop(knot_locs::Array{Float64, 2},
                      kern::AbstractConvolutionKernel,
                      data;
                      results_file::AbstractString = "temp.hdf5",
                      run_name::AbstractString = "run",
                      nknots::Integer = size(knot_locs, 1),
                      misclass::Float64 = 1e-3,
                      iters::Integer = 2000,
                      thin::Integer = 1,
                      warmup::Integer = 1000,
                      finish_adapt::Integer = 800,
                      adapt_every::Integer = 100,
                      prop_width::Array{Float64, 1} =
                               fill(0.5, length(nknots)),
                      init::Array{Float64, 1} =
                               randn(nknots),
                      RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

Function for performing Markov Chain Monte Carlo using the Metropolis
algorithm. Samples are saved in HDF5 format in `run_name.hdf5`. Specialized
for the permafrost model.
"""
function permafrost_metrop(knot_locs::Array{Float64, 2},
                           kern::AbstractConvolutionKernel,
                           data;
                           results_file::AbstractString = "temp.hdf5",
                           run_name::AbstractString = "run",
                           nknots::Integer = size(knot_locs, 1),
                           misclass::Float64 = 1e-3,
                           iters::Integer = 2000,
                           thin::Integer = 1,
                           warmup::Integer = 1000,
                           finish_adapt::Integer = 800,
                           adapt_every::Integer = 100,
                           prop_width::Array{Float64, 1} =
                                    fill(0.5, length(nknots)),
                           init::Array{Float64, 1} =
                                    randn(nknots),
                           RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

    # Checks
    @assert 0 < misclass < 1 "Misclass rate must be ∈ (0, 1)"
    @assert warmup ≤ iters "Too many warmup iterations"
    @assert length(init) == nknots "Need initial values for all parameters"
    @assert length(prop_width) == nknots "Need proposal widths for all parameters"

    # Data setup
    data_locs = Array{Float64, 2}(data[[:Distance, :Elevation]])
    data_vals = BitArray{1}(data[:PF_code])
    data_kwt = knot_wt(knot_locs,
                       kern,
                       data_locs)
    ndata = size(data, 1)

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
        # prop_dist = MvNormal(zeros(nknots), PDiagMat(prop_width))
        knot_samp = d_create(run_results, "knots",
                             datatype(Float64), dataspace(nknots, fld(iters, thin)),
                             "chunk", (nknots, 1))
        curr_knots = init
        prop_knots = copy(init)
        curr_pred = BitArray{1}(ndata)
        curr_pred  = (data_kwt * curr_knots) .> 0
        prop_pred = BitArray{1}(ndata)
        lp = d_create(run_results, "lp",
                      datatype(Float64), dataspace(fld(iters, thin), 1))
        curr_lp = 0.0
        curr_lp = permafrost_lp(curr_knots,
                                curr_pred,
                                data_vals,
                                misclass)
        prop_lp = -Inf
        knot_adj = Array{Float64, 1}(nknots)
        knot_seq = Array{Integer, 1}(nknots)

        if finish_adapt > 0
            adapt_log = Array{Float64, 2}(nknots, adapt_every)
            g_create(run_results, "prop_width")
            pw_log = run_results["prop_width"]
            knot_pw_log = d_create(pw_log, "knots",
                                   datatype(Float64),
                                   dataspace(nknots,
                                             finish_adapt ÷ adapt_every + 1),
                                    "chunk", (nknots, 1))
            knot_pw_log[:, 1] = prop_width
        else
            pw_log["prop_width/knots"] = prop_width
        end

        knot_idx = collect(1:nknots)
        @showprogress "Sampling..." for i in 1:iters
            knot_adj[:] = prop_width .* randn(RNG, nknots)
            knot_seq[:] = shuffle(RNG, knot_idx)

            for k in knot_seq
                permafrost_update!(k,
                                   knot_adj[k],
                                   prop_knots,
                                   prop_pred,
                                   data_kwt)
                prop_lp = permafrost_lp(prop_knots,
                                        prop_pred,
                                        data_vals,
                                        misclass)

            if (prop_lp ≥ curr_lp) || (prop_lp - curr_lp) > log(rand(RNG))
                    # Accept
                    curr_lp = prop_lp
                    curr_knots[:] = prop_knots
                    curr_pred[:] = prop_pred
                else
                    # Reject lower-prob sample
                    prop_knots[:] = curr_knots
                end
            end

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                if i % adapt_every != 0
                    adapt_log[:, i % adapt_every] = curr_knots
                else
                    adapt_log[:, adapt_every] = curr_knots
                    adapt_prop_width!(prop_width, adapt_log)
                    pw_idx = i ÷ finish_adapt + 1
                    knot_pw_log[:, pw_idx] = prop_width
                end
            end

            # Record thinned samples when appropriate
            if i % thin == 0
                idx = i ÷ thin
                lp[idx, 1] = curr_lp
                knot_samp[:, idx] = curr_knots
            end
        end
    finally
        close(results)
    end
    nothing
end
