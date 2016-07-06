using ProgressMeter
using HDF5

function soil_lp(kval::Array{Float64, 2},
                 pred_soil::Array{Int},
                 data_vals::Array{Int},
                 misclass::Float64;
                 prior = Normal(0, 1))

    # Prior
    lp = sum(logpdf(prior, kval))

    corr_lik = log(1 - misclass)
    incorr_lik = log(misclass)

    n_corr = sum(pred_soil .== data_vals)

    lp += n_corr * corr_lik
    lp += (length(data_vals) - n_corr) * incorr_lik

    lp
end

function predict_soil!(pred::Vector{Int},
                       proc_val::Array{Float64, 2})
    pred[:] = [indmax(proc_val[r, :]) for r in 1:size(proc_val, 1)]
end

function update_procs!(change_ind::Int,
                       change_adj::Float64,
                       knots::Array{Float64, 2},
                       procs::Array{Float64, 2},
                       kwt::Array{Float64, 2})
    change_proc = cld(change_ind, size(knots, 1))
    knots[change_ind] += change_adj
    procs[:, change_proc] = kwt * knots[:, change_proc]
    nothing
end

function metrop_soil(knot_locs::Array{Float64, 2},
                     kern::AbstractConvolutionKernel,
                     soil_data,
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
                                 randn(nknots, nproc))

    # Checks
    if misclass < 0.0 || misclass > 1.0
        error("misclass must be in (0, 1)")
    end
    if warmup > iters
        error("Number of iterations must be larger than warmup")
    end
    if size(init) != (nknots, nproc)
        error("Initial knot vector must be correct length")
    end
    if length(prop_width) != nknots * nproc
        error("Need a proposal width for each knot")
    end

    nparam = nknots * nproc

    # Data setup
    data_locs = Array{Float64, 2}(soil_data[:, 1:2])
    soil_types = Array{Int, 1}(soil_data[:, 3])
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
        prop_dist = MvNormal(zeros(prop_width), PDiagMat(prop_width))
        knot_samp = d_create(run_results, "knot_values",
                             datatype(Float64), dataspace(nknots, nproc, fld(iters, thin)),
                             "chunk", (nknots, nproc, 1))
        curr_knots = init
        prop_knots = copy(init)
        lp = d_create(run_results, "lp",
                      datatype(Float64), dataspace(fld(iters, thin), 1))
        curr_lp = 0.0
        prop_lp = 0.0
        curr_proc = Array{Float64, 2}(ndat, nproc)
        prop_proc = Array{Float64, 2}(ndat, nproc)
        curr_pred = Array{Int}(ndat)
        prop_pred = Array{Int}(ndat)

        knot_seq = Array{Int64, 1}(nparam)

        if finish_adapt > 0
            adapt_log = Array{Float64, 2}(nparam, adapt_every)
            pw = d_create(run_results, "prop_width",
                          datatype(Float64),
                          dataspace(nparam, fld(finish_adapt, adapt_every) + 1),
                          "chunk", (nparam, 1))
            pw[:, 1] = prop_width
        else
            pw = d_create(run_results, "prop_width",
                          datatype(Float64),
                          dataspace(nparam, 1),
                          "chunk", (nparam, 1))
            pw[:, 1] = prop_width
        end

        # Calculate knot weights for each data location
        dat_kwt = knot_wt(GaussianProcessConvolution(knot_locs),
                          kern,
                          data_locs)

        curr_proc[:, :] = dat_kwt * curr_knots
        prop_proc[:, :] = curr_proc
        predict_soil!(curr_pred, curr_proc)

        curr_lp = soil_lp(curr_knots, curr_pred, soil_types, misclass)

        @showprogress "Sampling..." for i in 1:iters
            knot_adj = rand(prop_dist, 1)
            # Update knot values in random order to preserve detailed balance
            knot_seq = sample(1:nparam, nparam, replace = false)

            for k in knot_seq
                update_procs!(k, knot_adj[k],
                              prop_knots,
                              prop_proc,
                              dat_kwt)
                predict_soil!(prop_pred, prop_proc)
                prop_lp = soil_lp(prop_knots, prop_pred, soil_types, misclass)

                if (prop_lp ≥ curr_lp) || (prop_lp - curr_lp) > log(rand(1))[1]
                    # Accept
                    curr_lp = prop_lp
                    curr_knots[:, :] = prop_knots
                    curr_pred[:] = prop_pred
                else
                    # Reject lower-prob sample
                    prop_knots[:, :] = curr_knots
                end
            end

            # println("iteration: ", i, "/", iters, " current: ", curr_lp, " correct: ", sum(curr_pred .== soil_types), "/1999")

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                adapt_idx = i % adapt_every
                if adapt_idx != 0
                    adapt_log[:, adapt_idx] = vec(curr_knots)
                else
                    adapt_log[:, end] = vec(curr_knots)
                    adapt_prop_width!(prop_width, adapt_log)
                    pw[:, (i ÷ adapt_every) + 1] = prop_width
                    prop_dist = MvNormal(zeros(prop_width),
                                         PDiagMat(prop_width))
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

function accept_rate(samples::AbstractArray)
    I, J = size(samples)
    accept = Array{Float64}(zeros(I))
    for i in 1:I
        for j in 2:J
            if samples[i, j] != samples[i, j - 1]
                accept[i] += 1
            end
        end
    end
    accept ./ (J - 1)
end

# "Borrowed" from Lora.jl's proposal width tuner
function erf_rate_score(x::AbstractArray, k::Real = 3.)
    erf(k .* x) .+ 1
end

function adapt_prop_width!(pw::Array{Float64, 1},
                           samples::Array{Float64, 2},
                           target_rate::Float64 = 0.44)
    acc = accept_rate(samples)
    pw[:] = pw .* erf_rate_score(acc .- target_rate)
    nothing
end
