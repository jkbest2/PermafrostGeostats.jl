function oneproc_lp(kval::Array{Float64, 1},
                    dat_kwt::Array{Float64, 2},
                    data_vals::BitArray{1},
                    misclass::Float64;
                    prior = Normal(0, 1))

    # Prior
    lp = sum(logpdf(prior, kval))

    corr_lik = log(1 - misclass)
    incorr_lik = log(misclass)

    # Process to predict permafrost *present*
    dat_pred = (dat_kwt * kval) .> 0

    for d in 1:length(dat_pred)
        if dat_pred[d] == data_vals[d]
            lp += corr_lik
        else
            lp += incorr_lik
        end
    end

    lp
end

function metrop_oneproc(knot_locs::Array{Float64, 2},
                        kern::AbstractConvolutionKernel,
                        data;
                        nknots::Integer = size(knot_locs, 1),
                        misclass::Float64 = 0.01,
                        iters::Integer = 2000,
                        warmup::Integer = 1000,
                        finish_adapt::Integer = 800,
                        adapt_every::Integer = 100,
                        prop_width::Array{Float64, 1} =
                                 ones(length(nknots)),
                        init::Array{Float64, 1} =
                                 rand(Normal(0, 1), nknots))

    # Checks
    if misclass < 0.0 || misclass > 1.0
        error("misclass must be in [0, 1]")
    end
    if warmup > iters
        error("Number of iterations must be larger than warmup")
    end
    if length(init) != nknots
        error("Initial knot vector must be correct length")
    end
    if length(prop_width) != nknots
        error("Need a proposal width for each knot")
    end

    # Initial set up
    prop_dist = MvNormal(zeros(nknots), PDiagMat(prop_width))
    knot_samp = Array{Float64, 2}(nknots, iters)
    lp = Array{Float64, 1}(iters)
    prop_lp = 0.0
    knot_adj = Array{Float64, 1}(nknots)

    # Data setup
    data_locs = Array{Float64, 2}(data[:, 1:2])
    data_vals = BitArray{1}(data[:, 3])
    ndat = size(data_locs, 1)

    # Calculate knot weights for each data location
    dat_kwt = Array{Float64, 2}(ndat, nknots)
    for d in 1:ndat
        dat_kwt[d, :] = conv_wt(kern, knot_locs' .- data_locs[d, :]')'
    end

    # Initial knot values and log posteriorelihood
    knot_samp[:, 1] = init
    lp[1] = oneproc_lp(knot_samp[:, 1],
                       dat_kwt,
                       data_vals,
                       misclass)
    println("Iteration 1/", iters, ": lp = ", lp[1])

    # Warmup iterations
    for i in 2:iters
        knot_adj = rand(prop_dist, 1)
        # Change the knot in a random order; ask Margaret exactly why?
        knot_seq = sample(1:nknots, nknots, replace = false)
        knot_samp[:, i] = knot_samp[:, i - 1]
        lp[i] = lp[i - 1]
        for k in knot_seq
            knot_samp[k, i] += knot_adj[k]
            prop_lp = oneproc_lp(knot_samp[:, i],
                                 dat_kwt,
                                 data_vals,
                                 misclass)
            if prop_lp > lp[i]
                # Accept higher-prob sample
                lp[i] = prop_lp
            elseif log(rand(1))[1] > (prop_lp - lp[i])
                # Reject lower-prob sample
                knot_samp[k, i] = knot_samp[k, i - 1]
            else
                # Accept lower-prob sample
                lp[i] = prop_lp
            end
        end
        println("Iteration ", i, "/", iters, ": lp = ", lp[i])

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                if i % adapt_every != 0
                    for k in keys(θ_curr)
                        adapt_log[k][:, i % adapt_every] = vec(θ_curr[k])
                    end
                else
                    adapt_log[:, adapt_every] = vec(θ_curr)
                    adapt_prop_width!(prop_width, adapt_log)
                    prop_dist = MvNormal(zeros(prop_width),
                                         PDiagMat(prop_width))
                    pw_idx = i ÷ finish_adapt + 1
                    pf_knot_pw[:, pw_idx] = prop_width
                end
            end
    end
    knot_samp, lp, prop_width
end
