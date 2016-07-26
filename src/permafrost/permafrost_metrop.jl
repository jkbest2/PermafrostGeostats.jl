"""
    permafrost_lp(θ::Dict{Symbol, AbstractArray},
                  pf_pred::BitArray{1},
                  pf_data::Dict{Symbol, AbstractArray}
                  misclass::Float64;
                  prior = Dict{Symbol, Distribution}(
                                    :knots => Normal(0, 1),
                                    :β => MvNormal([-2., 0.5],
                                                   diagm([0.5, 0.25]))

Log posterior of the permafrost model, allowing for misclassification.
"""
function  permafrost_lp(θ::Dict{Symbol, AbstractArray},
                        pf_pred::BitArray{1},
                        pf_data::Dict{Symbol, AbstractArray},
                        misclass::Float64;
                        prior = Dict{Symbol, Distribution}(
                                          :knots => Normal(0, 1),
                                          :β => MvNormal([-2., 0.5],
                                                         diagm([0.5, 0.25]))))
    # Prior
    lp = loglikelihood(prior[:knots], θ[:knots])
    lp += logpdf(prior[:β], θ[:β])

    corr_lik = log(1 - misclass)
    incorr_lik = log(misclass)

    n_corr = sum(pf_pred .== pf_data[:pf])
    lp += n_corr * corr_lik
    lp += (length(pf_data) - n_corr) * incorr_lik

    lp
end

"""
    permafrost_update!(param::Symbol,
                       idx::Integer,
                       adj::Float64,
                       θ::Dict{Symbol, AbstractArray},
                       proc::Dict{Symbol, AbstractArray},
                       pred::BitArray{1},
                       kwt::Array{Float64, 2})

Updates predictions of permafrost in-place.
"""
function permafrost_update!(param::Symbol,
                            idx::Integer,
                            adj::Float64,
                            θ::Dict{Symbol, AbstractArray},
                            proc::Dict{Symbol, AbstractArray},
                            pred::BitArray{1},
                            kwt::Array{Float64, 2},
                            data::Dict{Symbol, AbstractArray})
    if param == :knots
        θ[:knots][idx] += adj
        proc[:spat][:] = kwt * θ[:knots]
    elseif param == :β
        θ[:β][idx] += adj
        proc[:reg][:] = data[:lres] * θ[:β]
    end
    pred[:] = (proc[:spat] .+ proc[:reg]) .> 0
end

"""
    permafrost_update!(θ::Dict{Symbol, AbstractArray},
                       proc::Dict{Symbol, AbstractArray},
                       pred::BitArray{1},
                       kwt::Array{Float64, 2})

Updates predictions of permafrost in-place.
"""
function permafrost_update!(θ::Dict{Symbol, AbstractArray},
                            proc::Dict{Symbol, AbstractArray},
                            pred::BitArray{1},
                            kwt::Array{Float64, 2},
                            data::Dict{Symbol, AbstractArray})
    proc[:spat][:] = kwt * θ[:knots]
    proc[:reg][:] = data[:lres] * θ[:β]
    pred[:] = (proc[:spat] .+ proc[:reg]) .> 0
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
                      prop_width::Dict{Symbol, AbstractArray} = Dict{Symbol, AbstractArray}(
                               :knots => fill(0.5, length(nknots)),
                               :β => fill(0.5, 2)),
                      init::Dict{Symbol, AbstractArray} = Dict{Symbol, AbstractArray}(
                               :knots => randn(nknots),
                               :β => randn(2)),
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
                           prop_width::Dict{Symbol, AbstractArray} =
                                Dict{Symbol, AbstractArray}(
                                    :knots => fill(0.5, length(nknots)),
                                    :β => fill(0.5, 2)),
                           init::Dict{Symbol, AbstractArray} =
                                Dict{Symbol, AbstractArray}(
                                    :knots => randn(nknots),
                                    :β => randn(2)),
                           RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

    # Checks
    @assert 0 < misclass < 1 "Misclass rate must be ∈ (0, 1)"
    @assert warmup ≤ iters "Too many warmup iterations"
    @assert reduce(&,
                   [length(init[k]) == length(prop_width[k])
                    for k in keys(init)]) "Need initial values for all parameters"

    # Data setup
    data_locs = Array{Float64, 2}(data[[:Distance, :Elevation]])
    obs = Dict{Symbol, AbstractArray}(
                    :pf => BitArray{1}(data[:PF_code]),
                    :lres => hcat(ones(data[:lres]), data[:lres]))
    data_kwt = knot_wt(knot_locs,
                       kern,
                       data_locs)
    ndata = size(data, 1)

    curr_θ = deepcopy(init)
    curr_proc = Dict{Symbol, AbstractArray}(
                    :spat => Array{Float64, 1}(ndata),
                    :reg => Array{Float64, 1}(ndata))
    curr_pred = BitArray{1}(ndata)
    permafrost_update!(curr_θ, curr_proc, curr_pred, data_kwt, obs)
    curr_lp = permafrost_lp(curr_θ, curr_pred, obs, misclass)

    prop_θ = deepcopy(init)
    prop_proc = Dict{Symbol, AbstractArray}(
                    :spat => Array{Float64, 1}(ndata),
                    :reg => Array{Float64, 1}(ndata))
    prop_pred = BitArray{1}(ndata)
    permafrost_update!(prop_θ, prop_proc, prop_pred, data_kwt, obs)
    prop_lp = permafrost_lp(prop_θ, prop_pred, obs, misclass)

    θ_adj = Dict{Symbol, Vector{Float64}}()
    θ_idx = reduce(vcat,
                   [collect(zip(fill(k, length(curr_θ[k])),
                    1:length(curr_θ[k])))
                    for k in keys(curr_θ)])
    θ_seq = deepcopy(θ_idx)

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
        knot_samp = d_create(run_results, "knots",
                             datatype(Float64),
                             dataspace(nknots, fld(iters, thin)),
                             "chunk", (nknots, 1))
        β_samp = d_create(run_results, "β",
                          datatype(Float64),
                          dataspace(2, fld(iters, thin)),
                          "chunk", (2, 1))
        lp = d_create(run_results, "lp",
                      datatype(Float64),
                      dataspace(fld(iters, thin), 1))


        g_create(run_results, "prop_width")
        pw_log = run_results["prop_width"]
        if finish_adapt > 0
            adapt_log = Dict{Symbol, AbstractArray}(
                            :knots => zeros(length(prop_width[:knots]),
                                            adapt_every),
                            :β => zeros(length(prop_width[:β]),
                                        adapt_every))
            knot_pw_log = d_create(pw_log, "knots",
                                   datatype(Float64),
                                   dataspace(nknots,
                                             finish_adapt ÷ adapt_every + 1),
                                    "chunk", (nknots, 1))
            knot_pw_log[:, 1] = prop_width[:knots]
            β_pw_log = d_create(pw_log, "β",
                                datatype(Float64),
                                dataspace(size(curr_θ[:β], 1),
                                          finish_adapt ÷ adapt_every + 1),
                                "chunk", (size(curr_θ[:β], 1), 1))
            β_pw_log[:, 1] = prop_width[:β]
        else
            pw_log["knots"] = prop_width[:knots]
            pw_log["β"] = prop_width[:β]
        end

        @showprogress "Sampling..." for i in 1:iters
            for k in keys(prop_width)
                θ_adj[k] = prop_width[k] .* randn(RNG,
                                                  size(prop_width[k]))
            end
            θ_seq = shuffle(RNG, θ_idx)

            for (p, i) in θ_seq
                permafrost_update!(p,
                                   i,
                                   θ_adj[p][i],
                                   prop_θ,
                                   prop_proc,
                                   prop_pred,
                                   data_kwt,
                                   obs)
                prop_lp = permafrost_lp(prop_θ,
                                        prop_pred,
                                        obs,
                                        misclass)

                if (prop_lp ≥ curr_lp) || (prop_lp - curr_lp) > log(rand(RNG))
                    curr_lp = prop_lp
                    curr_θ[p][i] = prop_θ[p][i]
                    for k in keys(prop_proc)
                        curr_proc[k][:] = prop_proc[k]
                    end
                    curr_pred[:] = prop_pred
                else
                    prop_θ[p][i] = curr_θ[p][i]
                    for k in keys(curr_proc)
                        prop_proc[k][:] = curr_proc[k]
                    end
                    prop_pred[:] = curr_pred
                end
            end

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                if i % adapt_every != 0
                    for k in keys(curr_θ)
                        adapt_log[k][:, i % adapt_every] = vec(curr_θ[k])
                    end
                else
                    for k in keys(curr_θ)
                        adapt_log[k][:, adapt_every] = vec(curr_θ[k])
                        adapt_prop_width!(prop_width[k], adapt_log[k])
                    end
                    pw_idx = i ÷ finish_adapt + 1
                    knot_pw_log[:, pw_idx] = prop_width[:knots]
                    β_pw_log[:, pw_idx] = prop_width[:β]
                end
            end

            # Record thinned samples when appropriate
            if i % thin == 0
                idx = i ÷ thin
                lp[idx, 1] = curr_lp
                knot_samp[:, idx] = curr_θ[:knots]
                β_samp[:, idx] = curr_θ[:β]
            end
        end
    finally
        close(results)
    end
    nothing
end
