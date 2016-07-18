using ProgressMeter
using Distributions
using HDF5

"""
    function watercontent_lp(θ::Dict{Symbol, AbstractArray},
                             pred::Dict{Symbol, AbstractArray},
                             data::Dict{Symbol, AbstractArray},
                             prior::Dict{Symbol, Distribution} =
                              Dict{Symbol, Distribution}(
                                      :β_res => Normal(0., 20.),
                                      :σ => Chisq(1.)))

Log posterior of the water content model, using a linear relationship between
log resistivity and water content that is specific to each soil type.
"""
function watercontent_lp(θ::Dict{Symbol, AbstractArray},
                         pred::Dict{Symbol, AbstractArray},
                         data::Dict{Symbol, AbstractArray},
                         prior::Dict{Symbol, Distribution} =
                          Dict{Symbol, Distribution}(
                                  :β_res => Normal(0., 20.),
                                  :σ => Chisq(1.)))
    # Prior
    lp = sum([loglikelihood(prior[k], θ[k]) for k in keys(θ)])

    # logistic water content likelihood
    lp += loglikelihood(Normal(0., θ[:σ][1]), pred[:lwc] .- data[:lwc])

    lp
end

"""
    watercontent_update!(change_param::Symbol,
                         change_ind::Int,
                         change_adj::Float64,
                         θ::Dict{Symbol, AbstractArray},
                         pred::Dict{Symbol, AbstractArray},
                         data::Dict{Symbol, AbstractArray})

Update watercontent parameters and predictions in place.
"""
function watercontent_update!(change_param::Symbol,
                              change_ind::Int,
                              change_adj::Float64,
                              θ::Dict{Symbol, AbstractArray},
                              pred::Dict{Symbol, AbstractArray},
                              data::Array{Float64, 2})
    if change_param == :σ
        if (θ[:σ] + change_adj)[1] > 0
            θ[:σ] += change_adj
        end
        return nothing
    end

    if change_param == :β_res
        θ[:β_res][change_ind] += change_adj
        for i in 1:length(pred[:lwc])
            pred[:lwc][i] = (data[i, :] *
                                 θ[:β_res][:, data[:soil][i]])[1]
        end
    end
    return nothing
end

"""
    watercontent_update!(θ::Dict{Symbol, AbstractArray},
                         pred::Dict{Symbol, AbstractArray},
                         data::Dict{Symbol, AbstractArray})

Don't update parameter values, but calculate the predicted values. Useful for
finding initial prediction.
"""
function watercontent_update!(θ::Dict{Symbol, AbstractArray},
                              pred::Dict{Symbol, AbstractArray},
                              data::Dict{Symbol, AbstractArray})
    for i in 1:length(pred[:lwc_reg])
        proc[:lwc_reg][i] = (data[:lres][i, :] *
                                 θ[:β_res][:, data[:soil][i]])[1]
    end

    pred[:lwc][:] = proc[:lwc_reg]
    nothing
end

"""
    watercontent_metrop(knot_locs::Array{Float64, 2},
                        kern::AbstractConvolutionKernel,
                        data;
                        results_file::AbstractString = "results/temp.jdf5",
                        run_name::AbstractString = "run",
                        iters::Integer = 2000,
                        thin::Integer = 1,
                        warmup::Integer = 1000,
                        finish_adapt::Integer = 800,
                        adapt_every::Integer = 100,
                        prop_width::Dict{Symbol, Array{Float64, 1}} =
                           Dict{Symbol, Array{Float64, 1}}(
                               :β_res => fill(0.5, 2 * 6),
                               :σ => [0.1]),
                        init::Dict{Symbol, AbstractArray} =
                           Dict{Symbol, Array{Float64}}(
                               :β_res => randn(2, 6),
                               :σ => randexp(1)),
                        RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

Function for performing Markov Chain Monte Carlo using the Metropolis algorithm
to fit the water content model. Samples are saved in HDF5 format in
`results_file.hdf5`. Specialized for the water content model.
"""
function watercontent_metrop(knot_locs::Array{Float64, 2},
                             kern::AbstractConvolutionKernel,
                             data;
                             results_file::AbstractString = "results/temp.jdf5",
                             run_name::AbstractString = "run",
                             iters::Integer = 2000,
                             thin::Integer = 1,
                             warmup::Integer = 1000,
                             finish_adapt::Integer = 800,
                             adapt_every::Integer = 100,
                             prop_width::Dict{Symbol, Array{Float64, 1}} =
                                Dict{Symbol, Array{Float64, 1}}(
                                    :β_res => fill(0.5, 2 * 6),
                                    :σ => [0.1]),
                             init::Dict{Symbol, AbstractArray} =
                                Dict{Symbol, Array{Float64}}(
                                    :β_res => randn(2, 6),
                                    :σ => randexp(1)),
                             RNG::AbstractRNG = MersenneTwister(rand(UInt64)))

    # Checks
    @assert init[:σ] > 0 "Initial σ must be positive"
    @assert warmup ≤ iters "Number of iterations must be larger than warmup"
    @assert all((k) -> length(init[k]) == length(prop_width[k]), keys(init)) "Initial values and proposal widths must have same dimensions."

    # Data setup
    nknots = size(knot_locs, 1)
    data_locs = Array{Float64, 2}(data[[:Distance, :Elevation]])
    obs = Dict{Symbol, AbstractArray}()
    obs[:lwc] = Array{Float64, 2}(hcat(ones(data[:LWC]), data[:LWC]))
    obs[:lres] = Array{Float64, 1}(data[:lres])
    obs[:soil] = Array{Integer, 1}(data[:Soil])
    ndata = size(data_locs, 1)
    nsoilprocs = size(init[:β_res], 2)

    θ_curr = init
    θ_prop = deepcopy(init)

    θ_len = Dict{Symbol, Integer}()
    for k in keys(θ_curr)
        θ_len[k] = length(θ_curr[k])
    end

    pred_curr = Dict{Symbol, AbstractArray}(
                    :lwc => Array{Float64, 1}(ndata))
    pred_prop = Dict{Symbol, AbstractArray}(
                    :lwc => Array{Float64, 1}(ndata))

    update!(θ_curr,
            pred_curr,
            obs)
    update!(θ_prop,
            pred_prop,
            obs)

    lp_curr = wc_lp(θ_curr, pred_curr, obs)
    lp_prop = -Inf

    θ_adj = Dict{Symbol, Array{Float64, 1}}()
    for k in keys(prop_width)
        θ_adj[k] = randn(RNG, θ_len[k]) .* prop_width[k]
    end

    θ_ind = reduce(vcat,
                   [collect(zip(fill(k, length(θ_curr[k])), 1:length(θ_curr[k])))
                    for k in keys(θ_curr)])
    θ_seq = Array{Tuple{Symbol, Int}, 1}(θ_ind)

    if finish_adapt > 0
    end

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
        nsave = fld(iters, thin)
        β_res_samp = d_create(run_results, "β_res",
                              datatype(Float64),
                              dataspace(2, nsoilprocs, nsave),
                              "chunk", (2, nsoilprocs, 1))
        σ_samp = d_create(run_results, "σ",
                          datatype(Float64),
                          dataspace(1, nsave))
        lp_samp = d_create(run_results, "lp",
                           datatype(Float64),
                           dataspace(1, nsave))

        if finish_adapt > 0
            adapt_log = Dict{Symbol, AbstractArray}()
            for k in keys(θ_curr)
                adapt_log[k] = Array{Float64}(length(θ_curr[k]), adapt_every)
            end
            g_create(run_results, "prop_width")
            pw_log = run_results["prop_width"]
            β_res_pw = d_create(pw_log, "β_res",
                                datatype(Float64),
                                dataspace(length(θ_curr[:β_res]),
                                          finish_adapt ÷ adapt_every + 1))
            β_res_pw[:, 1] = prop_width[:β_res]
            σ_pw = d_create(pw_log, "σ",
                            datatype(Float64),
                            dataspace(1,
                                      finish_adapt ÷ adapt_every + 1))
            σ_pw[:, 1] = prop_width[:σ]
        end

        @showprogress "Sampling..." for i in 1:iters

            for k in keys(prop_width)
                θ_adj[k] = randn(RNG, θ_len[k]) .* prop_width[k]
            end
            θ_seq[:] = shuffle(RNG, θ_ind)

            for (par, idx) in θ_seq
                watercontent_update!(par, idx, θ_adj[par][idx],
                                     θ_prop,
                                     pred_prop,
                                     obs)
                lp_prop = watercontent_lp(θ_prop,
                                          pred_prop,
                                          obs)

                if (lp_prop ≥ lp_curr) || log(rand(RNG)) < (lp_prop - lp_curr)
                    lp_curr = lp_prop
                    θ_curr[par][idx] = θ_prop[par][idx]
                    for k in keys(proc_curr)
                        proc_curr[k] = copy(proc_prop[k])
                    end
                    for k in keys(pred_curr)
                        pred_curr[k] = copy(pred_prop[k])
                    end
                else
                    θ_prop[par][idx] = θ_curr[par][idx]
                    for k in keys(proc_curr)
                        proc_prop[k] = copy(proc_curr[k])
                    end
                    for k in keys(pred_curr)
                        pred_prop[k] = copy(pred_curr[k])
                    end
                end
            end

            # println("iteration: ", i, "/", iters, " lp: ", lp_curr)

            # Adapt proposal distribution during warmup
            if i ≤ finish_adapt
                if i % adapt_every != 0
                    for k in keys(θ_curr)
                        adapt_log[k][:, i % adapt_every] = vec(θ_curr[k])
                    end
                else
                    for k in keys(θ_curr)
                        adapt_log[k][:, adapt_every] = vec(θ_curr[k])
                        adapt_prop_width!(prop_width[k], adapt_log[k])
                        prop_dist[k] = MvNormal(zeros(prop_width[k]),
                                                PDiagMat(prop_width[k]))
                    end
                    pw_idx = i ÷ finish_adapt + 1
                    β_res_pw[:, pw_idx] = prop_width[:β_res]
                    σ_pw[:, pw_idx] = prop_width[:σ]
                end
            end

            # Record thinned samples when appropriate
            if i % thin == 0
                idx = i ÷ thin
                β_res_samp[:, :, idx] = θ_curr[:β_res]
                σ_samp[1, idx] = θ_curr[:σ]
                lp_samp[1, idx] = lp_curr
            end
        end
    finally
        close(results)
    end
    nothing
end

