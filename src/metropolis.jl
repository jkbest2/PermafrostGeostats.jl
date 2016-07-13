type MCMCRun
    iterations::Integer
    thin::Integer
    warmup::Integer
    finish_adapt::Integer
    adapt_every::Integer
    results_file::AbstractString
    run_name::AbstractString
end

type InitialValues
    θ::Dict{Symbol, Array{Float64}}
    prop_width::Dict{Symbol, Array{Float64}}
end

function Metropolis(model::AbstractBoreholeModel,
                    data::AbstractBHModelData,
                    run::MCMCRun,
                    init::InitialValues)

    # Initialize arrays (and dictionaries of arrays)
    prop_dist = Dict{Symbol, Distribution}()
    for k in keys(init.prop_width)
        prop_dist[k] = MvNormal(zeros(prop_width[k]),
                                PDiagMat(prop_width[k]))
    end

    θ_curr = init
    θ_prop = deepcopy(init)

    proc_curr = Dict{Symbol, AbstractArray}(
                    :lwc_reg => Array{Float64, 1}(ndata))
    proc_prop = Dict{Symbol, AbstractArray}(
                    :lwc_reg => Array{Float64, 1}(ndata))

    pred_curr = Dict{Symbol, AbstractArray}(
                    :lwc => Array{Float64, 1}(ndata))
    pred_prop = Dict{Symbol, AbstractArray}(
                    :lwc => Array{Float64, 1}(ndata))

    update!(θ_curr,
            proc_curr,
            pred_curr,
            obs)
    update!(θ_prop,
            proc_prop,
            pred_prop,
            obs)

    lp_curr = wc_lp(θ_curr, pred_curr, obs)
    lp_prop = -Inf

    θ_adj = Dict{Symbol, Array{Float64, 1}}()
    for k in keys(prop_dist)
        θ_adj[k] = rand(prop_dist[k])
    end

    θ_ind = reduce(vcat,
                   [collect(zip(fill(k, length(θ_curr[k])), 1:length(θ_curr[k])))
                    for k in keys(θ_curr)])
    θ_seq = Array{Tuple{Symbol, Int}, 1}(θ_ind)

    if finish_adapt > 0
        adapt_log = Dict{Symbol, AbstractArray}()
        for k in keys(θ_curr)
            adapt_log[k] = Array{Float64}(length(θ_curr[k]), adapt_every)
        end
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
            for k in keys(prop_dist)
                θ_adj[k] = rand(prop_dist[k])
            end
            θ_seq[:] = sample(θ_ind, length(θ_ind), replace = false)

            for (par, idx) in θ_seq
                update!(par, idx, θ_adj[par][idx],
                        θ_prop,
                        proc_prop,
                        pred_prop,
                        obs)
                lp_prop = wc_lp(θ_prop,
                                pred_prop,
                                obs)

                if (lp_prop ≥ lp_curr) || log(rand(1))[1] < (lp_prop - lp_curr)
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
