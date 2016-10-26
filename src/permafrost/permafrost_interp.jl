"""
    permafrost_interp(sample_file::AbstractString,
                      run_name::AbstractString,
                      kwt::AbstractArray,
                      new_locs::Array{Float64, 2},
                      log_res::Vector{Float64};
                      [knot_name = "knots",
                      β_name = "β"])

Reads the samples from `sample_file` and uses them to calculate the proportion
of samples where each new location *is* permafrost. Returns an Array{Float64}
corresponding to the locations in `new_locs`.
"""
function permafrost_interp(sample_file::AbstractString,
                           run_name::AbstractString,
                           kwt::AbstractArray,
                           new_locs::Array{Float64, 2},
                           log_res::Vector{Float64};
                           knot_name = "knots",
                           β_name = "β")
    lres = hcat(ones(log_res), log_res)

    post_warmup = h5read(sample_file,
                         string(run_name, "/meta/n_warmup_samp")) + 1
    nsamp_tot = h5read(sample_file,
                       string(run_name, "/meta/n_samp"))

    pf_knots = h5read(sample_file,
                      string(run_name, "/", knot_name),
                      (:, post_warmup:nsamp_tot))
    pf_β = h5read(sample_file,
                  string(run_name, "/", β_name),
                  (:, post_warmup:nsamp_tot))

    nlocs = size(new_locs, 1)
    nsamp = size(pf_knots, 2)

    spat_proc = Vector{Float64}(nsamp)
    reg_roc = Vector{Float64}(nsamp)

    pred = Vector{Float64}(nlocs)
    @showprogress for loc in 1:nlocs
        spat_proc = kwt[loc:loc, :] * pf_knots
        reg_proc = lres[loc:loc, :] * pf_β
        pred[loc] = mean(spat_proc .+ reg_proc .> 0)
    end
    pred
end

