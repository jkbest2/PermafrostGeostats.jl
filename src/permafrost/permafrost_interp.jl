"""
    permafrost_interp(sample_file::AbstractString,
                      run_name::AbstractString,
                      kern::AbstractConvolutionKernel,
                      new_locs::Array{Float64, 2})

Reads the samples from `sample_file` and uses them to calculate the proportion
of samples where each new location *is* permafrost. Returns an Array{Float64}
corresponding to the locations in `new_locs`.
"""
function permafrost_interp(sample_file::AbstractString,
                           run_name::AbstractString,
                           kwt::AbstractArray,
                           new_locs::Array{Float64, 2},
                           lres::Array{Float64, 2};
                           knot_name = "knots",
                           β_name = "β")
    pf_knots = h5read(sample_file, string(run_name, "/", knot_name))
    pf_β = h5read(sample_file, string(run_name, "/", β_name))

    nlocs = size(new_locs, 1)
    nsamp = size(pf_knots, 2)

    spat_proc = Vector{Float64}(nsamp)
    reg_roc = Vector{Float64}(nsamp)

    pred = Vector{Float64}(nlocs)
    @showprogress for loc in 1:nlocs
        spat_proc = kwt[loc, :] * pf_knots
        reg_proc = lres[loc, :] * pf_β
        pred[loc] = mean(spat_proc .+ reg_proc .> 0)
    end
    pred
end

