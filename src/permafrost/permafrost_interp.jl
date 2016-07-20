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
                           new_locs::Array{Float64, 2};
                           knot_name = "knots")
    pf_knots = h5read(sample_file, string(run_name, "/", knot_name))
    nlocs = size(new_locs, 1)
    pred = Vector{Float64}(nlocs)
    @showprogress for loc in 1:nlocs
        pred[loc] = mean(kwt[loc, :] * pf_knots .> 0)
    end
    pred
end

