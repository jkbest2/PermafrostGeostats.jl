"""
    watercontent_interp(sample_file::AbstractString,
                        run_name::AbstractString,
                        kern::AbstractConvolutionKernel,
                        new_locs::Array{Float64, 2},
                        summary::Function = mean)

Reads the samples from `sample_file` and uses them to calculate the water
content at each new location. Returns an Array{Float64} of the water content
on the logistic scale corresponding to the locations in `new_locs`.
"""
function watercontent_interp(sample_file::AbstractString,
                             run_name::AbstractString,
                             kern::AbstractConvolutionKernel,
                             new_locs::Array{Float64, 2},
                             summary::Function = mean)
    wc_knots = h5read(sample_file, string(run_name, "/knots"))
    nlocs = size(new_locs, 1)
    pred = Vector{Float64}(nlocs)
    @showprogress for loc in 1:nlocs
        pred[loc] = summary(kwt[loc, :] * pf_knots)
    end
    pred
end

