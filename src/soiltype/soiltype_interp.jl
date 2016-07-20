"""
    soiltype_interp(sample_file::AbstractString,
                    run_name::AbstractString,
                    knot_locs::Array{Float64, 2},
                    kern::AbstractConvolutionKernel,
                    new_locs::Array{Float64, 2})

Reads the samples from `sample_file` and uses them to calculate the proportion
of samples where each new location is a given soil type. Returns an
Array{Float64} corresponding to the locations in `new_locs`.
"""
function soiltype_interp(sample_file::AbstractString,
                         run_name::AbstractString,
                         kwt::AbstractArray,
                         new_locs::Array{Float64, 2})
    soil_knots = h5read(sample_file, string(run_name, "/knots"))
    # kwt = knot_wt(knot_locs, kern, new_locs)

    nlocs = size(new_locs, 1)
    nproc = size(soil_knots, 2)
    nsamp = size(soil_knots, 3)

    p =    zeros(nlocs, nproc)
    pred = zeros(nlocs, nproc)

    @showprogress for i in 1:nsamp
        for j in 1:nproc
            p[:, j] = kwt * soil_knots[:, j, i]
        end
        for j in 1:size(pred, 1)
            pred[j, indmax(p[j, :])] += 1.
        end
    end
    pred ./= nsamp
end
