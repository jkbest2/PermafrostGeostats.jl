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
                             soil_type::Array{Int, 1},
                             log_res::AbstractArray,
                             summary::Function = mean)
    β_res = h5read(sample_file, string(run_name, "/β_res"))
    β = permutedims(β_res, [1, 3, 2])

    lres = hcat(ones(log_res), log_res)
    nlocs = length(log_res)

    pred = Array{Float64, 1}(nlocs)
    @showprogress for l in 1:nlocs
        pred[l] = summary(lres[l, :] * β[:, :, soil_type[l]])[1]
    end
    pred
end

function interp_lwc!(pred::Array{Float64, 1},
                     β_res::Array{Float64, 3},
                     soil_type::Array{Int, 1},
                     log_res::Array{Float64, 2};
                     summary::Function = mean)

    nothing
end
