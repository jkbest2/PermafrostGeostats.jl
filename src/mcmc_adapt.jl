"""
    accept_rate(samples::AbstractArray)

Calculates the acceptance rate of proposed parameter values. Assumes
parameters are in rows and each column is one MCMC iteration.
"""
function accept_rate(samples::AbstractArray)
    I, J = size(samples)
    accept = Array{Float64}(zeros(I))
    for i in 1:I
        for j in 2:J
            if samples[i, j] != samples[i, j - 1]
                accept[i] += 1
            end
        end
    end
    accept ./ (J - 1)
end

"""
    erf_rate_score(x::AbstractArray, k::Real = 3.)

A sigmoid function used to adapt the proposal width toward a desired
acceptance rate. Will adjust proposal width by a positive factor less
than 2. Increased `k` results in more dramatic changes. Originally
from Lora.jl's proposal width tuner.
"""
function erf_rate_score(x::AbstractArray, k::Real = 3.)
    erf(k .* x) .+ 1
end

"""
    adapt_prop_width!(pw::Array{Float64, 1},
                      samples::Array{Float64, 2},
                      target_rate::Float64 = 0.44)

Adapts the MCMC proposal width *in place* based on a set of samples
in an effort to get the acceptance rate to the `target_rate`.
"""
function adapt_prop_width!(pw::Array{Float64, 1},
                           samples::Array{Float64, 2},
                           target_rate::Float64 = 0.44)
    acc = accept_rate(samples)
    pw[:] = pw .* erf_rate_score(acc .- target_rate)
    nothing
end
