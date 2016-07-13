abstract AbstractBoreholeModel <: Any
abstract AbstractBHModelData <: Any

type PermafrostData <: AbstractBHModelData 
    data_locs::Array{Float64, 2}
    pf_status::BitArray{1}
    kwt::Array{Float64, 2}

    function PermafrostData(model::PermafrostModel,
                            data::BoreholeTransect)
        notna = !isna(bh_dat.pf_status)
        kwt = knot_wt(model.knot_locs,
                      model.kern,
                      data.data_locs)

        new(Array{Float64, 2}(bh_dat.locs[notna, :]),
            BitArray{1}(bh_dat.pf_status[notna, :]),
            kwt)
    end
end

function lp!{T <: PermafrostModel}(state::MCMCState{T},
                                   model::T,
                                   data::ModelData{T})
   corr_lik = log(1. - model.misclass)
   incorr_lik = log(model.misclass)
   n_pfdata = length(data.pf_status)
   n_corr = sum(data.pf_status .== state.aux[:pf_pred])

   state.lp = loglikelihood(model.θ_prior[:pf_knots], state.θ[:pf_knots])
   state.lp += n_corr * corr_lik
   stat.lp += (n_pfdata - n_corr) * incorr_lik
end

function __lp{T <: PermafrostModel}(θ::Array{Float64, 1},
                                    aux::BitArray{1},
                                    model::T,
                                    data::ModelData{T})
   corr_lik = log(1. - model.misclass)
   incorr_lik = log(model.misclass)
   n_pfdata = length(data.pf_status)
   n_corr = sum(data.pf_status .== aux)

   logpost = loglikelihood(model.θ_prior[:pf_knots], θ)
   logpost += n_corr * corr_lik
   logpost += (n_pfdata - n_corr) * incorr_lik

    logpost
end

function update!{T <: PermafrostModel}(par::Symbol,
                                       idx::Integer,
                                       δ::Float64,
                                       state::MCMCState{T},
                                       model::T,
                                       data::PermafrostData)
    state.θ[:pf_knots][idx] += δ
    state.aux[:pf_status][:] = (data.kwt * state.θ) .> 0

    lp!(state,
        model,
        data)
end

type MCMCState{T} <: Any
    θ::Dict{Symbol, Array{Float64}}
    aux::Dict{Symbol, Any}
    lp::Float64
end

function MCMCState{T <: PermafrostModel}(model::T,
                                         data::PermafrostData,
                                         init::InitialValues)

    __θ = Dict{Symbol, Array{Float64}}(:pf_knots => init.θ[:pf_knots]),
    __aux = Dict{Symbol, BitArray{1}}(:pf_status => (data.kwt * θ) . >  0),
    __lp = BoreholePaperPackage.__lp(__θ[:pf_knots],
                                     __aux[:pf_status],
                                     model,
                                     data)
    MCMCState(__θ, __aux, __lp)
end

type PermafrostModel <: AbstractBoreholeModel
    θ_name::Array{Symbol}
    θ_dim::Dict{Symbol, Any}
    θ_index::Array{Tuple{Symbol, Int}, 1}
    θ_prior::Dict{Symbol, Distribution}
    knot_locs::Array{Float64, 2}
    misclass::Float64

    function PermafrostModel(knot_locs::Array{Float64, 2},
                             data::PermafrostData,
                             kern::AbstractConvolutionKernel;
                             prior::Distribution = Normal(0, 1),
                             misclass = 1e-3)
        kwt = knot_wt(knot_locs,
                      kern,
                      data.data_locs)
        nknots = size(knot_locs, 1)

        θ_ind = collect(zip(fill(:pf_knots, nknots), 1:nknots))

        new([:pf_knots],
            Dict(:pf_knots => nknots),
            θ_ind,
            Dict(:pf_knots => prior),
            knot_locs,
            misclass)
end
