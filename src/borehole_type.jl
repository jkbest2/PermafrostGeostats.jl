immutable DoubleDict{T1, T2}
    V1::T1
    V2::T2

    function DoubleDict(V1::T1, V2::T2)
        if length(V1) != length(V2)
            error("V1, V2 must be the same length")
        elseif T1 == T2 
            error("V1 and V2 must have different types")
        end

        new(V1, V2)
    end
end

type BoreholeTransect
    locs::Array{Real, 2}
    pf::DataArray{Bool, 1}
    soil::DataArray{Integer, 1}
    log_res::DataArray{Real, 1}
    moisture::DataArray{Real, 1}

    function BoreholeTransect(data::DataFrame;
                              distance,
                              elevation,
                              permafrost,
                              soil_type,
                              log_resistivity,
                              moisture)
        new(Array{Float64, 2}(hcat(data[distance],
                                   data[elevation])),
            data[permafrost],
            data[soil_type],
            data[log_resistivity],
            data[moisture])
    end
end

                               
                              
                              
    
    
    
