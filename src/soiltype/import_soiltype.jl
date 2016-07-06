using JLD
using DataFrames

dat = load("results/soil_type.jld")

soil_type = dat["soil_type"]
cores = dat["cores"]
int_elev = dat["int_elev"]
int_trans = dat["int_trans"]
int_dpth = dat["int_dpth"]


