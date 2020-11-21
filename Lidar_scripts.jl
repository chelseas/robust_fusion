using LazySets
import LazySets: HalfSpace, Hyperrectangle

using BSON
import BSON: @save, @load

using Flux

using NeuralVerification

using Plots

include("infrastructure.jl")

# for robustness in the LiDAR example, the desired output property is:
# +/- 1m from true distance to car ahead
buffer = 1.

# function to generate output properties for a given training example
function get_Y_constraints(y)
    output1 = HalfSpace([1.], y + buffer) # output <= label + 1
    output2 = HalfSpace([-1.], buffer - y) # output >= label - 1 aka -1*output <= 1 - label 
    # e.g. if y = 9, the constraints are: output <= 10 and -output <= -8 aka output >= 8
    @debug("Output must be within: [", y - buffer, ",", y + buffer, "]")
    return (output1, output2)
end

# the center around which the robustness radius is calculated is just the training point itself!

# function to generate input sets for bounds for network
get_input_set_for_bounds(x, lr, hr) = Hyperrectangle(low = (x .- lr), high = (x .+ hr))

# script away!!
# load training data and network
@load "lidar_model_1.bson" model
network = NeuralVerification.network(model)
flux_net = model

training_data = @load "training_data.bson" X Y

# Turned out to be unecessary...
# # for all training data point, compute required precision
# all_robustness = []
# all_precision = []
# for i = 1:length(Y)        
#     x = X[:,i]
#     y = Y[:,i][1]
#     output_constraints = get_Y_constraints(y) 
#     r, old_input_bounds_set = get_robustness_on_network_inputs(network, get_input_set_for_bounds, output_constraints, x)
#     if ~isnothing(r) # some adversarial examples are very far away...
#         p =  get_robustness_on_second_sensor(network, old_input_bounds_set, x, r, output_constraints, y)
#         push!(all_robustness, r)
#         push!(all_precision, p)
#     end
# end
# mean_robustness = sum(all_robustness)/length(all_robustness)
# mean_precision = sum(all_precision)/length(all_precision)
# println("mean_robustness=", mean_robustness)
# println("mean_precision=", mean_precision)
# # could also plot needed precision as a function of robustness?
# # Plots.scatter(all_robustness, all_precision, )
# # xlabel!("Robustness") # ylabel!("Required GPS Precision")
# # title!("Precision Needed vs. Robustness")
# # savefig("Precision_vs_robustness_over_training_data.png") # <- sadly low-res...
# # If you look at plot you can see that the required precision is ~1 which makes sense because
# # our output property is to keep the output in +/- 1
# # BUT for training data points that are not robust at all, a higher precision (smaller p) 
# # is needed APPARENTLY....but I think this is numerical error.

# @save "precision_robustness_over_training_data.bson" all_robustness all_precision

### Experiment 3: charting how robustness change as a function of precision on input 
# NOTE: only check examples that are correctly classified!!! ??
# for each training data point, compute how robustness changes 
all_robustness_curves = []
all_p_values = []
indices = rand(1:10000, 200) #check to make sure unique # usually at least 196 of 200 are unique tho
for counter = 1:200
    i = indices[counter]
    x = X[:,i]
    y = Y[:,i][1]
    output_constraints = get_Y_constraints(y) 
    
    # check if this training example is correctly classified
    correct =  all([Yset.a[1]*flux_net(x)[1] ≤ Yset.b for Yset in output_constraints])
    
    if correct
        #p = 2 # to begin with
        robustness = []
        p_array = [.05, .25, .5, .75, 1, 1.25]
        for i = 1:6
            p = p_array[i]
            r, old_input_bounds_set = get_robustness_on_network_inputs(network, get_input_set_for_bounds, output_constraints, x, fusion=true, new_input_center=y, p=p)
            #push!(p_array, p)
            push!(robustness, r)
            #p /= 2
        end
        push!(all_robustness_curves, robustness)
        push!(all_p_values, p_array)
    end
    # broke this loop when about 6000 examples had been processed because I can't 
    # plot 6,000 curves anyway :'D 
end

Plots.plot(all_p_values[1], all_robustness_curves[1], xlabel="Precision (meters)", ylabel="Robustness (meters)", title="Robustness as Precision is Changed", legend=false, marker=".", ylims=(0,2))
for i = 1:50
    idx = i+1
    if  !any(isnothing.(all_robustness_curves[idx]))
        Plots.plot!(all_p_values[idx], all_robustness_curves[idx], marker=".")
    end
end
Plots.plot!(all_p_values[104], all_robustness_curves[104], marker=".")
