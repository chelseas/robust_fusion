using LazySets
import LazySets: HalfSpace, Hyperrectangle

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


