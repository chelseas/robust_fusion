# infrastructure

using NeuralVerification
import NeuralVerification: Network, Layer, ReLU, Id, affine_map, forward_partition, Model, init_vars, get_bounds, add_set_constraint!, add_complementary_set_constraint!, max_disturbance!, BoundedMixedIntegerLP, encode_network!, OPTIMAL, value, symbolic_infty_norm

using LazySets
import LazySets: HalfSpace

using BSON
import BSON: @load @save

using JuMP
import JuMP: optimize!, termination_status

# function to setup problem with network
# returns model and input var array
function setup_problem(network, input_for_bounds)
    solver = MIPVerify()
    model = Model(solver)
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    # Future upgrade: call ai2 get_bounds to get tighter / faster bounds that the interval arithmetic bounds that are standard with MIPVerify in this implementation
    # Future upgrade: call planet if want tighter and slower (pass ai2 bounds as starting bounds)
    model[:bounds] = get_bounds(network, input_for_bounds, before_act=true)
    model[:before_act] = true
    encode_network!(model, network, BoundedMixedIntegerLP())

    # this is the set that the bounds are valid for: 
    add_set_constraint!(model, input_for_bounds, first(z))
    
    return model, z
end

# get robustness on inputs for 1 output constraint
function get_network_robustness_1_constraint(model, z, output_constraint, input_center)
    # z are the VariableReferences to the jump real valued variables for each network layer
    # solve a robustness query
    add_complementary_set_constraint!(model, output_constraint, last(z))
    # set objective
    o = max_disturbance!(model, first(z) - input_center)
    optimize!(model)

    # possible outcomes:
    # 1) infeasible. This means that the distance to the closest adversarial example is farther than the set input_for_bounds allows. Recourse: make the input_for_bounds set larger, try again
    if termination_status(model) != OPTIMAL 
        println("Try again with larger input set")
        return nothing
    else # if optimal! 
    # 2) optimal. This means that the distance to the closest adversarial example has been found! See what it is. 
        println("The distance to the closest adversarial example is: ", value(o))
        return value(o)
    end
end

# get robustness on OG NETWORK inputs for multiple output constraints
function get_initial_robustness_on_inputs(network, function_for_input_set, output_constraints, input_center)
    # input center is just the training example itself
    robustness = Inf # distance to closest adversarial example
    lr, hr = 1., 1. # used to define input set that is used to calculate bounds for network
    for oc in output_constraints
        found_robustness = false
        iters = 0
        while ~found_robustness && iters < 10
            # for each output constraint, get the robustness, then take the min
            input_for_bounds = function_for_input_set(input_center, lr, hr)
            @debug("input for bounds is: Hyperrectangle(low=", low(input_for_bounds), " , high=", high(input_for_bounds), ")")
            model, z = setup_problem(network, input_for_bounds)
            robustness_i = get_network_robustness_1_constraint(model, z, oc, input_center)
            iters += 1
            if !isnothing(robustness_i)
                robustness = min(robustness, robustness_i)
                found_robustness = true
            else
                # need to adjust input bounds used for bounding network nodes
                @debug("Enlarging input set to find robustness...")
                if iters % 2 == 0
                    lr -= 1 # extend in the downward direction
                elseif iters % 2 == 1
                    hr += 1 # extend in the upward direction
                end
            end
        end
        if (iters ≥ 10) &&  ~found_robustness
            error("Something is wrong with this example. Could not find large enough input set. Maybe there are no adversarial examples?")
        end
    end
    # found robustness!
    return robustness
end

function get_second_sensor_robustness_1_constraint(model, z, old_input_center, old_input_robustness, output_constraint, new_input_center)
    # add new input and fused output
    new_input = @variable(model)
    set_name(new_input, "new_input")
    # fused output
    fused_output = @variable(model)
    set_name(fused_output, "fused_output")
    @constraint(model, fused_output == 0.5*last(z)[1] + 0.5*new_input)

    ###### OLD INPUTS
    # and now add constraint corresponding to robust set that was calculated
    # first construct expression representing infinity norm over old inputs
    o = symbolic_infty_norm(first(z) - old_input_center)
    @constraint(model, o <= old_input_robustness) # last set value from last robustness query). 
    # basically we are saying that the infinity norm of the old inputs has to be less than or 
    # equal to this, which imposes constraints on the old inputs

    ###### OUPUTS
    # y not in Y (pass Y tho)
    # next add the output constraint to the FUSED SIGNAL not the old outputs (last(z))
    add_complementary_set_constraint!(model, output_constraint, [fused_output])

    ###### OBJECTIVE
    # set objective. Now we are setting the objective on the new_input
    o_new = max_disturbance!(model, [new_input - new_input_center])

    optimize!(model)

    # possible outcomes:
    # 1) infeasible. No perturbation on the second sensor could cause the fused signal to leave the set Y. Are things fused correctly? Because this seems unlikely. 
    if termination_status(model) != OPTIMAL 
        error("The second signal cannot break the fused signal....something is probably wrong.")
    else # if optimal! 
    # 2) optimal. This means that the distance to the closest adversarial example has been found! See what it is. 
        println("The distance to the closest adversarial example is: ", value(o_new))
        if value(o_new) < 1e-6
            @warn("The value of the objective is essentially 0, meaning that fusion strictly degrades the signal :/ Get a better second sensor?")
        else 
            println("If the signal is more precise than ", value(o_new), " it will improve the robustness of the network")
        end
        return value(o_new)
    end

end

function get_robustness_on_second_sensor(network, old_input_bounds_set, old_input_center, old_input_robustness, output_constraints, new_input_center)
    # input center is just the training example itself

    robustness = Inf # distance to closest adversarial example
    for oc in output_constraints
        # for each output constraint, get the robustness, then take the min
        model, z = setup_problem(network, old_input_bounds_set)
        robustness_i = get_second_sensor_robustness_1_constraint(model, z, old_input_center, old_input_robustness, oc, new_input_center)
        robustness = min(robustness, robustness_i)
    end
    # found robustness!
    return robustness
end


# TODO next:

# switch to scripting file and write code that uses/tests these functions above ^

# Write part 3: how does initial robustness change when tighter than needed
# second sensor is used?

# THENNN go to scripting file and write code that uses this code / run experiments!