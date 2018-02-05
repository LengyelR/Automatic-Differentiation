import math


ops = {"add": lambda x, y: x + y,
       "sub": lambda x, y: x-y,
       "sin": lambda x: math.sin(x),
       "cos": lambda x: math.cos(x),
       "log": lambda x: math.log(x),
       "square": lambda x: math.pow(x, 2)}

grad_ops = {"add": [(lambda x, y: 1), (lambda x, y: 1)],
            "sub": [(lambda x, y: 1), (lambda x, y: -1)],
            "sin": [(lambda x: math.cos(x))],
            "cos": [lambda x: -math.sin(x)],
            "log": [lambda x: 1 / x],
            "square": [(lambda x: 2*x)]}


def gradient(expression_list, values):
    # copy input variables
    input_vars = list(values.keys())

    # evaluation: traverse list
    for v, op_name, var_names in expression_list:
        op = ops[op_name]
        values[v] = op(*(values[var_name] for var_name in var_names))

    # propagate backwards the gradients
    delta = {'z': 1}
    for v, op_name, var_names in reversed(expression_list):
        for variable_idx in range(len(var_names)):
            op = grad_ops[op_name][variable_idx]
            var_name = var_names[variable_idx]
            if var_name not in delta.keys():
                delta[var_name] = 0
            params = (values[var_name] for var_name in var_names)
            delta[var_name] += delta[v] * op(*params)

    return {input_var: delta[input_var] for input_var in input_vars}


if __name__ == "__main__":
    graph = [("v1", "add", ["x0", "x1"]),
             ("v2", "square", ["v1"]),
             ("v3", "sin", ["x1"]),
             ("z", "sub", ["v2", "v3"])]
    init_values = {"x0": 5.0, "x1": 3.0}

    gradients = gradient(graph, init_values)

    epsilon = 1e-10
    assert abs(gradients['x0'] - 16.0) < epsilon
    assert abs(gradients['x1'] - 16.9899924966) < epsilon
