import numpy as np

nn_arch = []

def add_layer(output_dim, activation = "relu", input_dim = -1):
    if (len(nn_arch) == 0):
        if (input_dim == -1):
            print ("You must give input dimension for first layer")
            return
        nn_arch.append({"input" : input_dim, "output" : output_dim, "activation" : activation})
        return
    last_output_dim = nn_arch[-1]["output"]
    if (input_dim == -1):
        input_dim = last_output_dim
    elif (input_dim != last_output_dim):
        print ("Input dimension is not compatible with last output")
        return
    nn_arch.append({"input" : input_dim, "output" : output_dim, "activation" : activation})
 
def init_layers():
    np.random.seed(99)
    params = {}
    
    for idx, layer in enumerate(nn_arch):
	# we number network layers from 1
        layer_idx = idx + 1
        input_size = layer["input"]
        output_size = layer["output"]
        params['W' + str(layer_idx)] = np.random.randn(output_size, input_size) * 0.1
        params['b' + str(layer_idx)] = np.random.randn(output_size, 1) * 0.1
        
    return params

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def relu(u):
    return max(u, 0)
    
def single_layer_forward(A_prev, W_curr, b_curr, activation):
    Z_curr = W_curr.dot(A_prev) + b_curr
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        print ("Unknown activation function")
        return
    return activation_func(Z_curr), Z_curr #return Z too for easier backprop

def forward(input_data, params):
    cache = {}
    
    A_curr = input_data
    
    for idx, layer in enumerate(nn_arch):
	# we number network layers from 1
        layer_idx = idx + 1
        A_prev = A_curr
        W_curr = params['W' + str(layer_idx)]
        b_curr = params['b' + str(layer_idx)]
        activation = layer["activation"]
        A_curr, Z_curr = single_layer_forward(A_prev, W_curr, b_curr, activation)
        cache['A' + str(idx)] = A_prev
        cache['Z' + str(layer_idx)] = Z_curr
        
    return A_curr, cache #return NN output and cache of computed values

#def get_cost_value(Y_hat, Y):
#    m = Y_hat.shape[1] # matrix of shape 1 x number of examples
#    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
#    return np.squeeze(cost)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
    m = A_prev.shape[1] # matrix of shape 1 x number of examples
    if activation == "relu":
        activation_func = relu_backward
    elif activation == "sigmoid":
        activation_func = sigmoid_backward
    else:
        print ("Unknown activation function")
        return
    
    dZ_curr = activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr
    
def backward(Y_hat, Y, cache, params):
    gradient = {}
    m = Y.shape[1] # matrix of shape 1 x number of examples

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat)); # deritive of binary_cross_entropy
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_arch))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        
        activ_function_curr = layer["activation"]
        dA_curr = dA_prev
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev, 
                                                          activ_function_curr)
        
        gradient["dW" + str(layer_idx_curr)] = dW_curr
        gradient["db" + str(layer_idx_curr)] = db_curr
    
    return gradient

def update(params, gradient, learning_rate):
    for layer_idx, layer in enumerate(nn_arch, 1):
        params["W" + str(layer_idx)] -= learning_rate * gradient["dW" + str(layer_idx)]        
        params["b" + str(layer_idx)] -= learning_rate * gradient["db" + str(layer_idx)]
    return params;

def train(X, Y, epochs, learning_rate):
    
    add_layer(25, "relu", 2)
    add_layer(50, "relu")
    add_layer(50, "relu")
    add_layer(25)
    add_layer(1, "sigmoid")
    
    params = init_layers()

    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cache = forward(X, params)
        
        # TODO:
        # calculate cost and accuracy and add to history
        
        gradient = backward(Y_hat, Y, cache, params)

        params = update(params, gradient, learning_rate)
        
        # TODO:
        # print cost, accuracy etc
            
    return params
