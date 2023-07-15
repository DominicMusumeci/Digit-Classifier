import numpy as np

def neural_network(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamb):
    # weights are split back to theta, theta2
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))
    
    # forward propagation
    m = X.shape[0]
    one_matrix = np.ones((m, 1))
    X = np.append(one_matrix, X, axis=1)
    a1 = X # input layer
    z2 = np.dot(X, theta1.transpose()) # bias for 
    a2 = 1 / (1 + np.exp(-z2)) # sigmoid activation function for second layer
    one_matrix = np.ones((m, 1))
    a2 = np.append(one_matrix, a2, axis=1)
    z3 = np.dot(a2, theta2.transpose())
    a3 = 1 / (1 + np.exp(-z3)) # sigmoid acitivation function for third layer

    '''
    Transforming Y using one-hot encoding
    For each label between 0 and 9, there will be a vectory of length 10
    where the ith element will be if the label equals i
    '''
    y_vect = np.zeros((m, 10))
    for i in range(m):
        y_vect[i, int(Y[i])] = 1

    # Calculating Cost Function
    J = (1/m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (lamb / (2 * m)) * (
                sum(sum(pow(theta1[:, 1:], 2))) + sum(sum(pow(theta2[:, 1:], 2))))

    #back prop 
    delta3 = a3 - y_vect
    delta2 = np.dot(delta3, theta2) * a2 * (1 - a2)
    delta2 = delta2[:, 1:]

    # gradient
    theta1[:, 0] = 0
    theta1_grad = (1/m) * np.dot(delta2.transpose(), a1) + (lamb / m) * theta1
    theta2[:, 0] = 0
    theta2_grad = (1/m) * np.dot(delta3.transpose(), a2) + (lamb / m) * theta2
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))

    return J, grad