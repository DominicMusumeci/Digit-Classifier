import numpy as np
from scipy.io import loadmat
from Model import neural_network
from RandInitialize import initialize
from Prediction import predict
from scipy.optimize import minimize

data = loadmat('mnist-original.mat')

# data contains integers [0,255]
X = data['data']
X = X.transpose()

# normalize data
X = X / 255

# labels for data
Y = data["label"]
Y = Y.flatten()

# training set 
X_train = X[:60000, :]
Y_train = Y[:60000]

# testing set
X_test = X[60000:, :]
Y_test = Y[60000:]

m = X.shape[0]
input_layer_size = 784 # for each pixel of 28 x 28 image
hidden_layer_size = 100
num_labels = 10 # for each class [0,9]

#Randomly initialize theta1s for NN
initial_theta1 = initialize(hidden_layer_size, input_layer_size)
initial_theta2 = initialize(num_labels, hidden_layer_size)

# Put the parameters into a single column vector
initial_nn_params = np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()))
maxiter = 100
lambda_reg = 0.1 # to avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, Y_train, lambda_reg)

# Minimize cost function and train weights
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
                    options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"] # Trained theta

theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
                              hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                      (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

pred = predict(theta1, theta2, X_test)
print('Test Set Accuracy {:f}'.format((np.mean(pred == Y_test) * 100)))

pred = predict(theta1, theta2, X_train)
print('Test Set Accuracy {:f}'.format((np.mean(pred == Y_train) * 100)))

# Evaluate precision of the model
# true_positive = pred[pred == Y_train].count_nonzero()
# false_positive = pred[pred == 1 & Y_train == 0].count_nonzero()
true_positive = 0
for i in range(len(pred)):
    if pred[i] == Y_train[i]:
        true_positive += 1
false_positive = len(Y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))

# Save thetas
np.savetxt('theta1.txt', theta1, delimiter=' ')
np.savetxt('theta2.txt', theta2, delimiter=' ')


