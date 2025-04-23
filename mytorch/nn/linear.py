import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        shape_A = A.shape
        batch_size = 1
        for dim in shape_A[:-1]:
            batch_size *= dim
        A_reshaped = A.reshape(-1, shape_A[-1])
        self.A_reshaped = A_reshaped
        Z_shaped = A_reshaped @ self.W.T + self.b

        Z = Z_shaped.reshape(shape_A[:-1] + (self.b.shape[0], ))
        
        return Z
        # raise NotImplementedError

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        shape_grad = dLdZ.shape
        dLdZ_reshaped = dLdZ.reshape(-1, shape_grad[-1])
        dLdA_re = dLdZ_reshaped @ self.W
        self.dLdA = dLdA_re.reshape(shape_grad[:-1] + (self.W.shape[1],))
        self.dLdW = dLdZ_reshaped.T @ self.A_reshaped
        # self.dLdb = dLdZ_reshaped.T @ ones
        self.dLdb = np.sum(dLdZ_reshaped, axis=0)

        # self.dLdA = NotImplementedError
        # Compute gradients (refer to the equations in the writeup)
        # self.dLdA = NotImplementedError
        # self.dLdW = NotImplementedError
        # self.dLdb = NotImplementedError
        # self.dLdA = NotImplementedError
        return self.dLdA
        # Return gradient of loss wrt input
        # raise NotImplementedError
