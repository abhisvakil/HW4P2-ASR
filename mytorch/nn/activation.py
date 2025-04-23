import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
         
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter

        # self.A = NotImplementedError
        # raise NotImplementedError
        max = np.max(Z, axis = self.dim, keepdims= True)

        Z_new = Z - max
        sum = np.sum(np.exp(Z_new), axis = self.dim, keepdims= True)
        self.A = np.exp(Z_new)/sum
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]

        
           
        # Reshape input to 2D
        if len(shape) > 2:
            moved_dLdA = np.moveaxis(dLdA, self.dim, -1)
            moved_A = np.moveaxis(self.A, self.dim, -1)
            dims = 1
            for dim in moved_dLdA[:-1]:
                dims *= dim
            reshaped_dLdA = moved_dLdA.reshape(-1, C)
            reshaped_A = moved_A.reshape(-1, C)
            N = reshaped_A.shape[0]
            dLdZ_reshaped = np.zeros((N,C))
            for i in range(N):
                J = np.zeros((C, C))
                for m in range(C):
                    for n in range(C):
                        if(m == n):
                            J[m, n] = reshaped_A[i, m] * (1 - reshaped_A[i, m])
                        else:
                            J[m, n] = -reshaped_A[i, m] * reshaped_A[i, n]
                dLdZ_reshaped[i, :] = reshaped_dLdA[i, :] @ J
            # self.A = NotImplementedError
            # dLdA = NotImplementedError
            # self.A = reshaped_A
            # dLdA = reshaped_dLdA

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            # self.A = NotImplementedError
            # dLdZ = NotImplementedError
            moved_dLdZ = dLdZ_reshaped.reshape(shape[:-1] + (C, ))
            dLdZ = np.moveaxis(moved_dLdZ, -1, self.dim)

        return dLdZ
        # raise NotImplementedError
 

    