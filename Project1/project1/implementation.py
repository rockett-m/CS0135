import numpy as np
import sklearn.metrics
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """
    # TODO: implement
    A = np.array(X1)
    B = np.array(X2)
    # B.transpose() # transpose B for matrix multiplication # dot product
    # print(f'{A = }\n{B = }\n')

    # C = np.zeros(X1.shape[0])
    # C = (A @ B.transpose()) ** 2
    C = np.dot(A, B.transpose())

    C = np.array(C)

    return C


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # (Bonus) TODO: implement 

    # Compute the Euclidean distance between the input vectors
    # Compute the value of the Gaussian kernel function
    # Return the kernel value
    return None


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """
    # TODO: implement
    
    # Reshape a and y to be column vectors
    a.reshape(-1, 1)
    y.reshape(-1, 1)

    # calculate the distance

    # Compute the value of the objective function
    # The first term is the sum of all Lagrange multipliers  # np.sum(a)
    # The second term involves the kernel matrix (X), the labels (y) and the Lagrange multipliers (a)

    # obj_val = np.zeros(len(a))
    # obj_val = np.sum(a) - (0.5 * np.sum( np.dot(a, a.T) * np.dot(y, y.T) * (kernel(X, X)) ) )
    # obj_val = np.sum(a) - (0.5 * a.T @ (y.T * kernel(X, X) * y) @ a) # didn't work

    xya_total = 0

    for i, val0 in enumerate(a):
        for j, val1 in enumerate(a):
            xya_total += (a[i] * a[j] * y[i] * y[j] * kernel(X, X)[i, j])

    obj_val = np.sum(a) - (0.5 * xya_total)
    # print(f'{obj_val = }')
    return obj_val


class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute 
          the kernel matrix from data matrices; that matrix should be an array 
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features 
          is the number of features. For kernel=”precomputed”, the expected shape 
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        # save alpha parameters, weights, and bias weight

        # TODO: Define the constraints for the optimization problem
        constraints = ({'type': 'ineq', 'fun': lambda a: a },
                       {'type': 'eq',   'fun': lambda a: np.dot(a, y) }) # a.transpose()
                        # {'type': 'eq', 'fun': lambda a: np.dot(y.T, a)})  # a.transpose()

        # X (n_samples, n_features) # matrix of data x1, x2, ... xn
        # y (n_samples,) # vector of labels y1, y2, y3

        # ai + yi = 0 summation # a.T.y = 0
        # constraints are compared to 0  # ineq by default ... loops through all values of a

        # 'type': 'ineq', # > 0 # goes through all vals of a
        # 'type': 'eq', # == 0 # dot product  a[0] * y[0] + a[1] * y[1] + ... + a[n] * y[n]
        # with comparison to 0 ## > or < ## scipy.minimize constraints

        # TODO: Use minimize from scipy.optimize to find the optimal Lagrange multipliers
        self.a = np.zeros(X.shape[0])  # X : shape (n_samples, n_features) # n_samples
        res = minimize( (lambda a: (-objective_function(X=X, y=y, a=a, kernel=self.kernel)) ),
                       x0=self.a, constraints=constraints)  # fun(x, *args) -> float

        self.a = np.array(res.x)

        # TODO: Substitute into dual problem to find weights
        # self.w = ...  ## Coefficient Vector (ndarray with shape (n_features, )
        self.w = np.zeros(X.shape[1])  # X : shape (n_samples, n_features) # features

        # self.a = array([1.47549954e-01, 4.77448320e-01, 1.46822658e-15, 4.08827517e-01, 2.16170757e-01])
        for idx, alpha in enumerate(self.a):
            if alpha > 1e-8: # if alpha isn't super small
                # equivalent
                # self.w[0] += alpha * y[idx] * X[idx][0]
                # self.w[1] += alpha * y[idx] * X[idx][1]
                # self.w[2] += alpha * y[idx] * X[idx][2]
                self.w += (alpha * y[idx] * X[idx])  # optimal Lagrange values

        # result.coef_ = array([[0., 0.5, 0.99969451]])
        # svm.w = array([-1.44328993e-15, 4.99986204e-01, 1.00000345e+00])

        # TODO: Substitute into a support vector to find bias
        y.reshape(-1, 1)  # reshape y to proper dims to calc bias
        self.b = 0.0  ## Intercept Term (float)
        self.b = -0.5 * ( max(np.inner(self.w, X[y == -1])) + min(np.inner(self.w, X[y == 1])) )

        print(f'{self.a = }')
        print(f'{self.w = }')
        print(f'{self.b = }\n')

        '''
        res = message: Optimization
        terminated
        successfully
        success: True
        status: 0
        fun: 0
        x: [0.000e+00  0.000e+00  0.000e+00  0.000e+00  0.000e+00] # array we want
        nit: 1
        jac: [0.000e+00  0.000e+00  0.000e+00  0.000e+00  0.000e+00]
        nfev: 6
        njev: 1
        '''

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement

        # w.T @ X + b # gives predicted value of the new point
        # w · x + b = (w1x1 + w2x2 + · · · + wnxn) + b
        # then apply threshold function to determine +1, -1
        # w · x ! 0
        # +1   w @ X + b >= 0
        # -1   w @ X + b < 0

        # for idx, weight in enumerate(self.w):
        # result = linear_kernel(self.w, X) + self.b # also works
        result = np.dot(self.w, X.T) + self.b
        # print(f'{result = }')

        y_pred = np.zeros(X.shape[0])

        for idx, val in enumerate(result):
            if val >= 0:
                y_pred[idx] = 1
            else:
                y_pred[idx] = -1

        return y_pred

    # removed def outputs(X): # was the same thing as # def predict(self, X):

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels. 

        In multi-label classification, this is the subset accuracy which is a harsh 
        metric since you require for each sample that each label set be correctly 
        predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          True labels for X.

        Return
        ------
        score : float
          Mean accuracy of self.predict(X)
        """
        # TODO: implement
        score = 0.0
        from sklearn.metrics import accuracy_score

        acc_list = []
        for idx, x in X:
            out = self.predict(x)
            acc_list.append(out)

        return score


def one_versus_the_rest(train_mn, train_labels_mn):
    ...