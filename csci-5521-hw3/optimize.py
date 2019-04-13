
def batch_gradient_descent(w, iterations, eta, loss_grad, verbose=False):
    '''
    Implements batch gradient descent.  Note that this implementation
    does not use convergence of the loss function, you must specify
    the number of iterations to execute

    '''
    prev_loss = None
    for i in range(0, iterations):
        loss, grad = loss_grad(w)
        w -= (eta * grad)
        if prev_loss is not None and verbose:
            print('loss: {}, diff: {}'.format(loss, prev_loss - loss))
        prev_loss = loss
    return w
