def segmoid(z):
    return(1/(1+np.exp(-z)))
def costClassification(theta,X,y):
    h=segmoid(X.dot(theta))
    logh=np.log(h)
    log1h=np.log(1-h)
    term1=y*logh
    term2=(1-y)*log1h
    return(-np.sum(term1+term2)/(len(X)))
    
    ##################################
    ##################################

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = segmoid(np.dot(x, theta))
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost =costClassification(theta,x,y)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta
