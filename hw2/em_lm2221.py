import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sortedcontainers import SortedList

class ProbitRegression():

    def fit(self, X, Y):
        # labels = set(y)
        N, D = X.shape
        sigma = 1.5
        T = 100
        lam = 1
        # Initialize w0 to a vector of all zeros.
        self.E_phi = np.empty(N)
        w = np.random.randn(D) / np.sqrt(D+1)
        joint_likelihood = [] # t values
        wt = [] # t values
        

        Xx = 0
        for i in xrange(N):
            Xx += np.outer(X[i],X[i])
        
        first_term = np.linalg.inv(lam * np.eye(D) + Xx/sigma**2.0)

        # in each interation
        for t in xrange(T):
            print t


            # joint_likelihood
            a = (D/2.0) * np.log(lam/ (2.0*np.pi))
            b = -(lam/2.0)* np.dot(w,w)

            c = 0
            d = 0
            for i in xrange(N):
                cdf_2 = norm.cdf(np.dot(X[i],w)/sigma)
                c += Y[i]*np.log(cdf_2)
                d += (1-Y[i])*np.log(1-cdf_2) 

            joint_likelihood_current = a + b + c + d
            joint_likelihood.append(joint_likelihood_current)


            print "cost:", joint_likelihood_current

            # E-step
            for i in xrange(N):
                Xw = np.dot(X[i],w)
                pdf = norm.pdf(-Xw/sigma)
                cdf = norm.cdf(-Xw/sigma)
                if Y[i] == 1:
                    self.E_phi[i] = Xw + sigma*pdf/(1-cdf)
                else: 
                    self.E_phi[i] = Xw -sigma*pdf/(cdf)


            # M-step
            second_term = 0
            for i in xrange(N):
                second_term += X[i]*self.E_phi[i]/sigma**2.0
            
            w = np.dot(first_term,second_term)
            wt.append(w)


        self.w = w
        return wt, joint_likelihood


    def score(self, X, Y):
        y_hat = self.predict(X)
        score = np.mean(Y == y_hat)
        return score

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        sigma = 1.5
        return norm.cdf(np.dot(X, self.w)/sigma)

    def confusion_matrix(self, X, Y):
        P = self.predict(X)
        M = np.zeros((2, 2))
        M[0,0] = np.sum(P[Y == 0] == Y[Y == 0])
        M[0,1] = np.sum(P[Y == 0] != Y[Y == 0])
        M[1,0] = np.sum(P[Y == 1] != Y[Y == 1])
        M[1,1] = np.sum(P[Y == 1] == Y[Y == 1])
        return M

    def get_3_misclassified(self, X, Y):
        P = self.predict(X)
        N = len(Y)
        samples = np.random.choice(N, 3, replace=False, p=(P != Y)/float(np.sum(P != Y)))
        # probability = predict_proba[X]
        return X[samples], Y[samples], P[samples]

    # def get_3_most_ambiguous(self, X, Y):
    #     P = self.predict_proba(X)
    #     N = len(X)
    #     sl = SortedList(load=3) # stores (distance, sample index) tuples
    #     for n in xrange(N):
    #         p = P[n]
    #         dist = np.abs(p - 0.5)
    #         if len(sl) < 3:
    #             sl.add( (dist, n) )
    #         else:
    #             if dist < sl[-1][0]:
    #                 del sl[-1]
    #                 sl.add( (dist, n) )
    #     indexes = [v for k, v in sl]
    #     # probability = predict_proba[X]
    #     return X[indexes], Y[indexes], P[indexes]


def showimage(Q, X, title, savetitle):
    image = np.dot(Q, X)
    y = np.reshape(image, (28,28))
    plt.imshow(y)
    plt.ylabel('')
    plt.xlabel('')
    plt.title(title)
    plt.savefig(savetitle)

if __name__ == '__main__':

    path = '/Users/laurenmccarthy/Documents/Columbia/Fall2016/BaysianML/hw2/hw2_data_csv/'
    savepath = '/Users/laurenmccarthy/Documents/Columbia/Fall2016/BaysianML/hw2/images/'
    
    Xtest = pd.read_csv(path+'Xtest.csv', header=None).as_matrix()
    Xtrain = pd.read_csv(path+'Xtrain.csv', header=None).as_matrix()
    Ytest = pd.read_csv(path+'ytest.csv', header=None).as_matrix().flatten()
    Ytrain = pd.read_csv(path+'ytrain.csv', header=None).as_matrix().flatten()


    # ================================================================================
    # a) TRAINING THE CLASSIFIER
    # ================================================================================

    model = ProbitRegression()
    wt, joint_likelihood = model.fit(Xtrain, Ytrain)
    ypred = model.predict(Xtest)
    yprob = model.predict_proba(Xtest)
    trainaccur = model.score(Xtrain, Ytrain)
    testaccur = model.score(Xtest, Ytest)

    print 'train accuracy:', trainaccur
    print 'test accuracy:', testaccur


    # ================================================================================
    # b) Plot ln p(~y, wt|X) as a function of t.
    # ================================================================================

    plt.plot(joint_likelihood)
    plt.title('ln p(~y, wt|X) as a function of t')
    plt.ylabel('Likelihood')
    plt.xlabel('t')
    plt.savefig(savepath+'Joint_likelihood.png')
    
    # ================================================================================
    # c) Confusion Matrix
    # ================================================================================

    M = model.confusion_matrix(Xtest, Ytest)
    print "confusion matrix:"
    print M
    print "N:", len(Ytest)
    print "sum(M):", M.sum()

    # ================================================================================
    # d) Pick three misclassified digits
    # ================================================================================

    Q = pd.read_csv(path+'Q.csv', header=None).as_matrix()
    # misclassified, targets, predictions = model.get_3_misclassified(Xtrain, Ytrain)
    # for i, (x, y, p) in enumerate(zip(misclassified, targets, predictions)):
    # # for i, (a, b) in enumerate(zip(alist, blist)):
    #     t = yprob[i]
    #     showimage(Q, x, 'misclassified target=%s prediction=%s probability=%s' % (y, int(p), t), savepath+'Misclassified'+str(i)+'.png' )

    # print "Distributions for 3 misclassified examples:"
    count = 0
    for i,p in enumerate(ypred):
        if p != Ytest[i]:
            showimage(Q, Xtest[i], 'misclassified target=%s predictions=%s probability=%s' % (Ytest[i], ypred[i], yprob[i]), savepath+'Misclassified'+str(count)+'.png')
            #class 0 is 4, class 1 is 9
            # print "sample:", i, "predicted:", p, "actual:", ytest[i], "probability:", yprob[i]
            # image = np.dot(Q, Xtest[i])
            # y = np.reshape(image, (28,28))
            # plt.imshow(y)
            # plt.title(yprob[i])
            # # plt.savefig('Plot_number%i,jpg' i)
            # plt.show()
            count += 1
        if count >= 3:
            break
    # ================================================================================
    # e) Pick the three most ambiguous predictions
    # ================================================================================

    # ambiguous, targets, predictions = model.get_3_most_ambiguous(Xtrain, Ytrain)
    # for i, (x, y, p) in enumerate(zip(ambiguous, targets, predictions)):
    #     t = yprob[i]
    #     showimage(Q, x, 'ambiguous target=%s predictions=%s probability=%s' % (y, int(p), t))
    #     plt.savefig(savepath+'Ambiguous'+str(i)+'.jpg')

    # print "Most abiguous examples:"
    count = 0
    ambiguous =  np.argsort(np.absolute(yprob-.5))
    for i in ambiguous:
        showimage(Q, Xtest[i], 'ambiguous target=%s predictions=%s probability=%s' % (Ytest[i], ypred[i], yprob[i]), savepath+'Ambiguous'+str(count)+'.png')
        count += 1
        if count >= 3:
            break

    # ================================================================================
    # f) Treat the vector w_t as if it were a digit and reconstruct it as an image for 
    # t = 1, 5, 10, 25, 50, 100.
    # Show these images and comment on what you observe.
    # ================================================================================

    t = (1, 5, 10, 25, 50, 100)
    # t = (1, 5, 10)
    for tt in t:
        vector = wt[tt-1]
        showimage(Q, vector, 'Weights for t =%s' % tt, savepath+'Plotweights'+str(tt)+'.png')
        # viewing the weights as an image - red where w is pos, blue is where w is negative  