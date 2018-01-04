import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class BayesClassifier(object):

    def fit(self, X, y):
        labels = set(y)
        # first look at the zero class calculate the number of 0/total --> Beta
        # Beta for class 0 (y==0 +1) / (N + 2)
        N = len(y)
        self.py_Y = np.empty(len(labels))
        self.precision_arr = np.empty([2,15])
        self.df_arr = np.empty([2,15])
        self.mu_arr = np.empty([2,15])

        for k in labels:
            # print np.sum(y == k)
            self.py_Y[k] = (np.sum(y == k) + 1.0) / (N + 2.0) #returns all the rows where the class is 0
            
            mu_0 = 0
            a_0 = 1
            b_0 = 1
            c_0 = 1

            Xk = X[y == k]
            Nk, D = Xk.shape

            # find mu --> m, alpha --> b, beta --> c, kappa --> a
            for d in xrange(D):
                col = Xk[:,d]
                mu = (mu_0*a_0 + Nk*col.mean()) / (a_0 + Nk)
                a = a_0 + Nk
                b = b_0 + Nk/2.0
                c = c_0 + 0.5 * Nk * col.var() + (col.mean()-mu_0)**2/(2.0*(a_0+Nk))

                # find the 3 params for the t-distribution using the above
                precision = np.sqrt((b*a)/(c*(a+1)))
                df = 2*b
                mu = mu
                self.precision_arr[k,d] = precision
                self.df_arr[k,d] = df
                self.mu_arr[k,d] = mu
        # print "py_Y:", self.py_Y
        # print "precisions:", self.precision_arr
        # print "dfs:", self.df_arr
        # print "mus:", self.mu_arr
    

    def score(self, X, Y):
        y_hat = self.predict(X)
        score = np.mean(Y == y_hat)
        return score

    def predict(self, X): #pass in data with 50 rows
        return np.round(self.predict_proba(X)) #returns a vector of 50


    #defines the posterior
    def predict_proba(self, X):
        N, D = X.shape
        # posterior = np.ones([N,2])
        posterior = np.empty([N,2])

        for c in xrange(2):
            posterior[:,c] = self.py_Y[c]
            # loop through d=0 to 14
            # current_d = 1
            for d in xrange(D):
            #define the t-distribution for each
                df_t = self.df_arr[c,d]
                # print "X.shape:", X.shape, "d:", d
                col = X[:,d]
                mu_t = self.mu_arr[c,d]
                scale_t = 1/self.precision_arr[c,d]
                # pdf(X, self.mu[d], self.precision[d])
                new_probability = t.pdf(col, df_t, loc=mu_t, scale=scale_t)
                # posterior = posterior * new_probabiity
                posterior[:,c] = posterior[:,c] * new_probability
            # results from above prodcut 0-14 * py_Y
            # posterior[:,c] = posterior[:,c]*py_Y[c]

        # non normalized posterior class 0
        # non normalized posterior for class 1
        # some two non normalized posterios
        # class 0 /sum
        # class 1 / sum
        # py given 1
        posterior_final = posterior[:,1] / (posterior[:,0]+posterior[:,1])
        return posterior_final


def showimage(Q, X):

    image = np.dot(Q, X)
    y = np.reshape(image, (28,28))
    plt.imshow(y)
    plt.title(c)
    plt.show()
    
if __name__ == '__main__':

    path = '/Users/laurenmccarthy/Documents/Columbia/Fall2016/BaysianML/hw1_data_csv/'
    Xtrain = pd.read_csv(path+'Xtrain.csv', header=None).as_matrix()
    ytrain = pd.read_csv(path+'ytrain.csv', header=None).as_matrix().flatten()
    Xtest = pd.read_csv(path+'Xtest.csv', header=None).as_matrix()
    ytest = pd.read_csv(path+'ytest.csv', header=None).as_matrix().flatten()
    Q = pd.read_csv(path+'Q.csv', header=None)
   

    #a = implement the classifier
    bayesclassifier = BayesClassifier()

    # ================================================================================
    # TRAINING THE CLASSIFIER
    # ================================================================================

    bayesclassifier.fit(Xtrain, ytrain)
    ypred = bayesclassifier.predict(Xtest)
    yprob = bayesclassifier.predict_proba(Xtest)
    trainaccur = bayesclassifier.score(Xtrain, ytrain)
    testaccur = bayesclassifier.score(Xtest, ytest)

    print 'train accuracy:'
    print trainaccur
    print 'test accuracy:'
    print testaccur



    # ================================================================================
    # CONFUSION MATRIX part b
    # ================================================================================

    print "Confusion matrix:"
    
    print confusion_matrix(ytest, ypred)
    # C = np.zeros((2,2), dtype=int)
    # for p,t in zip(ypred.flatten(), ytest.flatten()):
    #     C[t,p] += int(1)
    # print C
    
    # ================================================================================
    # 3 MISSCLASSIFIED EXAMPLES c
    # ================================================================================

    print "Distributions for 3 misclassified examples:"
    count = 0
    for i,p in enumerate(ypred):
        if p != ytest[i]:

            #class 0 is 4, class 1 is 9
            print "sample:", i, "predicted:", p, "actual:", ytest[i], "probability:", yprob[i]
            image = np.dot(Q, Xtest[i])
            y = np.reshape(image, (28,28))
            plt.imshow(y)
            plt.title(yprob[i])
            # plt.savefig('Plot_number%i,jpg' i)
            plt.show()
            count += 1
        if count >= 3:
            break

    # ================================================================================
    # 3 MOST AMBIGUOUS EXAMPLES d
    # ================================================================================

    print "Most abiguous examples:"
    count = 0
    print np.argsort(np.absolute(yprob-.5))
    ambiguous = [511,  637, 1136]
    for i in ambiguous:
        print "sample:", i, "predicted:", ypred[i], "actual:", ytest[i], "probability:", yprob[i]
        image = np.dot(Q, Xtest[i])
        y = np.reshape(image, (28,28))
        plt.imshow(y)
        plt.title(yprob[i])
        plt.show()
        count += 1
        if count >= 3:
            break

