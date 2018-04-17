import matplotlib.pyplot as plt  # Para plotar graficos
import numpy as np  # Array do Python
from math  import sqrt, pi

class RBF(object):
    def __init__(self, etat=0.001, etas=0.0001, etav=0.007, epoch_max=10000, Ni=1, Nh=12, Ns=1):
        self.etat = etat
        self.etas = etas
        self.etav = etav
        self.epoch_max = epoch_max
        self.Ni = Ni
        self.Nh = Nh
        self.Ns = Ns
        self.Wini = 0.01

    def load_function(self):
        x = np.arange(-6, 6, 0.2)
        self.N = x.shape[0]
        xmax = np.max(x)

        self.X_train = x / xmax
        self.d = 1 / (1 + np.exp(-1 * x))*(np.cos(x) - np.sin(x))
    
    def train(self):
        self.t = np.zeros(self.Nh)
        self.V = np.random.rand(self.Ns, self.Nh + 1) * self.Wini

        idx = np.random.permutation(self.Nh)
        for j in xrange(self.Nh):
            self.t[j] = self.d[idx[j]]

        self.t = self.t.reshape(1, -1)

        DEmax = 0
        for j in xrange(self.Nh):
            for i in xrange(self.Nh):
                if (i > j):
                    DE = (self.t[0][i] - self.t[0][j])**2

                    if (DE > DEmax):
                        DEmax = DE
        
        sigma = np.random.rand(1, self.Nh) * ((DEmax**2) / self.Nh) * 100

        MSE = np.zeros(self.epoch_max)
        plt.ion()

        for epoca in xrange(self.epoch_max):
            z = np.zeros(self.N)
            E = np.zeros(self.N)

            idc = np.random.permutation(self.N)

            for n in xrange(self.N):
                i = idc[n]
                xi = np.array([self.X_train[i]]).reshape(1, -1)
                norma = (xi - self.t)**2
                y = np.insert(np.exp(-norma / sigma), 0, 1).reshape(1, -1)
                z[i] = np.dot(self.V, y.T)[0][0]

                e = self.d[i] - z[i]

                self.t += (2 * self.etat * e) * (self.V[:,1:]) * y[:,1:] * (xi - self.t) / sigma
                sigma += (self.etas * e) * self.V[:,1:] * y[:,1:] * norma / (sigma**2)
                self.V += (self.etav * e) * y

                E[i] = 0.5 * e**2

            MSE[epoca] = np.sum(E) / self.N

            if (epoca % 200 == 0 or epoca == self.epoch_max - 1):
                if (epoca != 0):
                    plt.cla()
                    plt.clf()
                
                self.plot(z, epoca)
        
        print MSE[-1]

        return MSE

    def plot(self, saida, epoca):
        plt.figure(0)
        y, = plt.plot(self.X_train, saida, label="y")
        d, = plt.plot(self.X_train, self.d, '*', label="d")
        plt.legend([y, d], ['Output of RBF', 'Desired Value'])
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('RBF')
        plt.text(np.min(self.X_train) - np.max(self.X_train) * 0.17  , np.min(self.d) - np.max(self.d) * 0.17, 'Progress: ' + str(round(float(epoca) / self.epoch_max * 100, 2)) + '%')
        plt.axis([np.min(self.X_train) - np.max(self.X_train) * 0.2, np.max(self.X_train) * 1.2, np.min(self.d) - np.max(self.d) * 0.2, np.max(self.d) * 1.5])
        plt.show()
        plt.pause(1e-10)

    def plot_MSE(self, MSE):
        plt.ioff()
        plt.figure(1)
        plt.title('Mean Square Error (MSE)')
        plt.xlabel('Training Epochs')
        plt.ylabel('MSE')
        plt.plot(np.arange(0, MSE.size), MSE)
        plt.show()


rbf = RBF()

rbf.load_function()
MSE = rbf.train()