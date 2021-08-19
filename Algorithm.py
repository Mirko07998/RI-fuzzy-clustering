import numpy as np
import math
import random
from os import listdir
from os.path import isfile, join
from numpy import dot
from numpy.linalg import norm
import sys
SMALL_VALUE = 0.0001

class Pretprocessor:
    def __init__(self,folderPath):
        self.folderPath=folderPath
    def storeMapFromFile(self,filePath):
        wordOccurence={}
        with open(filePath) as fp:
            lines = fp.readlines()
            for line in lines:
                wordOccurence[line.split(' ')[0]]=int(line.split(' ')[1])
        return wordOccurence
    def calculateInitialMatrix(self):
        files = [join(self.folderPath, f) for f in listdir(self.folderPath) if isfile(join(self.folderPath, f))]#Sapajanje putanja iz direktorijuma
        row=len(files)
        fileMaps=[]
        allWords=set()
        for f in files:
            fileMap=self.storeMapFromFile(f)
            for key in fileMap.keys():
                allWords.add(key)
            fileMaps.append(fileMap)
        column=len(allWords)
        initMatrix=np.zeros((row,column))
        k = 0
        for i in range(row): #Za svaki dokument pisemo koliko ima reci k kolone i vrste j su reci
            for j in allWords:
                if j in fileMaps[i]:
                    initMatrix[i][k] = (fileMaps[i])[j]
                k += 1
            k = 0
        self.normalize(initMatrix)
        return initMatrix
    def normalize(self,matrix):
        row=matrix.shape[0]
        for i in range(row):
            matrix[i] /= np.sum(matrix[i])
class FCM:

    def __init__(self, n_clusters=5, m=2, max_iter=200):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.u = None  # The membership
        self.m = m  # the fuzziness, m=1 is hard not fuzzy. see the paper for more info
        self.max_iter = max_iter
    #done
    def init_memerships(self, num_of_points):
        self.init_memerships_random(num_of_points)
    #done
    def init_memerships_random(self, num_of_points):
        self.u = np.zeros((num_of_points, self.n_clusters))
        for i in range(num_of_points):
            row_sum = 0.0
            for c in range(self.n_clusters):
                if c == self.n_clusters - 1:  # poslednja iteracija
                    self.u[i][c] = 1.0 - row_sum
                else:
                    rand_clus = random.randint(0, self.n_clusters - 1)
                    rand_num = random.random()
                    rand_num = round(rand_num, 2)
                    if rand_num + row_sum <= 1.0:  # da izbegnemo da suma tacak bude veca od 1.0
                        self.u[i][rand_clus] = rand_num
                        row_sum += self.u[i][rand_clus]
    #done ne zovemo ga nikada zato sto je equal jako los
    def init_memerships_equal(self, num_of_points):
        self.u = np.zeros((num_of_points, self.n_clusters))
        for i in range(num_of_points):
            row_sum = 0
            for c in range(self.n_clusters):
                if c == self.n_clusters - 1:  # poslednja iteracija
                    self.u[i][c] = 1 - row_sum
                else:
                    rand_num = round(1.0 / self.n_clusters, 2)
                    if rand_num + row_sum >= 1.0:
                        if rand_num + row_sum - 0.1 >= 1.0:
                            self.logger.error("Nesto nije uredu sa init_memership")
                            return None
                        else:
                            self.u[i][c] = rand_num - 0.01
                    else:
                        self.u[i][c] = rand_num
                    row_sum += self.u[i][c]
    #done
    def compute_cluster_centers(self, X):
        num_of_points = X.shape[0]
        num_of_features = X.shape[1]
        centers = []

        for c in range(self.n_clusters):
            sum1_vec = np.zeros(num_of_features)
            sum2_vec = 0.0
            for i in range(num_of_points):
                interm1 = (self.u[i][c] ** self.m)
                interm2 = interm1 * X[i]
                sum1_vec += interm2
                sum2_vec += interm1
            if np.any(np.isnan(sum1_vec)):
                raise Exception("There is a nan in compute_cluster_centers method if")
            if sum2_vec == 0:
                sum2_vec = 0.000001
            centers.append(sum1_vec / sum2_vec)
        return centers

    #done

    def distance_squared(self, x, c):
        #sum_of_sq = 0.0
        #for i in range(len(x)):
        #    sum_of_sq += (x[i] - c[i]) **2
        sum_of_sq = dot(x, c) / (norm(x) * norm(c))
        return sum_of_sq
    #done
    def compute_membership_single(self, X, datapoint_idx, cluster_idx):
        clean_X = X
        d1 = self.distance_squared(clean_X[datapoint_idx], self.cluster_centers[cluster_idx])
        sum1 = 0.0
        for c in self.cluster_centers:  # ovo je da izracunamo sigma)
            d2 = self.distance_squared(c, clean_X[datapoint_idx])
            if d2 == 0.0:
                d2 = SMALL_VALUE
            sum1 += (d1 / d2) ** (1.0 / (self.m - 1))
        if np.any(np.isnan(sum1)):
            raise Exception("nan is found in computer_memberhip_single method in the inner for")

        if sum1 == 0:  # because otherwise it will return inf
            return 1.0 - SMALL_VALUE
        if np.any(np.isnan(sum1 ** -1)):
            raise Exception("nan is found in computer_memberhip_single method")
        return sum1 ** -1

    #done
    def update_membership(self, X):
        for i in range(X.shape[0]):
            for c in range(len(self.cluster_centers)):
                self.u[i][c] = self.compute_membership_single(X, i, c)

    #done
    def fit(self, X):
        X = np.array(X)
        if self.u is None:
            num_of_points = X.shape[0]
            self.init_memerships_random(num_of_points)
        for i in range(self.max_iter):
            self.cluster_centers = self.compute_cluster_centers(X)
            self.update_membership(X)

        return self.u
if __name__ == '__main__':
    # np.set_printoptions(threshold=np.inf)
    p=Pretprocessor('dataset')
    X=p.calculateInitialMatrix()
    fcm=FCM()
    T=np.array(fcm.fit(X))
    print(T)