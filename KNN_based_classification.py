import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(datadict['labels'])
        return X,Y


class KNN():
    def __init__(self,k=5,n=5):
        self.k = k
        self.n = n

    def train(self,train_data_X, train_data_Y):
        self.X = train_data_X
        self.Y = train_data_Y
        size,dim = train_data_X.shape
        size = int(size/self.n)
        kfold = np.array([train_data_X[i*size:(i+1)*size] for i in range(self.n)])
        yfold = np.array([train_data_Y[i*size:(i+1)*size] for i in range(self.n)])
        bestacc = np.zeros((0,2))
        while True:
            acctemp = []
            for i in range(self.n):
                self.X = kfold[:i]
                self.X = np.append(self.X, kfold[i+1:], axis=0)
                self.X = self.X.reshape(size*(self.n-1),dim)
                val_X = kfold[i]
                self.Y = yfold[:i]
                self.Y = np.append(self.Y, yfold[i+1:])
                self.Y = self.Y.reshape(size*(self.n-1))
                val_Y = yfold[i]
                ypr = np.zeros(0)
                ypr = self.predict(val_X)
                acctemp.append(self.compute_Accuracy(val_Y, ypr))
            newcaa = np.mean(np.array(acctemp))
            print("k=",self.k,"accuracy=",newcaa)
            if newcaa > bestacc[:,1].any():
                bestacc = np.append(bestacc, [[self.k, newcaa]], axis=0)
            if bestacc.size>10 or self.k>size:
                break
            self.k += 5
        self.k = int(bestacc[np.argmax(bestacc[:,1])][0])
        self.X = train_data_X
        self.Y = train_data_Y

    def predict(self,testdata):
        nearestk = np.zeros(self.k)
        nearestkY = np.zeros(self.k)
        size,dim = self.X.shape
        ypr = np.array([])
        for x in testdata:
            for i in range(size):
                dist = self.calc_dist(self.X[i],x)
                if i<self.k:
                    nearestk[i] = dist
                    nearestkY[i] = self.Y[i]
                if (dist<nearestk.any()):
                    index = np.where(nearestk == np.min(nearestk))[0]
                    nearestk[index] = dist
                    nearestkY[index] = self.Y[i]
            count = np.zeros(10)
            for data in nearestkY:
                count[int(data)] += 1
            ypr = np.append(ypr, np.argmax(count))
        return ypr

    def calc_dist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def compute_Accuracy(self,val,pr):
        P = np.sum(val==pr)
        acc = P/val.size
        return acc

def get_feature(dataset):
    h,w,depth = dataset[0].shape
    features = np.zeros((0,h*w*depth))
    for image in dataset:
        vec = image.flatten()
        vec = vec / np.sum(vec**2)
        features = np.append(features, [vec], axis=0)
    return features

if __name__ == "__main__":
    train_x_img,train_y = load_CIFAR_batch("./cifar-10-py/data_batch_1")
    train_x = get_feature(train_x_img)
    knn = KNN()
    knn.train(train_x,train_y)

    test_x_img,test_y = load_CIFAR_batch("./cifar-10-py/test_batch")
    test_x = get_feature(test_x_img[:1000])
    pr = knn.predict(test_x)
    acc=knn.compute_Accuracy(test_y[:1000],pr)
    print(acc)