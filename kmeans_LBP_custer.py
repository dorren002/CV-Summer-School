import numpy as np
import random,pickle
import matplotlib.pyplot as plt
import cv2

# load the dataset
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X

def calc_Dist(vec1,centroid):
    if len(centroid)==1:
        return np.sqrt(np.sum((vec1-centroid[0])**2))
    index = 0
    for i in range(len(centroid)):
        temp = np.sqrt(np.sum((vec1-centroid[i])**2))
        if i==0:
            minn=temp
        if temp<minn:
            index = i
            minn = temp
    return index


def calc_NewCentroid(dataset, clas):
    summ = 0
    for i in clas:
        summ += dataset[i]
    return summ/len(clas)


def k_means(dataset, k, th):
    size, dim = dataset.shape
    centroid = np.zeros((k,dim))
    indexes = np.ones(k)*(-1)
    i = 0
    while i<k:
        index = random.randint(0, size-1)
        if index != np.any(indexes):
            centroid[i] = dataset[index]
            i+=1
    error = th+0.1
    clas = {k:[] for k in range(10)}
    reclas = np.zeros(size)
    epoch=0
    while error>th:
        if True:
            clas = {k:[] for k in range(10)}
            reclas=np.zeros(size)
        for j in range(size):
            index = calc_Dist(dataset[j],centroid)
            clas[index].append(j)
            reclas[j] = index
        errork=-1
        for h in range(k):
            newcent = calc_NewCentroid(dataset, clas[h])
            if h==0:
                errork = calc_Dist(newcent,[centroid[h]])
            else:
                error = max(errork, calc_Dist(newcent,[centroid[h]]))
            centroid[h] = newcent
        epoch+=1
        if epoch%10==0:
            print('epoch:', epoch)
            print('\terror:',error)

    with open("model",'w')as f:
        for h in centroid:
            f.write(str(h))
            f.write('\n')

    return reclas


def calc_LBP(window):
    binary=[]
    for i in range(3):
        for j in range(3):
            if i==1 and j==1:
                continue
            if window[i][j] >= window[1][1]:
                binary.append(1)
    res=0
    for k in range(len(binary)):
        res += binary[k] * 2 ** (len(binary)-k-1)
    return res


def calc_LBP_Hist(feature):
    hist = np.zeros(256)
    for f in feature:
        hist[f]+=1
    return hist


def zero_padding(img):
    h,w = img.shape
    res = np.zeros((h+2,w+2),dtype='uint8')
    res[1:h+1,1:w+1]=img
    return res


def main(k):
    filename = './cifar-10-py/data_batch_'
    h,w,depth = 32,32,3
    vectors = np.zeros((0,h*w*depth))
    for xx in range(1,k):
        data = load_CIFAR_batch(filename+str(xx))
        # h,w,depth = data[0].shape
        for img in data:
            for i in range(depth):
                feature = []
                img1 = zero_padding(img[:,:,i])
                for x in range(1,h+1):
                    for y in range(1,w+1):
                        res = calc_LBP(img1[x-1:x+2,y-1:y+2])
                        vector.append(res)
                hist = calc_LBP_Hist(vector)
                vectors = np.append(vectors,[hist],axis=0)
        del data
        print("batch_",xx,'done!')

    res = k_means(vectors, 10, 0.01)
    with open("new2.txt",'w')as f:
        for i in range(len(res)):
            f.write(str(int(res[i])))
            f.write('\n')


def drawLBP(filename):
    img = cv2.imread("te.jpg")
    h,w,depth=img.shape
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mi = np.zeros((h,w))
    img1 = zero_padding(img)
    for x in range(1,h+1):
        for y in range(1,w+1):
            res = calc_LBP(img1[x-1:x+2,y-1:y+2])
            mi[x-1][y-1]=res
    plt.subplot(121)
    plt.imshow(img,plt.cm.gray)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(mi,plt.cm.gray)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main(1)
   