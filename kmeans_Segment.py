import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import random

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
    
    return reclas, centroid


def getColorFeature(img):
    h,w,depth = img.shape
    res = np.zeros((0,5))
    for x in range(h):
        for y in range(w):
            res=np.append(res,[img[x][y]],axis=0)
    return res


def getColorLocFeature(img):
    h,w,depth = img.shape
    res = np.zeros((0,5))
    for x in range(h):
        for y in range(w):
            temp = [t for t in img[x][y]]
            temp.append(x)
            temp.append(y)
            res=np.append(res,[temp],axis=0)
    return res


def reConstruct(clas,centroid,x,y):
    res = np.zeros((x,y,3))
    k=0
    for i in range(x):
        for j in range(y):
            res[i][j] = centroid[int(clas[k])][:3]
            k+=1
    return res/255.0


def downsampling(img):
    x,y,depth = img.shape
    res = np.zeros((int(x/2),int(y/2),depth))
    for i in range(x-1):
        for j in range(y-1):
            if i%2==0 and j%2==0:
                res[int(i/2)][int(j/2)] = img[i][j]
    return res

if __name__ == "__main__":
    img = Image.open("ma.jpg")
    img1 = np.asarray(img, dtype='uint8')
    img1 = downsampling(img1)
    feature = getColorLocFeature(img1)
    k=8
    res,centroid = k_means(feature,k,0.01)
    x,y,depth = img1.shape
    img2 = reConstruct(res,centroid,x,y)
    plt.subplot(121)
    plt.imshow(img1/255.0)
    plt.title("original picture")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(img2)
    plt.title("color+location k="+str(k))
    plt.axis("off")
    plt.show()