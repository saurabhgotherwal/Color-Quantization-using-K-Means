import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

UBIT = "sgotherwa"
np.random.seed(sum([ord(c) for c in UBIT]))

FOLDER_PATH = ''

def GetCoordinates():
    x = np.asarray([5.9,4.6,6.2,4.7,5.5,5.0,4.9,6.7,5.1,6.0])
    y = np.asarray([3.2,2.9,2.8,3.2,4.2,3.0,3.1,3.1,3.8,3.0])
    coordinates = np.array(list(zip(x,y)), dtype=np.float32)
    return x , y, coordinates

def CalculateEuclideanDistance(point1, point2):
    return np.linalg.norm(point1 - point2, axis = 1)

def GetInitialCentroids():
    centroids = np.array(([(6.2,3.2),(6.6,3.7),(6.5,3.0)]), dtype=np.float32)
    return centroids

def GetInitialCentroidsForImageClustering(noOfClusters,image):    
    centroids = []    
    indices=np.random.randint(image.shape[0],size=noOfClusters)
    centroids=image[indices]        
    return np.float32(centroids)   

def GetUpdatedCentroids(pointsVector,noOfClusters,classificationVector):
    centroids = []
    for i in range(noOfClusters):
        points = [pointsVector[j] 
                  for j in range(len(pointsVector)) 
                  if classificationVector[j] == i]
        centroids.append(np.mean(points, axis=0))
    return np.asarray(centroids)

def GetClassificationVector(pointsVector, centroids):
    classificationVector = []
    for i in range(len(pointsVector)):
        distances = CalculateEuclideanDistance(pointsVector[i], centroids)
        cluster = np.argmin(distances)
        classificationVector.append(cluster) 
    return np.asarray(classificationVector)
    
def KMeans(centroids, pointsVector, noOfClusters, noOfItterations = 20):    
    CentroidsOld = np.zeros(centroids.shape)    
    classificationVector = np.zeros(len(pointsVector))    
    itteration = 0    
    while itteration < noOfItterations:        
        classificationVector = GetClassificationVector(pointsVector, centroids)
        for i in range(0,noOfClusters):
            CentroidsOld[i]=centroids[i]
        centroids = GetUpdatedCentroids(pointsVector,noOfClusters,classificationVector)
        itteration += 1
        if(np.array_equal(CentroidsOld,centroids)):
            break    
    return centroids, classificationVector

def SaveClassificationPlot(figName, pointsVector, numberOfClusters,clusters):
    colors = ['r', 'g', 'b']    
    for i in range(numberOfClusters):
        points = np.array([pointsVector[j] for j in range(len(pointsVector)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1],marker="^",s=100, c=colors[i])        
        for coords in points:
                plt.annotate(coords,(coords[0],coords[1]))        
    plt.savefig(FOLDER_PATH + figName)
    plt.close()

def SaveCentroidPlot(figName,centroids, numberOfClusters):
    colors = ['r', 'g', 'b']    
    for i in range(numberOfClusters):   
        plt.scatter(centroids[i, 0], centroids[i, 1], marker='o',s=150, c=colors[i])
        for coords in centroids:
                plt.annotate(coords,(coords[0],coords[1]))        
    plt.savefig(FOLDER_PATH + figName)
    plt.close()

def DoImageQuantization(numberOfClusters,imageCoordinates,img):
    imageCentroids = GetInitialCentroidsForImageClustering(numberOfClusters,imageCoordinates)
    updatedImageCentroids, classificationVectorImage = KMeans(imageCentroids, imageCoordinates, numberOfClusters)    
    updatedCentroids = np.uint8(updatedImageCentroids)
    image_output_flattened = updatedCentroids[np.uint8(classificationVectorImage.flatten())]
    output_image = image_output_flattened.reshape(img.shape)
    return output_image

centroids = GetInitialCentroids()
x , y, coordinates = GetCoordinates()
updatedCentroids, classificationVector = KMeans(centroids, coordinates, 3, 1)
print(classificationVector)
SaveClassificationPlot("task3_iter1_a.jpg",coordinates,3,classificationVector)
SaveCentroidPlot("task3_iter1_b.jpg",updatedCentroids,3)
updatedCentroids, classificationVector = KMeans(centroids, coordinates, 3, 4)
print(classificationVector)
SaveClassificationPlot("task3_iter2_a.jpg",coordinates,3,classificationVector)
SaveCentroidPlot("task3_iter2_b.jpg",updatedCentroids,3)

K_list = [3,5,10,20]

for K in K_list:  
    img = cv.imread(FOLDER_PATH + 'baboon.jpg',1)
    coordinatesImage = img.reshape((-1,3))
    coordinatesImage = np.float32(coordinatesImage)
    output1 = DoImageQuantization(K,coordinatesImage,img)
    im_name = "task3_baboon_" + str(K) + ".jpg"
    cv.imwrite(FOLDER_PATH + im_name,output1)
