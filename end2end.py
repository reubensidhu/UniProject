import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
import sys
import math

def nothing(x):
    pass

# Load image
src_img = cv.imread(r'C:\Users\reuby\OneDrive\Pictures\pool1.jpeg')
# Convert image to HSV colour space
hsv_img = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)
# Get shape of image and extract a portion if the image near the middle
h, w, c = hsv_img.shape
cropped_img = (hsv_img[int(0.4*h):int(0.75*h),int(0.25*w):int(0.75*w)].copy())

# KMeans with cluster of 4
clt = KMeans(n_clusters=4)
clt.fit(cropped_img.reshape(-1, 3))
clt.labels_
clt.cluster_centers_

# Function to display three images
def show_img_compar(img_1, img_2, img_3):
    f, ax = plt.subplots(1, 3, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    ax[0].axis('off') 
    ax[1].axis('off')
    ax[2].axis('off')
    f.tight_layout()
    plt.show()

def show_img_compar_i(imgs):
    f, ax = plt.subplots(1, len(imgs), figsize=(10,10))
    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)
        ax[i].axis('off')
    f.tight_layout()
    plt.show()

# Function which finds the most common colour in the image and returns its HSV value
def palette_perc(k_cluster):
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) 
    curMax = 0
    index = 0
    for i in counter:
        fraction = np.round(counter[i]/n_pixels, 2)
        if fraction > curMax:
            curMax = fraction
            index = i

    hsv_vals = k_cluster.cluster_centers_[index]
    
    return hsv_vals

clt_1 = clt.fit(cropped_img.reshape(-1, 3))

hsv = palette_perc(clt_1)
print (hsv)
# Set upper and lower bounds from the calculated HSV value
lower_bound = np.array([hsv[0] - 20, hsv[1] - 115, hsv[2] - 115]) 
upper_bound = np.array([hsv[0] + 20, hsv[1] + 115, hsv[2] + 115]) 

# Compute mask and remove unnecessary noise from it
mask = cv.inRange(hsv_img, lower_bound, upper_bound)
kernel = np.ones((7,7),np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
median = cv.medianBlur(mask, 5)

# Compute all the contours in the mask 
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# Extract the largest of these contours (i.e contour of table)
c = max(contours, key = cv.contourArea)
# Compute center of table and draw contour
M = cv.moments(c)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
# draw the contour and center of the shape on the image
src_img2 = cv.drawContours(np.zeros_like(src_img), [c], -1, (180, 255, 255), 1)
# Draw the largest contour 
#src_img2 = cv.drawContours(np.zeros_like(src_img), c, -1, (180, 255, 255), 1) 
# Use probabalistic Hough Transform to extract line segments from contour
src_img2 = cv.cvtColor(src_img2, cv.COLOR_BGR2GRAY)
#lines = cv.HoughLinesP(src_img2, 1, np.pi/180, 150, None, 400, 100) #minlinelength is dependent on pixels
lines = cv.HoughLines(src_img2, 1, np.pi / 180, 100, None, 0, 0)
lengths = []
# Draw lines on image
#for line in lines:
 #   x1, y1, x2, y2 = line[0]
  #  lengths.append(((x1 - x2)**2 + (y1 - y2)**2)**0.5)
   # cv.line(src_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

strong_lines = np.zeros([4,1,2])
n2 = 0
for n1 in range(0,len(lines)):
    for rho,theta in lines[n1]:
        if n1 == 0:
            strong_lines[n2] = lines[n1]
            n2 = n2 + 1
        else:
            if rho < 0:
               rho*=-1
               theta-=np.pi
            closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10)
            closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/1)
            closeness = np.all([closeness_rho,closeness_theta],axis=0)
            if not any(closeness) and n2 < 4:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1

print("h", strong_lines)

if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(src_img, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

#print(lengths)
print(lines)

# If more than 4 lines calculated, use Kmeans to cluster into 4 clusters then calcultae intersection
def segment_by_angle_kmeans(lines, k=4, **kwargs):
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections



table_center = cv.circle(src_img2, (cX, cY), 7, (255, 255, 255), -1)
segmented = segment_by_angle_kmeans(strong_lines)
intersections = segmented_intersections(segmented)
for i in intersections:
    cv.circle(src_img, (i[0][0], i[0][1]), 10, (255, 255, 255), -1)
#print(intersections)

hMap = {}

for i in intersections:
    x = i[0][0]
    y = i[0][1]
    if x < 0 or y < 0 : continue    
    dist = math.sqrt( ((y-cY)**2) + ((x-cX)**2) )
    if dist not in hMap:
        hMap[dist] = []
    hMap[dist].append([x,y])

v = sorted(list(hMap.keys()))[:4]
p = []
for d in v:
    p += hMap[d]
#print(hMap)
#print("points",p)
for i in p:
    cv.circle(src_img, (i[0], i[1]), 3, (255, 0, 255), -1)

h = int(2000)
w = int(h/2)
src_pts =  np.array(p, dtype=np.float32)
dst_pts = np.array([[0, 0],   [h, 0],  [h, w], [0, w]], dtype=np.float32)
M = cv.getPerspectiveTransform(src_pts, dst_pts)
warp = cv.warpPerspective(src_img, M, (h, w))


cv.circle(src_img, (cX, cY), 30, (255, 255, 255), -1)

show_img_compar_i([mask, src_img2, src_img])

