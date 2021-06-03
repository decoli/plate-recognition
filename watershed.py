import cv2
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

# img = cv2.imread('debug/samples/water_coins.jpg')
img = cv2.imread('samples/plate_0021.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('debug/output/gray.png', gray)

# Otsu法で画像を二値化する
# thresh,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh, bin_img = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('debug/output/bin.png', bin_img)

# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = 2)
# 去掉图像黑点
kernel = np.ones((9,9),np.uint8)
opening = cv2.morphologyEx(bin_img,cv2.MORPH_CLOSE,kernel,iterations = 4)
cv2.imwrite('debug/output/opening.png', opening)

sure_bg = cv2.dilate(opening,kernel,iterations=2)
cv2.imwrite('debug/output/sure_bg.png', sure_bg)

ignored_edge_rate = 0.2
ignored_edge = np.zeros_like(opening)
ignored_edge_w = int(ignored_edge.shape[0] * ignored_edge_rate)
ignored_edge_h = int(ignored_edge.shape[1] * ignored_edge_rate)
ignored_edge[ignored_edge_w: ignored_edge.shape[0] - ignored_edge_w, ignored_edge_h: ignored_edge.shape[1] - ignored_edge_h] += 1

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform,cmap='jet')
plt.savefig('debug/output/dist_transform.png')
plt.close()

opening = opening * ignored_edge
cv2.imwrite('debug/output/opening_ignored.png', opening)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform,cmap='jet')
plt.savefig('debug/output/dist_transform_ignored.png')
plt.close()

ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
cv2.imwrite('debug/output/sure_fg.png', sure_fg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imwrite('debug/output/unknown.png', unknown)

ret, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers, cmap='jet')
plt.savefig('debug/output/markers.png')
plt.close()

markers = markers + 1
markers[unknown==255] = 0
plt.imshow(markers, cmap='jet')
plt.savefig('debug/output/markers_unknown.png')
plt.close()

# unknown -> 0
# background -> 1
# foreground -> 2~25のint

markers = cv2.watershed(img,markers)
plt.imshow(markers, cmap='jet')
plt.savefig('debug/output/markers_watershed.png')
plt.close()

# 境界領域 -> -1
# background -> 1
# foreground -> 2~25のint

img[markers == -1] = [255,0,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('debug/output/markers_th.png')
plt.close()

print('demo')
