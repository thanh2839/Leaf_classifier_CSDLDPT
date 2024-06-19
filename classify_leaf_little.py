import numpy as np
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2


dataset = pd.read_csv("leaflittle.csv")

dataset.head(5)

type(dataset)

ds_path = "./images/Leaves"
img_files = os.listdir(ds_path)

breakpoints = [1060,1122,1123,1194,1268,1323,1386,1437,1438,1496,2051,2113,2424,2485,2547,2612,3511,3563,3566,3621]

print (len(breakpoints))

target_list = []
for file in img_files:
    target_num = int(file.split(".")[0])
    flag = 0
    i = 0 
    for i in range(0,len(breakpoints),2):
        if((target_num >= breakpoints[i]) and (target_num <= breakpoints[i+1])):
            flag = 1
            break
    if(flag==1):
        target = int((i/2))
        target_list.append(target)
print(target_list)

y = np.array(target_list)


X = dataset.iloc[:,1:]

X.head(5)

y[0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 142)

X_train.head(5)

y_train[0:5]

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X_train[0:2]

y_train[0:2]

clf = svm.SVC()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

metrics.accuracy_score(y_test, y_pred)

# print(metrics.classification_report(y_test, y_pred))

parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
               'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
             ]

svm_clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
svm_clf.fit(X_train, y_train)

svm_clf.best_params_

means = svm_clf.cv_results_['mean_test_score']
stds = svm_clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, svm_clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_pred_svm = svm_clf.predict(X_test)

metrics.accuracy_score(y_test, y_pred_svm)

pca = PCA()
pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)

# test_img_path = './leaf_linkoping_UNI/leaves test/2 - positive/1192.jpg'
# img = cv2.imread(test_img_path)

def feature_extract(img):
    names = ['area', 'Eccentridity', 'Extent', 'Contour Perimeter','Solidity', \
             'Equivalent Diameter', 'Perimeter Area Ratio']
    df = pd.DataFrame(columns=names)
    #Preprocessing
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    #shape features
    contour, image = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    M = cv2.moments(cnt)


    contour_area = cv2.contourArea(cnt)
#Eccentricity sau
    numerator = 4 * M['mu11']**2 + (M['mu20'] - M['mu02'])**2
    denominator = (M['mu20'] + M['mu02'])**2
    eccentricity = np.sqrt(numerator / denominator)
    # print(f'Eccentricity: {eccentricity}')

    #Extent
    # Tính hình chữ nhật bao quanh
    x, y, w, h = cv2.boundingRect(cnt)
    # Tính diện tích của hình chữ nhật bao quanh
    bounding_rect_area = w * h


    # Tính Extent
    extent = contour_area / bounding_rect_area
    # print(f'Object Area: {contour_area}')


    #Chu vi đường viền
    perimeter = cv2.arcLength(cnt,True)


    # "Solidity" là tỷ lệ của diện tích của đường viền đến diện tích của lồi lõm.
    hull = cv2.convexHull (cnt)
    convex_hull_area = cv2.contourArea(hull)
    solidity = contour_area / convex_hull_area
    # print(f'Solidity of contour : {solidity}')



    # equivalent_diameter
    equivalent_diameter = np.sqrt(contour_area * 4 / np.pi)
    # print(f'Equivalent Diameter of contour : {equivalent_diameter}')

    
    #Tỷ lệ chu vi-diện tích (Perimeter-area ratio) 
    perimeter_area_ratio = perimeter / contour_area
    vector = [contour_area, eccentricity, extent, perimeter, solidity, equivalent_diameter, perimeter_area_ratio]
    df_temp = pd.DataFrame([vector], columns= names)
    df = pd.concat([df, df_temp], ignore_index=True)

    return df
features_of_img = feature_extract()
features_of_img

scaled_features = sc_X.transform(features_of_img)
print(scaled_features)
# y_pred_mobile = svm_clf.predict(features_of_img)
y_pred_mobile = svm_clf.predict(scaled_features)
y_pred_mobile[0]