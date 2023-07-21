# Turkish-Bean-Variety-Classification-

# Problem Statement
Develop a machine learning model that can accurately classify the seven most well-known types of dry beans in Turkey (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira) based solely on the dimension and shape features of the beans. The goal is to create a model that can differentiate between the different types of beans using only internal characteristics, such as form, shape, type, and structure, without relying on any external discriminatory features. The project aims to find the best machine learning algorithm that can effectively classify the beans based on these specific features.

# Data Set Information

Seven different types of dry beans were used in this project, taking into account the features such as form, shape, type, and structure by the market situation. Use best machine learning algorithm to classify the most well-known 7 types of beans in Turkey; Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira, depending only on dimension and shape features of bean varieties with no external discriminatory features.

# Features Information

|Feature | Detail|
|----------|-------|
| Area (A)| The area of a bean zone and the number of pixels within its boundaries.|
| Perimeter (P)| Bean circumference is defined as the length of its border.|
| Major axis length (L)| The distance between the ends of the longest line that can be drawn from a bean.|
|Minor axis length (l)| The longest line that can be drawn from the bean while standing perpendicular to the main axis.|
| Aspect ratio (K)| Defines the relationship between L and l.|
| Eccentricity (Ec)| Eccentricity of the ellipse having the same moments as the region.|
| Convex area (C)| Number of pixels in the smallest convex polygon that can contain the area of a bean seed.|
|Equivalent diameter (Ed)|The diameter of a circle having the same area as a bean seed area.|
| Extent (Ex)| The ratio of the pixels in the bounding box to the bean area.|
|Solidity (S)| Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.|
| Roundness (R)| Calculated with the following formula: (4piA)/(P^2)|
|Compactness (CO)| Measures the roundness of an object: Ed/L|
| ShapeFactor1| (SF1)|
|ShapeFactor2 |(SF2)|
|ShapeFactor3| (SF3)|
|ShapeFactor4| (SF4)|
|Class| (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)|

# Project Report

### `EDA`

* The dataset consists of 13,611 rows and 17 columns. The column names are: ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'Roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class'].


* All variables in the dataset have the correct data types assigned to them.


* The dataset does not contain any null values, ensuring that all data is complete.


* Several pairs of variables in the dataset exhibit high correlation.


* After identifying and removing 68 duplicate rows, approximately 0.49% of the data was eliminated, resulting in a cleaner dataset.


* A thorough check of the unique values in each variable was performed to identify any suspicious values or typos. Fortunately, no such values were found in the dataset.


* After confirming the cleanliness of the data, a copy of the dataset was created to preserve the original state.


* Outliers were detected using the IQR method, and it was observed that the columns 'Area' and 'ConvexArea' contained outliers.


* Since removing these outliers would result in a significant loss of data, an alternative approach was used to handle the outliers. The Z-score method was applied to treat the outliers in the dataset.


* After removing the outliers, approximately 8.34% of the data was lost, resulting in a dataset with a reduced number of extreme values.

----------------------------

### `FEATURE ENGINEERING`


* I began by performing feature encoding and found that the target variable "Class" was the only variable in categorical form. I replaced the categories using the df.replace() function and mapped them as follows: 'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'HOROZ': 4, 'SIRA': 5, 'DERMASON': 6.


* Next, I examined the distribution of the data and observed the following results for variables with skewness values:

**Approximately Symmetric Variables (skewness between -0.5 and 0.5):**

MajorAxisLength (0.519943)

ShapeFactor1 (0.073565)

ShapeFactor3 (0.152033)

**Moderately Skewed Variables (skewness between 0.5 and 1 or -0.5 and -1):**

Perimeter (0.602878)

AspectRatio (0.626851)

Eccentricity (-0.834302)

EquivDiameter (0.611550)

Compactness (-0.038839)

ShapeFactor2 (0.218911)


**Highly Skewed Variables (skewness greater than 1 or less than -1):**


Area (0.940250)

MinorAxisLength (0.663292)

ConvexArea (0.942741)

Extent (-0.743348)

Solidity (-1.164965)

Roundness (-0.495015)

ShapeFactor4 (-1.456584)

Class (-0.615489)

Most of the variables are not normally distributed, but since it is a classification problem, no transformation is necessary.

* I also investigated the correlation between variables and identified 21 pairs of highly correlated variables. Based on the lowest correlation with the target variable ("Class"), I decided to drop the following variables:

['Perimeter', 'MajorAxisLength', 'EquivDiameter', 'AspectRation', 'Area', 'Compactness', 'ShapeFactor3', 'ConvexArea', 'MinorAxisLength']

After dropping these variables, I was left with a reduced dataset containing 8 variables, including the target variable.


* After dropping these variables, I divided the data into a training set and a test set in a 70:30 ratio.


* Next, I checked for the need for feature scaling and determined that none of the features required scaling.


* Subsequently, I examined the class imbalance in the dataset and found a significant imbalance.


* To address the class imbalance, I applied oversampling techniques using SMOTE and ADASYN. I opted not to use undersampling techniques as they would result in the loss of a significant amount of data.


* I evaluated the performance of SMOTE and ADASYN by building a base model using random forest and analyzing various evaluation metrics on both the training and test sets. Based on the results, I found that SMOTE performed better than ADASYN, so I proceeded with SMOTE for further analysis.


* I then performed feature selection using the random forest feature importance method and discovered that all features were important.


* Following that, I assessed the Variance Inflation Factor (VIF) and observed that all variables had high VIF values. Considering that dropping variables individually would leave no variables for model building, I decided to retain all the variables and proceeded with model building. If the model shows signs of overfitting or underfitting, I will utilize PCA to address multicollinearity.
-------------------------------

### `MODEL BUILDING`

* I have tried multiple algorithms such as Logistic Regression, Decision Tree Classifier, KNN Classifier, Support Vector Classifier (SVC), Gaussian Naive Bayes, Ada Boost Classifier, Gradient Boosting Classifier, RandomForestClassifier, and XGBClassifier on the training data. The results are as follows:


**Training Data Result**


|Algorithm|	Accuracy|	Precision|	Recall	|F1-score|	Balanced Accuracy|	Building Time (s)|
|-----------|--------|------------|---------|-----------|----------------|-----------------|
|	LogisticRegression|	0.728611	|0.719692	|0.728611	|0.720430|	0.728611|	4.305557|
|	DecisionTreeClassifier|	1.000000|	1.000000	|1.000000	|1.000000|	1.000000|	0.203090|
|	KNeighborsClassifier|	0.898065|	0.896117	|0.898065|	0.895951|	0.898065|	0.561598|
|	GaussianNB|	0.920952|	0.921782	|0.920952|	0.921115|	0.920952	|0.015581|
|	AdaBoostClassifier|	0.834541|	0.852369	|0.834541	|0.826781|	0.834541|	1.337142|
|	GradientBoostingClassifier|	0.96924|	0.969473|	0.969294	|0.969365|	0.969294|	29.370274|
|	RandomForestClassifier|	1.000000|	1.000000|	1.000000|	1.000000	|1.000000	|0.749819|
|	XGBClassifier|	0.999829	|0.999829	|0.999829|	0.999829	|0.999829	|5.498693|
|	SVC	|0.741510|	0.745399|	0.741510|	0.740249|	0.741510	|32.688252|


* On the training set, Random Forest, XG Boost, and Gradient Boost have the highest accuracy and balanced accuracy. However, Random Forest appears to overfit the data.


* Next, I evaluated the algorithms on the test set, and the results are as follows:


**Test Data Result**


|Algorithm|	Accuracy|	Precision|	Recall|	F1-score|	Balanced Accuracy|
|---------|---------|-------------|-----|---------|-----------------------|
|	LogisticRegression|	0.689128	|0.731225	|0.689128|	0.702422|	0.618724|
|	DecisionTreeClassifier|	0.881342	|0.881103|	0.881342	|0.881016|	0.905213|
|	KNeighborsClassifier|	0.716779	|0.734833|	0.716779	|0.722647|	0.786315|
|	GaussianNB|	0.904966|	0.905253	|0.904966	|0.904841	|0.926015|
|	AdaBoostClassifier|	0.808591|	0.819106	|0.808591	|0.799033	|0.817510|
|	GradientBoostingClassifier|	0.917584	|0.917881	|0.917584|	0.917657|	0.935755|
|	RandomForestClassifier|	0.922416|	0.922890|	0.922416|	0.922543|	0.940651|
|	XGBClassifier|	0.914631|	0.915150|	0.914631|	0.914811	|0.932512|
|	SVC	|0.693154|	0.750443|	0.693154|	0.717590|	0.765858|


* On the test set, Random Forest, Gradient Boost, and XG Boost have the highest accuracy and balanced accuracy.


* Based on these results, I choose XG Boost for further analysis as it does not overfit and performs better on both the training and test data.


* I performed hyperparameter tuning, accuracy is not incresed. Therefore, I select the XG Boost model without hyperparameter tuning.


* Next, I applied cross-validation to increase accuracy. The cross-validation scores (accuracy) are as follows:

[0.9532097, 0.94977169, 0.95861872, 0.96803653, 0.97203196]. 

The mean accuracy score is 0.9603337209075098.


* Then, I evaluated the XG Boost model on the train and test sets, and the results are as follows:


|Dataset|	Accuracy|	Balanced Accuracy|	Precision|	Recall	|F1-score|
|-------|-----------|----------------|---------------|------------|--------|
|Train|	0.999829|	0.999829|	0.999829	|0.999829	|0.999829|
|	Test|	0.914631|	0.932512|	0.861429|	0.932512|	0.885005|


* Furthermore, I checked the confusion matrix for both the train and test sets:


**Confusion Matrix of Train Set**


|SEKER|	BARBUNYA|	BOMBAY|	CALI	|HOROZ|	SIRA	|DERMASON|
|------|-------|------------|--------|------|-------|-----------|
|SEKER	|14.285714|	0.000000	|0.000000|	0.000000|	0.000000|	0.000000|	0.000000|
|BARBUNYA|	0.000000|	14.285714|	0.000000|	0.000000|	0.000000|	0.000000	|0.000000|
|BOMBAY|	0.000000|	0.000000|	14.285714|	0.000000|	0.000000	|0.000000|	0.000000|
|CALI	|0.000000|	0.000000	|0.000000|	14.285714	|0.000000	|0.000000|	0.000000|
|HOROZ|	0.000000|	0.000000|	0.000000	|0.000000	|14.285714|	0.000000|	0.000000|
|SIRA	|0.000000	|0.000000	|0.000000|	0.000000	|0.000000	|14.280007|	0.005707|
|DERMASON|	0.000000|	0.000000	|0.000000	|0.000000|	0.000000|	0.011415|	14.274299|


**Confusion Matrix of Test Set**


|SEKER|	BARBUNYA	|BOMBAY|	CALI	|HOROZ|	SIRA	|DERMASON|
|--------|----------|--------|---------|-------|-------|----------|
|SEKER|	13.986577|	0.214765	|0.000000	|0.000000|	0.000000	|0.402685|	0.402685|
|BARBUNYA|	0.026846|	9.798658|	0.026846	|0.375839|	0.053691|	0.107383|	0.000000|
|BOMBAY|	0.000000	|0.000000	|0.026846|	0.000000|	0.000000	|0.000000|	0.000000|
|CALI	|0.026846|	0.563758|	0.000000	|12.000000|	0.107383	|0.080537|	0.000000|
|HOROZ|	0.000000|	0.053691|	0.000000|	0.268456|	12.617450|	0.375839	|0.134228|
|SIRA	|0.268456|	0.053691|	0.000000|	0.107383|	0.375839|	18.281879|	1.959732|
|DERMASON|	0.402685|	0.000000	|0.000000|	0.000000|	0.026846|	2.120805|	24.751678|


* I also checked the ROC-AUC curve and found that the mean AUC on the training set is 1.0, indicating excellent performance. The mean AUC on the test set is 0.9957, which is also very high.


* Finally, I analyzed the important features in the model, and the top features based on their importance are as follows:


|Feature|	Importance|
|-----|-----------|
|	ShapeFactor1|	0.339255|
|	Eccentricity|	0.205589|
|	ShapeFactor2	|0.176098|
|	roundness|	0.138000|
|	ShapeFactor4|	0.071907|
|	Solidity|	0.046856|
|	Extent|	0.022296|


* These features were used to build the model, with "ShapeFactor1" and "Eccentricity" being the most important, accounting for approximately 53% of the information gain.


* Furthermore, I found that many features are correlated, and even after removing some features, multi-collinearity between variables remained. Therefore, I decided to apply Principal Component Analysis (PCA) to remove colinearity and multi-collinearity.


* Additionally, I applied standard scaling to all variables, as PCA assumes that the features are normally distributed with a mean of 0 and a standard deviation of 1. This step ensures that PCA can effectively reduce the dimensionality of the data.During PCA, I found that only three principal components (PCs) capture 100% of the variance in the data.During PCA, I found that only three principal components (PCs) capture 100% of the variance in the data.


* During PCA, I found that only three principal components (PCs) capture 100% of the variance in the data.


* Next, I applied various algorithms to these three principal components, and the results are as follows:


**PCA result on the Training Set**


|Algorithm	|Accuracy|	Precision|	Recall|	F1-score|	Balanced Accuracy|	Building Time (s)|
|------------|--------|-------|---------|---------|---------------------|----------------------|
|	LogisticRegression|	0.649449	|0.682035	|0.649449	|0.612610|	0.649449|	0.971483|
|	DecisionTreeClassifier	|1.000000|	1.000000|	1.000000	|1.000000	|1.000000|	0.103513|
|	KNeighborsClassifier|	0.881913|	0.882672	|0.881913|	0.882019|	0.881913	|0.508087|
|	GaussianNB|	0.775070	|0.776465	|0.775070	|0.773509|	0.775070|	0.031244|
|	AdaBoostClassifier|	0.525370|	0.370476	|0.525370|	0.404165|	0.525370	|0.917904|
|	GradientBoostingClassifier|	0.895154	|0.895359	|0.895154|	0.895170|	0.895154	|16.007691|
|	RandomForestClassifier|	1.000000	|1.000000	|1.000000	|1.000000|	1.000000	|0.688190|
|	XGBClassifier|	0.973346|	0.973454|	0.973346|	0.973342|	0.973346|	6.020092|
|	SVC	0.636779|	0.635077|	0.636779|	0.628233|	0.636779|	30.236159|


* Random Forest Classifier overfits the data, while XG Boost has the highest accuracy on the training set.


**PCA result of the Test Set**


|Algorithm	|Accuracy|	Precision	|Recall|	F1-score	|Balanced Accuracy|
|---------|---------|-----------|------------|-----------|--------------|
|	LogisticRegression|	0.587651	|0.666476|	0.587651	|0.546386|	0.639146|
|	DecisionTreeClassifier|	0.783893|	0.788083|	0.783893|	0.785656	|0.798545|
|	KNeighborsClassifier|	0.671678|	0.678754|	0.671678|	0.674489	|0.698427|
|	GaussianNB|	0.736107|	0.745607	|0.736107	|0.738061|	0.761558|
|	AdaBoostClassifier|	0.430872	|0.240083	|0.430872	|0.294259|	0.524197|
|	GradientBoostingClassifier|	0.843490|	0.847301	|0.843490|	0.844923|	0.853445|
|	RandomForestClassifier|	0.846443|	0.848964	|0.846443|	0.847434|	0.855234|
|	XGBClassifier|	0.834899	|0.837789	|0.834899|	0.836094	|0.845882|
|	SVC	0.617181|	0.629000|	0.617181	|0.617466|	0.642170|


* Random Forest Classifier, Gradient Boosting Classifier, and XG Boost Classifier have the highest accuracy on the test set.


* Based on these results, I decided to choose the XG Boost model without PCA because it provides better result than all models with PCA.


* Lastly, I saved the XG Boost model without PCA for future use.
