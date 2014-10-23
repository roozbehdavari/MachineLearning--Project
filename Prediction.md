---
title: "Predicting the Type of Physical Exercise"
author: "Roozbeh Davari"
output: html_document
---


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this analysis, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which the exercise is done.

## Data 

The training data for this study are available here: 

[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) which is a result of the following study:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


## Processing Data

First, we download and load the datasets.


```r
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv','training.csv','curl')
training <- read.csv('training.csv')
unkonw.activities <- read.csv('test.csv')
dim(training); 
```

```
## [1] 19622   160
```

The size of the training dataset is rather large. We explore the properties of different columns (i.e. potential predictors) and remove the columns with high fraction of missing values (i.e., have more than 90% missing values.) 


```r
# Counting the number of missing values
count.NAs <- sapply(training,function(x) sum(is.na(x)))

# Finding the columns that have more than 90% missing values
index.NAs <- c()
for (i in 1:length(count.NAs)) {
   if (count.NAs[[i]]/dim(training)[1] >= 0.9)
     {index.NAs <- append(index.NAs,i)}
  }

# Removing the variables with more than 95% missing values
training <- training[,-index.NAs]
dim(training)
```

```
## [1] 19622    93
```

With this approach, we have been able to remove 33 variables. There are still 127 potential predictors.

By using caret package, we look at the variability of each variable. We use nearZeroVar which diagnoses predictors that have one unique value (i.e. are zero variance predictors) or predictors that are have both of the following characteristics: they have very few unique values relative to the number of samples and the ratio of the frequency of the most common value to the frequency of the second most common value is large.

We eliminate the predictors with near zero variance. 


```r
library(caret)
near0 <- nearZeroVar(training,saveMetrics = T)
head(near0)
```

```
##                      freqRatio percentUnique zeroVar   nzv
## X                        1.000     100.00000   FALSE FALSE
## user_name                1.101       0.03058   FALSE FALSE
## raw_timestamp_part_1     1.000       4.26562   FALSE FALSE
## raw_timestamp_part_2     1.000      85.53155   FALSE FALSE
## cvtd_timestamp           1.001       0.10193   FALSE FALSE
## new_window              47.330       0.01019   FALSE  TRUE
```

```r
training <- training[,near0$nzv == FALSE]
dim(training); names(training)
```

```
## [1] 19622    59
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

Another 68 predictors are eliminated and the size of training set is more manageable with remaining 59 predictors. We can now look at individual variables and decide whether or not they potentially can add any useful information for building our prediction model. 

The observation ID , also user's name are not needed for building a prediction model and therefore we eliminate them. Furthermore, the provided information about the date are unlikely to contribute to our prediction as it seems that it merely indicates the date of performed observation. As a result, those are removed, too.


```r
training <- training[,-c(1,2,3,4,5)]
dim(training)
```

```
## [1] 19622    54
```


## Creating Training and Testing Subsets

We create a training and testing subset in order to build our prediction model. We pick 5,000 random observations for training and 1,500 random observations for cross-validation. 


```r
trainSub <- training[sample(nrow(training), 5000), ]
testSub <- training[sample(nrow(training), 1500), ]
```

## Training

We use two different methods for building our prediction models; random forests and boosted regression models. These methods are two of the most powerful and popular models for building prediction algorithm. We compare the results obtained from both methods to get a sense of the predictions robustness. 



```r
gbm.modelFit <- train(classe ~ .,method="gbm",data=trainSub,verbose=F)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
rf.modelFit <- train(classe ~ .,method="rf",data=trainSub,verbose=F)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


## Accuracy

We use confusion matrix to find the accuracy of our model fits.


```r
confusionMatrix(testSub$classe,predict(gbm.modelFit,testSub))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 417   2   0   0   0
##          B   7 292   6   0   0
##          C   0   3 276   0   0
##          D   0   0   1 250   1
##          E   0   1   0   3 241
## 
## Overall Statistics
##                                        
##                Accuracy : 0.984        
##                  95% CI : (0.976, 0.99)
##     No Information Rate : 0.283        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.98         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.983    0.980    0.975    0.988    0.996
## Specificity             0.998    0.989    0.998    0.998    0.997
## Pos Pred Value          0.995    0.957    0.989    0.992    0.984
## Neg Pred Value          0.994    0.995    0.994    0.998    0.999
## Prevalence              0.283    0.199    0.189    0.169    0.161
## Detection Rate          0.278    0.195    0.184    0.167    0.161
## Detection Prevalence    0.279    0.203    0.186    0.168    0.163
## Balanced Accuracy       0.991    0.985    0.986    0.993    0.996
```

```r
confusionMatrix(testSub$classe,predict(rf.modelFit,testSub))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 419   0   0   0   0
##          B   4 299   2   0   0
##          C   0   1 278   0   0
##          D   1   0   1 250   0
##          E   0   0   0   0 245
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.989, 0.997)
##     No Information Rate : 0.283         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.992         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.988    0.997    0.989    1.000    1.000
## Specificity             1.000    0.995    0.999    0.998    1.000
## Pos Pred Value          1.000    0.980    0.996    0.992    1.000
## Neg Pred Value          0.995    0.999    0.998    1.000    1.000
## Prevalence              0.283    0.200    0.187    0.167    0.163
## Detection Rate          0.279    0.199    0.185    0.167    0.163
## Detection Prevalence    0.279    0.203    0.186    0.168    0.163
## Balanced Accuracy       0.994    0.996    0.994    0.999    1.000
```

It can be seen that both methods have very high accuracy with random Forrest having slightly higher accuracy. 

## Prediction on the unknown sample

Finally, we predict the manner in which the exercise is done (i.e. 'classe' factor) for the a sample with unknown activities.



```r
predict(gbm.modelFit,unkonw.activities)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
predict(rf.modelFit,unkonw.activities)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

They return identical results which is reassuring.


