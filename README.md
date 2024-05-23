# Overview  
This project is part of **Data Mining & Machine Learning** elective course in Computer Engineering & Informatics Department of University of Patras for Spring Semester 2024 (Semester 8). 
* **Dataset**: [https://archive.ics.uci.edu/dataset/779/harth](URL) : The Human Activity Recognition Trondheim (HARTH) dataset contains recordings of 22 participants wearing two 3-axial Axivity AX3 accelerometers for around 2 hours in a free-living setting. One sensor was attached to the right front thigh and the other to the lower back. The provided sampling rate is 50Hz. 

    Each subject's recordings are provided in a separate .csv file. One such .csv file contains the following columns:
    1. **timestamp**: date and time of recorded sample
    2. **back_x**: acceleration of back sensor in x-direction (down) in the unit g
    3. **back_y**: acceleration of back sensor in y-direction (left) in the unit g
    4. **back_z**: acceleration of back sensor in z-direction (forward) in the unit g
    5. **thigh_x**: acceleration of thigh sensor in x-direction (down) in the unit g
    6. **thigh_y**: acceleration of thigh sensor in y-direction (right) in the unit g
    7. **thigh_z**: acceleration of thigh sensor in z-direction (backward) in the unit g
    8. **label**: annotated activity code

    The dataset contains the following annotated activities with the corresponding coding:
    1: **walking**	
    2: **running**	
    3: **shuffling**
    4: **stairs (ascending)**
    5: **stairs (descending)**	
    6: **standing**	
    7: **sitting**	
    8: **lying**	
    13: **cycling (sit)**	
    14: **cycling (stand)**	
    130: **cycling (sit, inactive)**
    140: **cycling (stand, inactive)**

* **Goals**:
1. Statistic study of the features of the dataset. Our main goal is to reveal probable "hidden" dependancies between the data.

![Στιγμιότυπο οθόνης 2024-05-23 180454](https://github.com/miltiadiss/Data-Mining/assets/45690339/4ee72766-c13a-4788-88ad-333bbdfceba7) 

2. Training of 3 types of classifiers that predict the activity of every participant for every timestamp of the testing set based on the N previous values (lags) of the accelometers. It is also noted that the original dataset is split into training set (80% of the original dataset) and testing set (20% of the original dataset). Our main goal is to make a close approximation for every case, not to make perfectly accurate predictions.
    1. **Neural Networks**: We chose to work with an **Artificial Neural Network** that is trained for 50 epochs, using the *tensorflow* library and *keras*. For increased accuracy we can use a greater amount of epochs, however this increase in epochs would require a significantly greater amount of time. 
    2. **Bayesian Networks**:
    3. **Random Forests**: We chose to work with 30 decision trees (estimators) to create the **Random Forest**, using the *sklearn.ensemble* module. We can also use more estimators for increased accuracy like the case of the **Neural Network**.
3. We also used some performance metrics to evaluate the efficiency of the training of the models, such as: **Accuracy**, **F1 - score**, **Precision**, **Recall** e.t.c.
4. Splitting of the 22 participants into clusters based on their physical activity history using the following three clustering techniques:
    1. **k - means Algorithm**: To determnine the most suitable amount of clusters, we used the **Elbow Method with Squared Sum Error (SSE)**. Thus, we conlcuded that 3 clusters should be generated by the algorithm. For the execution of this algorithm we used the *sklearn.preprocessing* and *sklearn.cluster* modules.
    2. **Kohonen Network**:
    3. **Gaussian Mixture**:

* **Programming Language and Environment**: Python, Jupyter Notebook.
  
Further details of our implementations can be found in the technical report in the **Documentation** folder. 
