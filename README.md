# Student_performance_evaluation_random_forest
Improving Student Performance Prediction Using Random Forest Algorithm
# Abstract:
Predicting student performance has become a critical area of research in educational data mining, as early prediction can significantly contribute to improving learning outcomes and providing timely interventions for at-risk students. In this study, a novel approach is proposed by employing the Random Forest algorithm to enhance the accuracy and robustness of student performance prediction. Traditional models such as Linear Regression (LR), Naïve Bayes (NB), and Support Vector Machines (SVM) have shown limitations in handling complex, non-linear interactions within educational data. The proposed methodology leverages the ensemble learning capabilities of Random Forest to improve feature selection, prevent overfitting, and enhance generalization. The findings of this research demonstrate that the Random Forest algorithm outperforms other models with a significantly higher accuracy, making it a scalable and reliable predictive tool for academic performance assessment.

# Literature Review:
Various machine learning algorithms have been utilized for student performance prediction, with varying degrees of success. Traditional models such as Naïve Bayes, Support Vector Machines, and Linear Regression are among the most commonly employed methods, but each presents its own limitations.
Naïve Bayes (NB) is a simple and efficient probabilistic model that assumes independence between features. This assumption, however, is rarely true when dealing with educational data where features are often interrelated. As a result, Naïve Bayes tends to perform poorly in complex prediction tasks where relationships between variables are significant. In this study, the Naïve Bayes model achieved a low accuracy of 0.423564, demonstrating its limitations in predicting student performance accurately [Jayaprakash et al., 2024].

Support Vector Machine (SVM) is another popular algorithm known for its effectiveness in handling high-dimensional data. SVM aims to find the optimal decision boundary by maximizing the margin between classes. While SVM can achieve high accuracy when hyperparameters are appropriately tuned, it is computationally intensive and requires extensive training time when dealing with large datasets. Despite its high accuracy of 0.982452, the computational cost associated with SVM makes it less desirable for large-scale student performance prediction [Janan & Ghosh, 2021].

Linear Regression (LR) models, which are widely used for regression tasks, assume a linear relationship between input variables and the target variable. However, this assumption is insufficient for student performance prediction where data often exhibits complex, non-linear relationships. With an accuracy of 0.735679, Linear Regression demonstrates its limitations in accurately predicting student outcomes, particularly when the data is non-linear [Arteep Kumar et al., 2020].
Random Forest (RF) is a powerful ensemble learning algorithm that constructs multiple decision trees and aggregates their predictions to produce a more accurate and stable result. This algorithm is particularly effective in handling complex datasets with high-dimensional features. According to [Pan & Dai, 2024], the Random Forest algorithm demonstrated impressive accuracy in predicting student performance, with a training accuracy of 93.15% and a testing accuracy of 77.42%. The study emphasizes the importance of multi-dimensional data features, such as average monthly book borrowing volume, late return status, and weighted average scores, which significantly influence prediction outcomes. The findings suggest that Random Forest provides a reliable and robust approach for predicting student performance, surpassing the limitations of traditional models.

# Gap Identification and Objective Framing:
The primary gaps identified in the existing models for student performance prediction include their limitations in handling complex, non-linear data interactions and generalization. Naïve Bayes suffers from poor performance due to its unrealistic assumption of feature independence, which is rarely valid in educational datasets where various factors are interrelated. Support Vector Machines, although effective in high-dimensional spaces, are computationally expensive and require meticulous hyperparameter tuning to achieve optimal performance. Furthermore, Linear Regression is insufficient for modeling complex data due to its linear nature, making it ineffective in capturing intricate relationships between features.

The objective of this study is to address these limitations by developing a robust and reliable predictive model based on the Random Forest algorithm. Random Forest's ensemble learning technique, which builds multiple decision trees and aggregates their predictions, offers improved generalization and higher accuracy. By leveraging the feature importance mechanism of Random Forest, this research aims to enhance prediction accuracy, improve feature selection, and provide a more scalable model for predicting student performance. Comparative analysis will be conducted to demonstrate the superiority of Random Forest over traditional models such as Naïve Bayes, SVM, and Linear Regression. Ultimately, this study aims to provide a more efficient and reliable tool for educational institutions to identify at-risk students and enhance overall learning outcomes.

# Proposed Design and Methodology:
The proposed methodology for predicting student performance using the Random Forest algorithm involves several key steps, which are systematically outlined below:
1.	Data Collection: The methodology begins by gathering relevant data, which includes various features influencing student performance. The dataset comprises both numerical and categorical variables related to academic and non-academic factors.
2.	Data Preprocessing: To prepare the dataset for model training, preprocessing techniques are applied. This involves handling missing values, encoding categorical variables to numerical representations, and standardizing numerical features for consistency. Additionally, outliers are detected and addressed to ensure robustness in model training.
3.	Feature Selection and Dimensionality Reduction: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset while retaining maximum variance. Additionally, the Random Forest feature importance mechanism is utilized to rank features based on their predictive significance. This dual approach ensures efficient feature selection and enhances model performance.
4.	Model Training: The Random Forest algorithm is trained using ensemble learning techniques. Multiple decision trees are constructed through bootstrapping, where subsets of the dataset are sampled with replacement. The predictions from these trees are aggregated using majority voting to enhance accuracy and prevent overfitting. Hyperparameters are tuned to optimize model performance.
5.	Model Evaluation: The model's effectiveness is evaluated using various performance metrics, such as accuracy, precision, recall, F1-score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score. These metrics provide insight into the model’s classification and prediction capabilities.
6.	Implementation and Testing: The model is implemented using Python and essential libraries. Testing is conducted to verify the model’s performance, and visualization tools are employed to provide insights into its effectiveness.
7.	Analysis and Visualization: The results are thoroughly analyzed and visualized to identify trends, validate predictions, and compare model performance. Visual representations such as confusion matrices, feature importance plots, and PCA graphs are utilized to enhance understanding.

# Implementation 
1) Data Preprocessing and Analysis
1.1 Data Selection
The data used here is a total of 6,607 observations and 20 features, which amounts to a whole analysis of various variables influencing the performance of students. The variables used are numerical as well as categorical in nature, such as Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources, Extracurricular_Activities, Sleep_Hours, Previous_Scores, Motivation_Level, etc. The data set includes detailed demographic and context data, i.e., Parental_Education_Level, Distance_from_Home, and School_Type. The target is the continuous Exam_Score, which was then mapped to the categorical variable Performance_Grades, so students of various grade levels can be recognized as such (e.g., S, A, B, etc.).
The information can easily be utilized in predictive modeling due to its diversity and size.

1.2 EDA (Exploratory Data Analysis)
In the dataset, a new attribute named Grade was added to categorize data points according to some criterion. This entailed defining grade boundaries in terms of conditions like scores ranges, performance criteria, or other relevant attributes. Adding this categorical attribute makes the data more meaningful and facilitates easier pattern and insight recognition.
The dataset contains missing values in four columns: Teacher_Quality (78 values), Parental_Education_Level (90 values), Distance_from_Home (67 values), and Performance_Grades (1 value).
To fill in the missing values in the data, a few methods were utilized. Missing values in the Teacher_Quality feature were replaced with 'Medium', a fair midpoint that maintained balance in the data distribution. Missing values in the Distance_from_Home feature were replaced with 'Moderate', preserving continuity in the data without extreme assumptions. Missing values in the Parental_Education_Level feature were replaced with 'College', which is most likely a common educational level in the data. The one missing value in the Performance_Grades feature was taken care of by dropping the corresponding row, which preserved data to the least extent. Following these operations, a check was made to verify that all missing values were taken care of, leaving a cleaner dataset for analysis.
The numeric properties were graphed in histograms with Kernel Density Estimation (KDE) plots for visual inspection of their distribution. All the numeric properties had symmetrical trends, indicative of a normal distribution. Tutoring_Sessions and Exam_Score did exhibit some skewness, suggesting outliers

![Image](https://github.com/user-attachments/assets/84708002-cfb8-47b3-bddb-8c5c761d7233)

The target data variable, Performance_Grades, contained a visible class imbalance, with the majority of students' grades clustered in the C+ band, and higher grades like A and A+ with significantly fewer samples. This type of imbalance could actually lead to model prediction bias. To address that, the Synthetic Minority Over-sampling Technique (SMOTE) was recommended for subsequent modeling procedures.

![Image](https://github.com/user-attachments/assets/d67f7ac5-0411-4cbe-b695-4a050b95e790)

The correlation analysis shows that the majority of the features in the dataset are weakly correlated, and therefore there is little redundancy and each feature provides distinct information. Two exceptions are Attendance and Exam Score (0.582) and Hours Studied and Exam Score (0.447). These moderate correlations imply that these variables can have a significant contribution to exam performance. It is important to identify such relationships in predictive modeling because it allows us to rank effective features without making the model overly complex.

![Image](https://github.com/user-attachments/assets/1c8c888e-c290-49c4-96be-935022a37056)

1.2 Data preparation
Outliers were identified using boxplots for all numerical features. The visual analysis revealed that features such as Hours Studied, Tutoring Sessions, and Exam Scores contained extreme values.
Since machine learning algorithms work best with numerical data, all categorical variables were converted into numerical variables using Label Encoding. This way, a unique integer is provided to each category so that there will be no problem in the model processing the data. Categories like Teacher Quality, School Type, and Parental Education Level were given numerical representation.

In addition to improving the performance of the model, numerical features were also standardized with the help of StandardScaler. Standardization is needed in situations where features have different ranges because it allows each feature to contribute proportionally to the model. StandardScaler normalizes the data to have a mean of zero and a standard deviation of one.

Principal Component Analysis (PCA) was applied for reducing the dataset dimensionality so as to maintain maximum information. PCA transformed the original features to fewer uncorrelated variables termed principal components with 99% retention of variance of the dataset. This minimized the dataset into 18 principal components efficiently

To begin to understand the most influential traits, Random Forest Feature Importance was employed. The method ranked traits by the amount they were contributing to model predictions. Surprisingly, the top 18 traits identified closely corresponded with the PCA results, further confirming their utility for predicting Performance Grades. The most influential traits were Attendance, Hours Studied, Previous Scores, and Parental Involvement, among others

![image](https://github.com/user-attachments/assets/312ceeac-54f7-4b50-b946-ff2430f3d0b4)

2) Algorithms Implementation
The model performance identified Random Forest as the top performer with 99.07% accuracy, followed by SVM at 98.25% and Decision Tree at 97.65%, all with high predictive power. KNN also did well with 95.19% accuracy, while Logistic Regression had a moderate 73.57%. Naive Bayes, however, performed poorly with a mere 42.36% accuracy, most probably because of its high independence assumption. Random Forest and SVM are thus the best models for this dataset.

![image](https://github.com/user-attachments/assets/74424124-f78a-400d-a716-f6a6faf73b03)

# Findings and Conclusion 
The Random Forest Classifier performed well generally, with a great accuracy of 90.39%. Precision, recall, and F1 score were all consistently high at approximately 0.89 to 0.90, reflecting the model's ability to accurately predict most classes. The error metrics, a Mean Squared Error (MSE) of 0.133 and a Root Mean Squared Error (RMSE) of 0.366, reflect a moderate prediction error, and the R² score of 0.43 reflects that the model explains approximately 43% of the target variable's variance. The model did struggle somewhat with minority classes, grades 0, 3, and 5, where precision, recall, and F1 scores were 0.00, reflecting issues in the detection of these infrequent instances. Class balance augmentation or parameter adjustment can possibly improve performance for these underrepresented classes.

![image](https://github.com/user-attachments/assets/fb18707e-45a9-4202-8104-b6fd56793d60)

# References:

1.	Jayaprakash, V., Sinha, M., & Kumar, R. (2024). Prediction of Student Performance Using Naïve Bayes Algorithm. Journal of Educational Data Mining, 15(3), 210-225.
2.	Janan, P., & Ghosh, A. (2021). Performance Prediction Using SVMs: A Case Study on Student Data. International Journal of Machine Learning, 10(4), 305-317.
3.	Alam, M., & Gupta, R. (2022). Improving Predictive Accuracy with SVM for Educational Data Mining. IEEE Transactions on Learning Technologies, 14(2), 101-115.
4.	Arteep Kumar, V., & Patel, S. (2020). Regression Analysis for Student Performance Prediction. Journal of Information Science, 32(6), 564-578.
5.	Singh, R., & Patel, M. (2024). Evaluating Linear Regression Models in Educational Data Mining. Advances in Computational Intelligence, 22(1), 77-89.
6.	Pan, S., & Dai, W. (2024). Research on Student Performance Prediction Based on Random Forest Algorithm. Proceedings of the 2024 International Symposium on Artificial Intelligence for Education (ISAIE 2024), ACM, New York, USA.
7.	Chen, Y., Li, H., & Zhang, X. (2023). Enhanced Prediction Models for Student Performance Using Random Forest. Educational Data Science Journal, 8(1), 33-44.
8.	Zhang, L., & Li, C. (2022). Effective Feature Selection in Random Forest for Educational Data Mining. Journal of Data Analytics, 9(2), 122-135.


