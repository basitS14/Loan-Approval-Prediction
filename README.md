## Loan Approval Prediction Project

This project focuses on building a machine learning model to predict loan approval status based on various applicant and loan-related features. The `LoanApproval.ipynb` Jupyter notebook covers data loading, extensive data cleaning and preprocessing, exploratory data analysis, and the final model training and export.

### Project Structure

* `LoanApproval.ipynb`: The main Jupyter notebook containing all the code for data processing, analysis, and model building.

### `LoanApproval.ipynb` - Detailed Breakdown

This notebook executes the following major steps:

1.  **Data Loading and Initial Inspection**
    * Loads the dataset from `'Copy of loan - loan.csv'`.
    * Performs initial checks on the dataset including displaying the head, information (`df.info()`), shape (`df.shape`), and checking for duplicate rows (`df.duplicated().sum()`).

2.  **Missing Value Analysis**
    * Identifies and quantifies missing values across different features, providing a percentage of missing data for each column.

3.  **Data Cleaning and Preprocessing**
    * **Target Variable Transformation**: Converts the `Loan_Status` column from categorical ('Y', 'N') to numerical (1 for 'Y', 0 for 'N') for model compatibility.
    * **Imputation of Missing Values**:
        * Missing categorical values in `Gender`, `Married`, `Dependents`, `Loan_Amount_Term`, and `Credit_History` are filled using the mode of their respective columns.
        * Missing values in `Self_Employed` are filled with "Unknown".
        * `LoanAmount` missing values are imputed using the median `LoanAmount` grouped by `Education` level.

4.  **Feature Separation**
    * Separates the dataset into numerical and categorical features for distinct processing and analysis.

5.  **Exploratory Data Analysis (EDA)**
    * Includes sections for comprehensive exploratory data analysis, with a specific focus on univariate analysis of numerical features, likely involving visualizations and statistical summaries to understand data distributions and relationships.

6.  **Model Building**
    * Trained diffrent model like Naive Bayes , Random Forest , Logistic Regression , XGBoost on our dataset.
    * Evaluated model performance on F1-Score as main metric supported by PU ROC metric.
    * Handled class imabalanced in our dataset using SMOTE technique.
    * Chose best performing model and enhanced it's performance by tuning hyperparameters using RandomizedSearchCV techqnique.

7.  **Model Exporting**
    * After the model building phase,the best performing model and its parameters are saved for future use.
    * The best parameters are saved to `best_params.json`.
    * The entire trained pipeline/estimator is saved as `best_model.pkl` using `joblib`.

### Libraries Used

The project makes use of the following Python libraries:

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
* `seaborn`: For high-level interface for drawing attractive and informative statistical graphics.
* `sklearn.impute.KNNImputer`: Potentially used for advanced missing value imputation, although the provided snippets show mode/median imputation.
* `joblib`: For efficient saving and loading of Python objects, especially large NumPy arrays.
* `json`: For working with JSON data, specifically for saving model parameters.

### Setup and Usage

To run this notebook, ensure you have Jupyter Notebook or JupyterLab installed.

1.  **Clone the repository** (if applicable, or download the `LoanApproval.ipynb` file).
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
3.  **Place data file**: Ensure `Copy of loan - loan.csv` is in the same directory as the `LoanApproval.ipynb` notebook, or update the file path within the notebook.
4.  **Run `LoanApproval.ipynb`**: Open the notebook in Jupyter and execute all cells sequentially to perform the data analysis, cleaning, and model building, including saving the model artifacts.
