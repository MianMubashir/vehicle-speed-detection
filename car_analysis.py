import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import os  # Add this import for visualize_results function

# Load and prepare the car dataset
def load_data():
    # You can use any of these popular car datasets:
    # 1. UCI Car Evaluation Database
    # 2. Vehicle Dataset
    # 3. Cars.com Dataset
    
    # For this example, we'll assume we have a CSV file named 'car_data.csv'
    df = pd.read_csv('car_data.csv')
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical using one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

def train_regression_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipelines with preprocessing
    lr_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', LinearRegression())
    ])
    
    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', RandomForestRegressor(n_estimators=200, 
                                          max_depth=20,
                                          min_samples_split=5,
                                          min_samples_leaf=2,
                                          random_state=42))
    ])
    
    # Train and evaluate models
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)
    
    lr_pred = lr_pipeline.predict(X_test)
    rf_pred = rf_pipeline.predict(X_test)
    
    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    return {
        'linear_regression': {'model': lr_pipeline, 'mse': lr_mse},
        'random_forest': {'model': rf_pipeline, 'mse': rf_mse}
    }

def train_classification_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define parameter grids for GridSearchCV
    log_param_grid = {
        'classifier__C': [0.1, 1.0, 10.0, 100.0],
        'classifier__max_iter': [2000],  # Increased max iterations
        'classifier__solver': ['lbfgs', 'liblinear', 'saga'],  # Added saga solver
        'classifier__class_weight': ['balanced']
    }
    
    rf_param_grid = {
        'classifier__n_estimators': [300, 500],  # Increased estimators
        'classifier__max_depth': [20, 25, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__class_weight': ['balanced', 'balanced_subsample']  # Added balanced_subsample
    }
    
    # Create pipelines with preprocessing
    log_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', LogisticRegression())
    ])
    
    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Perform GridSearchCV with reduced cross-validation folds
    log_grid = GridSearchCV(log_pipeline, log_param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1)
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1)
    
    # Train models
    log_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train)
    
    # Get predictions
    log_pred = log_grid.predict(X_test)
    rf_pred = rf_grid.predict(X_test)
    
    # Calculate accuracies
    log_accuracy = accuracy_score(y_test, log_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("\nBest Logistic Regression parameters:", log_grid.best_params_)
    print("Best Random Forest parameters:", rf_grid.best_params_)
    
    print("\nDetailed Classification Report - Logistic Regression:")
    print(classification_report(y_test, log_pred))
    
    print("\nDetailed Classification Report - Random Forest:")
    print(classification_report(y_test, rf_pred))
    
    return {
        'logistic_regression': {'model': log_grid.best_estimator_, 'accuracy': log_accuracy},
        'random_forest': {'model': rf_grid.best_estimator_, 'accuracy': rf_accuracy}
    }

def visualize_results(df, regression_results, classification_results):
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Regression results visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['Linear Regression', 'Random Forest Regression'], 
            [regression_results['linear_regression']['mse'], 
             regression_results['random_forest']['mse']])
    plt.title('Regression Models - Mean Squared Error Comparison')
    plt.ylabel('MSE')
    plt.savefig('visualizations/regression_comparison.png')
    plt.close()
    
    # Classification results visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['Logistic Regression', 'Random Forest Classifier'], 
            [classification_results['logistic_regression']['accuracy'], 
             classification_results['random_forest']['accuracy']])
    plt.title('Classification Models - Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('visualizations/classification_comparison.png')
    plt.close()

def train_sgd_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define parameter grid for SGD
    sgd_param_grid = {
        'classifier__loss': ['hinge', 'log_loss', 'modified_huber'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__max_iter': [2000],
        'classifier__learning_rate': ['optimal', 'adaptive'],
        'classifier__class_weight': ['balanced']
    }
    
    # Create pipeline with preprocessing
    sgd_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', SGDClassifier(random_state=42))
    ])
    
    # Perform GridSearchCV
    sgd_grid = GridSearchCV(sgd_pipeline, sgd_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    sgd_grid.fit(X_train, y_train)
    
    # Get predictions
    sgd_pred = sgd_grid.predict(X_test)
    
    # Calculate accuracy
    sgd_accuracy = accuracy_score(y_test, sgd_pred)
    
    print("\nBest SGD parameters:", sgd_grid.best_params_)
    print("\nDetailed Classification Report - SGD:")
    print(classification_report(y_test, sgd_pred))
    
    print(f"\nSGD Classifier Accuracy: {sgd_accuracy * 100:.2f}%")
    
    return {
        'model': sgd_grid.best_estimator_,
        'accuracy': sgd_accuracy
    }

def perform_hierarchical_clustering(X, n_clusters=5):
    # Scale the features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(X_scaled)
    
    # Create linkage matrix for dendrogram
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig('visualizations/dendrogram.png')
    plt.close()
    
    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_labels, bins=n_clusters, rwidth=0.8)
    plt.title('Distribution of Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.savefig('visualizations/cluster_distribution.png')
    plt.close()
    
    return cluster_labels

def main():
    # Load and preprocess data
    df = load_data()
    
    # Store the Category column before preprocessing
    category_labels = df['Category']
    
    # Remove Category column before preprocessing
    df_for_preprocessing = df.drop(['Category'], axis=1)
    
    # Preprocess the data
    processed_df = preprocess_data(df_for_preprocessing)
    
    # Prepare features for classification
    X_class = processed_df.drop(['Price'], axis=1)
    y_class = category_labels
    
    # Train SGD model
    sgd_results = train_sgd_model(X_class, y_class)
    
    # Perform hierarchical clustering
    cluster_labels = perform_hierarchical_clustering(X_class)
    
    print("\nHierarchical Clustering completed.")
    print(f"Number of samples in each cluster: {np.bincount(cluster_labels)}")

if __name__ == "__main__":
    main()