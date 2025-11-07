import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define the 43 column names for the KDD dataset
KDD_COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_data(file_path):
    """
    Loads a KDD dataset file into a pandas DataFrame.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, header=None, names=KDD_COL_NAMES)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def build_preprocessing_pipeline(df):
    """
    Identifies feature types and builds the preprocessing pipeline.
    """
    # 1. Simplify the label: 0 for 'normal', 1 for 'anomaly'
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 2. Separate features (X) and target (y)
    X = df.drop(['label', 'difficulty'], axis=1)
    y = df['label']
    
    # 3. Identify categorical and numerical features
    # These are the 3 features that are non-numeric strings
    categorical_features = ['protocol_type', 'service', 'flag']
    
    # All other features are numerical
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    # 4. Create the preprocessing transformers
    
    # Pipeline for numerical features: Scale them
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features: One-Hot Encode them
    # handle_unknown='ignore' is crucial: it prevents errors if the
    # test set has a category not seen in the training set.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 5. Create the ColumnTransformer to apply transformers to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not listed
    )
    
    return X, y, preprocessor

def main():
    """
    Main function to run the full training and evaluation pipeline.
    """
    print("--- Starting Network Intrusion Detection Model Training ---")
    
    # === Step 1: Load Training Data ===
    train_file = 'KDDTrain+.txt'
    df_train = load_data(train_file)
    if df_train is None:
        return

    # === Step 2: Build Preprocessing Pipeline ===
    print("Building preprocessing pipeline...")
    X_train, y_train, preprocessor = build_preprocessing_pipeline(df_train)
    
    # === Step 3: Create the Full ML Pipeline ===
    # This pipeline combines preprocessing and the classifier.
    # We use a RandomForestClassifier, a powerful and common non-DL model.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42, 
            n_jobs=-1, # Use all available CPU cores
            n_estimators=100 # 100 trees is a good default
        ))
    ])
    
    # === Step 4: Train the Model ===
    print("Training the Random Forest model... (This may take a few minutes)")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    # === Step 5: Save the Model ===
    model_filename = 'nids_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model saved as '{model_filename}'")
    
    # === Step 6: Load and Evaluate on Test Data ===
    print("\n--- Starting Model Evaluation ---")
    test_file = 'KDDTest+.txt'
    df_test = load_data(test_file)
    if df_test is None:
        return

    # Prepare the test data
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    X_test = df_test.drop(['label', 'difficulty'], axis=1)
    y_test = df_test['label']
    
    # === Step 7: Make Predictions ===
    print("Evaluating model on test data...")
    y_pred = model_pipeline.predict(X_test)
    
    # === Step 8: Show Results ===
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomaly (1)'])
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("---------------------------------")

if __name__ == "__main__":
    main()