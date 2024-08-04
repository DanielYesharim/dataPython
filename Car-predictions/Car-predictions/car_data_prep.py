import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def prepare_data(df):


    # Replace commas and handle non-numeric values for 'Km' and 'capacity_Engine' 
    def convert_to_float(value):
        try:
            return float(value.replace(',', ''))
        except:
            return np.nan
    
    # Apply the conversion function to 'Km' and 'capacity_Engine'
    df['Km'] = df['Km'].apply(convert_to_float)
    df['capacity_Engine'] = df['capacity_Engine'].apply(convert_to_float)
    
    # Fixing manufacturer and Engine_type names
    df['manufactor'] = df['manufactor'].replace('Lexsus', 'לקסוס')
    df['Engine_type']= df['Engine_type'].replace('היברידי', 'היבריד')
    df['City'] = df['City'].replace('jeruslem', 'ירושלים')
    df['City'] = df['City'].replace('Rehovot', 'רהבות')
    df['City'] = df['City'].replace('Rishon LeTsiyon', 'ראשון לציון')
    df['City'] = df['City'].replace('haifa', 'חיפה')
    df['City'] = df['City'].replace('Tel aviv', 'תל אביב')
    df['City'] = df['City'].replace('ashdod', 'אשדוד')
    df['City'] = df['City'].replace('Tzur Natan', 'צור נתן')


     # Drop columns with excessive missing values or less efect on the modal
    df.drop(columns=['Prev_ownership', 'Curr_ownership', 'Test', 'Pic_num', 'Cre_date', 'Repub_date', 'Description'], inplace=True)

    # Define numerical columns
    numerical_columns = ['Year', 'Hand', 'Km', 'capacity_Engine', 'Supply_score',]

    # Define categorical columns
    categorical_columns = ['manufactor', 'Gear', 'Engine_type', 'model', 'Color','Area','City']

    # Create preprocessing pipeline for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Prepare features and target variable
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Transform the data
    X_prepared = preprocessor.fit_transform(X)

    # Get feature names for the transformed data
    num_features = numerical_columns
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns)
    all_features = np.concatenate([num_features, cat_features])

    # Create a DataFrame with the transformed data
    df_prepared = pd.DataFrame(X_prepared, columns=all_features)

     # Add the target variable 'Price' back to the DataFrame
    df_prepared['Price'] = y
    
    # Calculate correlation matrix, to check which columns to keep
    corr_matrix = df_prepared.corr()

    # Select columns with correlation higher than a threshold with 'Price'
    threshold = 0.005
    high_corr_columns = corr_matrix[abs(corr_matrix['Price']) > threshold].index

    # Filter all_features to keep only columns with high correlation to 'Price'
    df_prepared = df_prepared[high_corr_columns]
    

    return df_prepared

