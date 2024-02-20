import cudf
import pandas as pd
import numpy as np
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split
from sklearn.model_selection import KFold
from cuml.preprocessing import StandardScaler
import zarr
from pathlib import Path

class BarcodeFilter:
    def __init__(self, data_dir_path):
        self._data_dir_path = data_dir_path
        self._features = ["max_intensity", "area", "min_dispersion", "mean_distance_l2", "mean_distance_cosine", "called_distance_l2", "called_distance_cosine"]
        self._target = "gene_id"
        self._model = LogisticRegression()
        self._threshold = 0.5  # Default threshold
        self._target_metric = .05
        self._scaler = StandardScaler()
        self.full_df, self.df = self._load_data()
        self._X_train_scaled = None
        self._y_train = None
        self._X_test_scaled = None
        self._y_test = None
        self._load_codebook()
        self._prepare_data()
        
    def _load_codebook(self):

        calibration_dir_path = self._data_dir_path / Path('calibrations.zarr')
        self._calibration_zarr = zarr.open(calibration_dir_path,mode='r')
        self._df_codebook = pd.DataFrame(self._calibration_zarr.attrs['codebook'])
        self._df_codebook.fillna(0, inplace=True)
        
        self._codebook_matrix = self._df_codebook.iloc[:, 1:17].to_numpy().astype(int)
        self._gene_ids = self._df_codebook.iloc[:, 0].tolist()
        
        self._total_blank = 0
        self._total_coding = 0
        for gene in self._gene_ids:
            if "Blank" in gene:
                self._total_blank += 1
            else:
                self._total_coding += 1

    def _load_data(self,
                   data_type: str = 'csv'):
        
        decoded_dir_path = self._data_dir_path / Path('decoded')
        decoded = []
        if data_type == 'csv':
            decoded_files = decoded_dir_path.glob('*.csv')
            for decoded_file in decoded_files:
                decoded.append(pd.read_csv(decoded_file))
        df = pd.concat(decoded, ignore_index=True)
        df[self._target] = df[self._target].apply(lambda x: 0 if x.startswith("Blank") else 1)
        
        # Balance the dataset
        blanks = df[df[self._target] == 0]
        non_blanks = df[df[self._target] == 1]
        
        # Randomly sample non-blanks to match the number of blanks
        non_blanks_sampled = non_blanks.sample(n=len(blanks), random_state=42)
        
        # Concatenate the balanced datasets
        balanced_df = pd.concat([blanks, non_blanks_sampled], ignore_index=True)
        
        return df, balanced_df
    
    def _prepare_data(self):
        gdf = cudf.from_pandas(self.df)
        X_train, X_test, y_train, y_test = train_test_split(gdf[self._features], gdf[self._target], test_size=0.2, random_state=42)
        self._X_train_scaled = self._scaler.fit_transform(X_train)
        self._X_test_scaled = self._scaler.transform(X_test)
        self._y_train = y_train
        self._y_test = y_test
    
    def train_model(self):
        self._model.fit(self._X_train_scaled, self._y_train)
    
    def find_optimal_threshold(self):
        # Get prediction probabilities
        y_prob_df = self._model.predict_proba(self._X_test_scaled)
        
        # Ensure it's a cuDF DataFrame to use .iloc, then convert to NumPy array for processing
        if isinstance(y_prob_df, cudf.DataFrame):
            y_prob = y_prob_df.iloc[:, 1].to_numpy()
        elif isinstance(y_prob_df, cudf.Series):
            # If y_prob_df is a Series, directly convert to NumPy array
            y_prob = y_prob_df.to_numpy()
        else:
            # Handle other types if necessary
            raise TypeError("Unexpected return type from predict_proba")

        closest_metric = float('inf')
        best_threshold = 0.5
        for threshold in np.linspace(0, 1, 101):
            y_pred = (y_prob >= threshold).astype(int)
            metric_value = self._calculate_custom_metric(self._y_test.to_numpy(), y_pred, self._total_blank, self._total_coding)
            
            # Calculate the absolute difference from the target metric
            metric_difference = abs(metric_value - self._target_metric)
            
            # Update if this threshold gives a closer metric to the target
            if metric_difference < closest_metric:
                closest_metric = metric_difference
                best_threshold = threshold
                best_metric = metric_value  # Keep track of the best metric value for reporting
    
        self._threshold = best_threshold
        print(f"Optimal Threshold: {self._threshold}, Metric Value: {best_metric}, Closest to Target Metric: {self._target_metric}")

    
    @staticmethod
    def _calculate_custom_metric(y_true, y_pred,total_blank,total_coding):
        num_blanks = np.sum((y_pred == 0) & (y_true == 0))
        num_coding = np.sum((y_pred == 1) & (y_true == 1))
        if num_coding == 0:
            return np.inf
        else:
            print(num_blanks,num_coding,(num_blanks / total_blank) / (num_coding / total_coding))
            return (num_blanks / total_blank) / (num_coding / total_coding)
    
    def filter_dataframe(self):
        gdf = cudf.from_pandas(self.full_df)
        X = self._scaler.transform(gdf[self._features])
        y_prob_df = self._model.predict_proba(X)
        # Ensure correct access to the second column
        if isinstance(y_prob_df, cudf.DataFrame):
            y_prob = y_prob_df.iloc[:, 1]  # This should not raise StopIteration if y_prob_df is a DataFrame
        else:
            raise TypeError("Unexpected type for prediction probabilities")
        gdf['prediction'] = (y_prob >= self._threshold).astype(int)
        filtered_gdf = gdf[gdf['prediction'] == 1]
        filtered_df = filtered_gdf.to_pandas()
        return filtered_df

    def print_entry_counts(self, df=None):
        if df is None:
            df = self.full_df
        num_blanks = df[self._target].value_counts().get(0, 0)
        num_coding = df[self._target].value_counts().get(1, 0)
        print(f"Total Blanks: {num_blanks}, Total Coding: {num_coding}")
        print(f"Misidentification rate: {(num_blanks / self._total_blank) / (num_coding / self._total_coding)}")
        
    def cross_validate_model(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        accuracies = []
        custom_metrics = []
        
        for train_index, test_index in kf.split(self.df[self._features], self.df[self._target]):
            # Use indices to split the cuDF DataFrame
            X_train, X_test = self.df.iloc[train_index][self._features], self.df.iloc[test_index][self._features]
            y_train, y_test = self.df.iloc[train_index][self._target], self.df.iloc[test_index][self._target]
            
            # Scale features
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_test_scaled = self._scaler.transform(X_test)
            
            # Train model
            self._model.fit(X_train_scaled, y_train)
            
            # Predict on test set
            y_pred = self._model.predict(X_test_scaled)
            
            # Calculate accuracy and custom metric
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            custom_metric = self._calculate_custom_metric(y_test.to_array(), y_pred.to_array(), self._total_blank, self._total_coding)
            custom_metrics.append(custom_metric)
            
        # Calculate and print average metrics
        print(f"Average Accuracy: {np.mean(accuracies)}, Average Custom Metric: {np.mean(custom_metrics)}")