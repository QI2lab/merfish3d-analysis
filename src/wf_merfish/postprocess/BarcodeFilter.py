import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, classification_report
import zarr
from pathlib import Path

class BarcodeFilter:
    def __init__(self, data_dir_path):
        self._data_dir_path = data_dir_path
        self._features = ["coding", "max_intensity", "area", "min_dispersion", "mean_distance_l2", "mean_distance_cosine", "called_distance_l2", "called_distance_cosine"]
        self._target = "coding"
        self._classify = "gene_id"
        self._model = MLPClassifier(alpha=1,max_iter=1000)
        self._target_fdr = .2
        self._scaler = StandardScaler()
        self.full_df = self._load_data()
        self._X_train_scaled = None
        self._y_train = None
        self._X_test_scaled = None
        self._y_test = None
        self._load_codebook()
        #self._prepare_data()
        
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
                decoded.append(pd.read_csv(decoded_file, index_col=0))
        df = pd.concat(decoded, ignore_index=True)
        df[self._target] = df[self._classify].apply(lambda x: 0 if x.startswith("Blank") else 1)
        
        return df
    
    def _find_threshold_wollman(self):
        
        correction = (self._total_blank+self._total_coding) / self._total_blank
        c = self.full_df['called_distance_l2']
        vmin,vmax = np.percentile(c,[.001,99.999])
        bins = np.linspace(vmin,vmax,1000)
        fpr = np.zeros_like(bins)
        for i,b in enumerate(bins):
            filtered_spots = self.full_df[self.full_df['called_distance_l2']<b]
            if filtered_spots.shape[0] >1:
                fpr[i] = np.round(correction*np.sum(filtered_spots[self._target]==0)/filtered_spots.shape[1],3)
            else:
                fpr[i] = 0
            if fpr[i]>self._target_fdr: #HARDCODED
                thresh = np.round(bins[i-1],3)
                print('thresh: '+str(thresh))
                print('FPR: '+str(fpr[i-1]))
                break
            
        self._thresh = thresh
            
    def _prepare_data_wollman(self):
        training_data = self.full_df[self.full_df['dist']<self._thresh].copy()
        data = training_data[self._features]
        data_true = data.loc[data[data['coding']==1].index]
        data_false = data.loc[data[data['coding']==0].index]
        application_data = self.full_df.copy()
        if np.min([data_true.shape[0],data_false.shape[0]])>0:
                """ downsample to same size """
                s = np.min([data_true.shape[0],data_false.shape[0]])
                data_true_down = data_true.loc[np.random.choice(data_true.index,s,replace=False)]
                data_false_down = data_false.loc[np.random.choice(data_false.index,s,replace=False)]
                data_down = pd.concat([data_true_down,data_false_down])
                X_train, X_test, y_train, y_test = train_test_split(data_down.drop('coding',axis=1),data_down['coding'], test_size=0.30,random_state=42)
                X_train_scaled = self._scaler.transform(X_train)
                X_test_scaled = self._scaler.transform(X_test)
                self._model.fit(X_train_scaled,y_train)
                predictions = self.model.predict(X_test_scaled)
                print(classification_report(y_test,predictions))
                data = application_data[[i for i in columns if not 'X'==i]]
                X = self.scaler.transform(data)
                pixels['predicted_X'] = self.model.predict(X)
                pixels['probabilities_X'] = self.model.predict_proba(X)[:,1] 
                filtered_pixels = pixels.copy()
                data = application_data[[i for i in columns if not 'X'==i]]
                X = self.scaler.transform(data)
                pixels['predicted_X'] = self.model.predict(X)
                pixels['probabilities_X'] = self.model.predict_proba(X)[:,1] 
                filtered_pixels = pixels.copy()
    
    def _prepare_data(self):

        X, y = gdf[self._features], gdf[self._target]
        
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scaling the features
        self._X_train_scaled = self._scaler.fit_transform(X_train)
        self._X_test_scaled = self._scaler.transform(X_test)
        self._y_train = y_train
        self._y_test = y_test
    
        # gdf = cudf.from_pandas(self.full_df)
        # X_train, X_test, y_train, y_test = train_test_split(gdf[self._features], gdf[self._target], test_size=0.3, random_state=42)
        
        # X_train_pd = X_train.to_pandas()
        # y_train_pd = y_train.to_pandas()
        
        # # Apply SMOTE
        # svm = SVC()
        # nn = NearestNeighbors(n_neighbors=6)
        # X_train_res, y_train_res = SVMSMOTE(
        #     k_neighbors=nn,
        #     m_neighbors=nn,
        #     svm_estimator=svm,
        #     random_state=42
        #     ).fit_resample(X_train_pd, y_train_pd)
        
        # # Convert back to cuDF DataFrame for further processing
        # self._X_train_scaled = self._scaler.fit_transform(cudf.DataFrame.from_pandas(X_train_res))
        # self._X_test_scaled = self._scaler.transform(X_test)
        # self._y_train = cudf.Series(y_train_res)
        # self._y_test = y_test
        
    
    def train_model(self):
        self._model.fit(self._X_train_scaled, self._y_train)
        
    def find_optimal_threshold_v2(self):
        # Predict probabilities on the test set
        # Predict probabilities
        y_probs = self._model.predict_proba(self._X_test_scaled).to_pandas()  # Convert to pandas DataFrame for compatibility

        # Select the probabilities for the positive class
        positive_class_probs = y_probs.iloc[:, 1].to_numpy()  # Convert to NumPy array

        # Ensure self._y_test is also a NumPy array for compatibility with Scikit-learn
        y_test_numpy = self._y_test.to_numpy() if isinstance(self._y_test, cudf.Series) else self._y_test

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test_numpy, positive_class_probs)

        f1_scores = 2 * recall * precision / (recall + precision)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        self._threshold = optimal_threshold
        print(f"Optimal Threshold: {self._threshold}")

    def filter_dataframe_v2(self):
        gdf = cudf.from_pandas(self.full_df)
        X_scaled = self._scaler.transform(gdf[self._features])
        
        y_probs = self._model.predict_proba(X_scaled).to_pandas()
        predictions = (y_probs.iloc[:, 1].to_numpy() >= self._threshold).astype(int)
        
        gdf['prediction'] = predictions
        filtered_gdf = gdf[gdf['prediction'] == 1]
        
        return filtered_gdf.to_pandas()
    
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
            if metric_difference < closest_metric and metric_value>0:
                closest_metric = metric_difference
                best_threshold = threshold
                best_metric = metric_value  # Keep track of the best metric value for reporting
    
        self._threshold = best_threshold
        print(f"Optimal Threshold: {self._threshold}, Metric Value: {best_metric}, Closest to Target Metric: {self._target_metric}")

    
    @staticmethod
    def _calculate_custom_metric(y_true, y_pred,total_blank,total_coding):
        num_blanks = np.sum((y_pred == 0))
        num_coding = np.sum((y_pred == 1))
        if num_coding == 0:
            return np.inf
        else:
            print(num_blanks,total_blank,num_coding,total_coding,(num_blanks / total_blank) / (num_coding / total_coding))
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

    def print_entry_counts(self, df):
        
        gene_ids = df.iloc[:, 1].tolist()
        
        num_blanks = 0
        num_coding = 0
        for gene in gene_ids:
            if "Blank" in gene:
                num_blanks += 1
            else:
                num_coding += 1
                
        print(f"Total Blanks: {num_blanks}, Total Coding: {num_coding}")
        print(f"Misidentification rate: {(num_blanks / self._total_blank) / (num_coding / self._total_coding)}")