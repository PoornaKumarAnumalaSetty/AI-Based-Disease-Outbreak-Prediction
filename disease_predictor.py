import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import logging

# FileAccessHandler Class: Handles loading data from CSV files
class FileAccessHandler:
    def __init__(self):
        self.data_directory = Path(r"C:\Users\joshu\Desktop\CODING\Hackare 2.0")
        
        self.diseases = [
            'hepatitis', 'measles', 'mumps', 
            'pertussis', 'polio', 'rubella', 'smallpox'
        ]
        
        # Setup logging to track file operations
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def list_available_files(self):
        """
        List all CSV files in the directory
        """
        try:
            csv_files = list(self.data_directory.glob('*.csv'))
            if csv_files:
                self.logger.info("Found CSV files:")
                for file in csv_files:
                    self.logger.info(f"- {file.name}")
                return csv_files
            else:
                self.logger.warning("No CSV files found in the directory")
                return []
        except Exception as e:
            self.logger.error(f"Error accessing directory: {str(e)}")
            return []

    def load_csv_file(self, filename):
        """
        Load a single CSV file
        """
        try:
            file_path = self.data_directory / filename
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded {filename}")
            self.logger.info(f"Columns found: {', '.join(df.columns)}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            return None

    def load_all_disease_data(self):
        """
        Load all disease CSV files from the directory
        """
        all_data = {}
        available_files = self.list_available_files()
        
        for disease in self.diseases:
            filename = f"{disease}.csv"
            if any(file.name == filename for file in available_files):
                data = self.load_csv_file(filename)
                if data is not None:
                    all_data[disease] = data
                    
        return all_data

# Create an instance of the FileAccessHandler
file_handler = FileAccessHandler()

# Load all disease data
loaded_data = file_handler.load_all_disease_data()

# Print the data that has been loaded
for disease, data in loaded_data.items():
    print(f"\nData for {disease}:")
    print(data.head())  # Use .head() to print the first few rows of each dataframe

# DiseaseOutbreakPredictor Class: Handles prediction and analysis
class DiseaseOutbreakPredictor:
    def __init__(self):
        self.diseases = ['hepatitis', 'measles', 'mumps', 'pertussis', 'polio', 'rubella', 'smallpox']
        self.models = {}
        self.scalers = {}
        self.data = None
        self.states = []

    def load_and_preprocess_data(self):
        all_data = []
        
        for disease in self.diseases:
            df = pd.read_csv(f'C:\\Users\\joshu\\Desktop\\CODING\\Hackare 2.0\\{disease}.csv')
            self.states = df['state'].unique()  # Extract unique states
            df['disease'] = disease
            all_data.append(df)
            
        self.data = pd.concat(all_data, ignore_index=True)
        self.data.replace('\\N', np.nan, inplace=True)
        self.data['week'] = pd.to_datetime(self.data['week'], errors='coerce')
        self.data['year'] = self.data['week'].dt.year
        self.data['month'] = self.data['week'].dt.month
        self.data['week_of_year'] = self.data['week'].dt.isocalendar().week
        self.data['cases_lag_1'] = self.data.groupby(['state', 'disease'])['cases'].shift(1)
        self.data['cases_lag_4'] = self.data.groupby(['state', 'disease'])['cases'].shift(4)
        self.data['cases'] = pd.to_numeric(self.data['cases'], errors='coerce')
        self.data['cases_lag_1'] = pd.to_numeric(self.data['cases_lag_1'], errors='coerce')
        self.data['cases_lag_4'] = pd.to_numeric(self.data['cases_lag_4'], errors='coerce')
        self.data = self.data.dropna()

    def train_models(self):
        for disease in self.diseases:
            disease_data = self.data[self.data['disease'] == disease]
            features = ['year', 'month', 'week_of_year', 'cases_lag_1', 'cases_lag_4']
            X = disease_data[features]
            y = disease_data['cases']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            self.models[disease] = model
            self.scalers[disease] = scaler
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nMetrics for {disease}:")
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R² Score: {r2:.2f}")

    def get_highest_cases_by_disease(self, state):
        if state not in self.states:
            print(f"State '{state}' is not available in the data.")
            return

        print(f"\nHighest cases by disease in {state.upper()}:\n")
        
        for disease in self.diseases:
            disease_data = self.data[(self.data['state'] == state) & (self.data['disease'] == disease)]
            if not disease_data.empty:
                max_cases = disease_data['cases'].max()
                print(f"{disease.capitalize()}: {max_cases} cases")

    def predict_outbreak(self, state, disease, current_cases):
        if disease not in self.models:
            return None
        
        current_date = datetime.now()
        features = {
            'year': [current_date.year],
            'month': [current_date.month],
            'week_of_year': [current_date.isocalendar()[1]],
            'cases_lag_1': [current_cases],
            'cases_lag_4': [current_cases]  # Simplified for demo
        }
        
        input_df = pd.DataFrame(features)
        input_scaled = self.scalers[disease].transform(input_df)
        prediction = self.models[disease].predict(input_scaled)[0]
        
        return max(0, prediction)

    def predict_future_outbreak(self, state, disease, current_cases, future_years):
        predictions = []
        current_date = datetime.now()
        current_week = current_date.isocalendar()[1]

        for year in range(future_years):
            for week in range(1, 53):  # Assuming 52 weeks in a year
                features = {
                    'year': [current_date.year + year],
                    'month': [1],  # January
                    'week_of_year': [week],
                    'cases_lag_1': [current_cases],
                    'cases_lag_4': [current_cases]  # Simplified for demo
                }
                
                input_df = pd.DataFrame(features)
                input_scaled = self.scalers[disease].transform(input_df)
                predicted_cases = self.models[disease].predict(input_scaled)[0]
                
                predictions.append({
                    'year': current_date.year + year,
                    'week': week,
                    'predicted_cases': max(0, predicted_cases)
                })

        return predictions

    def plot_bar_chart(self, state, disease, actual_cases, predicted_cases):
        plt.figure(figsize=(10, 6))
        labels = ['Actual Cases', 'Predicted Cases']
        values = [actual_cases, predicted_cases]
        plt.bar(labels, values, color=['blue', 'orange'])
        plt.title(f'Actual vs Predicted Cases for {disease.capitalize()} in {state.upper()}')
        plt.ylabel('Number of Cases')
        plt.savefig(f'bar_chart_{state}_{disease}.png')
        plt.show()

    def plot_pie_chart(self, state, disease, actual_cases, predicted_cases):
        plt.figure(figsize=(8, 8))
        labels = ['Actual Cases', 'Predicted Cases']
        sizes = [actual_cases, predicted_cases]
        colors = ['lightblue', 'lightcoral']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.axis('equal')
        plt.title(f'Proportion of Actual vs Predicted Cases\n({disease.capitalize()} in {state.upper()})')
        plt.savefig(f'pie_chart_{state}_{disease}.png')
        plt.show()

# Create an instance of DiseaseOutbreakPredictor
predictor = DiseaseOutbreakPredictor()

# Load and preprocess data
predictor.load_and_preprocess_data()

# Train the prediction models
predictor.train_models()

# User interaction and visualization
def user_input_and_visualization():
    state = input("Enter the state abbreviation (e.g., CA for California): ").strip().upper()
    disease = input("Enter the disease name (e.g., measles): ").strip().lower()
    
    # Validate user input for disease
    if disease not in predictor.diseases:
        print(f"Invalid disease name. Available diseases: {', '.join(predictor.diseases)}")
        return

    try:
        current_cases = int(input("Enter the current number of cases: "))
    except ValueError:
        print("Invalid input for current cases. Please enter a numeric value.")
        return

    # Get highest cases by disease
    predictor.get_highest_cases_by_disease(state)

    # Predict outbreak for current year
    predicted_cases = predictor.predict_outbreak(state, disease, current_cases)
    print(f"\nPredicted cases for {disease.capitalize()} in {state.upper()}: {predicted_cases}")

    # Visualize actual and predicted cases
    try:
        actual_cases = int(input("Enter the actual cases for comparison: "))
        predictor.plot_bar_chart(state, disease, actual_cases, predicted_cases)
        predictor.plot_pie_chart(state, disease, actual_cases, predicted_cases)
    except ValueError:
        print("Invalid input for actual cases. Please enter a numeric value.")
        return

    # Option to predict future outbreaks
    future_prediction = input("Would you like to predict future outbreaks for the next few years? (yes/no): ").strip().lower()
    if future_prediction == 'yes':
        try:
            future_years = int(input("Enter the number of future years to predict: "))
            predictions = predictor.predict_future_outbreak(state, disease, current_cases, future_years)
            print(f"\nWeekly predictions for {disease.capitalize()} in {state.upper()} over the next {future_years} years:\n")
            
            for pred in predictions:
                print(f"Year: {pred['year']}, Week: {pred['week']}, Predicted Cases: {pred['predicted_cases']:.2f}")
            
            # Optional: you can visualize the predictions as a time series plot if needed.
            visualize_predictions(predictions, state, disease)
        except ValueError:
            print("Invalid input for the number of years. Please enter a numeric value.")
            return

def visualize_predictions(predictions, state, disease):
    """ 
    Visualize future predictions as a time series plot.
    """
    df_predictions = pd.DataFrame(predictions)
    plt.figure(figsize=(12, 6))
    plt.plot(df_predictions['week'], df_predictions['predicted_cases'], marker='o', label='Predicted Cases')
    plt.title(f'Future Predicted Cases for {disease.capitalize()} in {state.upper()}')
    plt.xlabel('Week')
    plt.ylabel('Predicted Cases')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'future_predictions_{state}_{disease}.png')
    plt.show()

# Call the user input and visualization function
user_input_and_visualization()
