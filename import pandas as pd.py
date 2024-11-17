import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('vehicle_count_data.csv')

# Convert Timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Feature engineering: Extracting hour from Timestamp
data['Hour'] = data['Timestamp'].dt.hour

# Split dataset into features (X) and target (y)
X = data[['Hour']]
y = data['VehicleCount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Hour', y='VehicleCount', data=data)
plt.title('Vehicle Count by Hour')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['VehicleCount'], bins=30, edgecolor='k')
plt.title('Distribution of Vehicle Count')
plt.xlabel('Vehicle Count')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Hour'], data['VehicleCount'])
plt.title('Vehicle Count vs Hour')
plt.xlabel('Hour')
plt.ylabel('Vehicle Count')
plt.show()

# Pearson correlation
correlation = data.corr()
print(correlation)

# Heatmap of correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Adding predictions to the test set
test_data = X_test.copy()
test_data['TrueCount'] = y_test
test_data['PredictedCount'] = y_pred

# Plotting true vs predicted counts
plt.figure(figsize=(10, 6))
plt.scatter(test_data['Hour'], test_data['TrueCount'], label='True Count')
plt.scatter(test_data['Hour'], test_data['PredictedCount'], label='Predicted Count', marker='x')
plt.xlabel('Hour')
plt.ylabel('Vehicle Count')
plt.legend()
plt.title('True vs Predicted Vehicle Count')
plt.show()


