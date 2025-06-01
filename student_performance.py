import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('student_scores.csv')

# Display first few rows
print(data.head())

# Visualize
sns.scatterplot(data=data, x='Hours', y='Scores')
plt.title("Hours vs Scores")
plt.show()

# Prepare data
X = data[['Hours']]
y = data['Scores']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict score for a custom value
hours = 9.25
predicted_score = model.predict([[hours]])
print(f"Predicted score for {hours} hours of study: {predicted_score[0]:.2f}")