#Initial Model 1:
"""import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv('train.csv')

# Calculate the percentage change in WAP to determine the market direction
data['wap_change'] = data['wap'].pct_change()

# Define market direction based on wap_change: 1 for uptrend, -1 for downtrend, 0 for sideways
threshold = 0.01  # Threshold for considering a change significant enough to not be sideways
data['market_direction'] = np.where(data['wap_change'] > threshold, 1,
                                    np.where(data['wap_change'] < -threshold, -1, 0))

# Create additional features: spread, mid price, and volume imbalance
data['spread'] = data['ask_price'] - data['bid_price']
data['mid_price'] = (data['ask_price'] + data['bid_price']) / 2
data['volume_imbalance'] = data['bid_size'] - data['ask_size']

# Drop rows with NaN values that might have been introduced by pct_change
data.dropna(inplace=True)

# Features and Target
X = data[['spread', 'mid_price', 'volume_imbalance', 'bid_price', 'ask_price', 'bid_size', 'ask_size']]
y = data['market_direction']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict the market direction on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the predictions
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
"""

#Model 2: Improved

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load your dataset
data = pd.read_csv('train.csv')

# Calculate the percentage change in WAP to determine the market direction
data['wap_change'] = data['wap'].pct_change()

# Define market direction based on wap_change: 1 for uptrend, -1 for downtrend, 0 for sideways
threshold = 0.01  # Threshold for considering a change significant enough to not be sideways
data['market_direction'] = np.where(data['wap_change'] > threshold, 1,
                                    np.where(data['wap_change'] < -threshold, -1, 0))

# Create additional features: spread, mid price, and volume imbalance
data['spread'] = data['ask_price'] - data['bid_price']
data['mid_price'] = (data['ask_price'] + data['bid_price']) / 2
data['volume_imbalance'] = data['bid_size'] - data['ask_size']
data['price_impact'] = data['wap_change'] * data['volume_imbalance']

# Drop rows with NaN values that might have been introduced by pct_change
data.dropna(inplace=True)

# Features and Target
X = data[['spread', 'mid_price', 'volume_imbalance', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 'price_impact']]
y = data['market_direction']

# Oversample the minority classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.002, random_state=42)

# Initialize and train the Random Forest Classifier with adjusted parameters
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=None, class_weight='balanced', random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict the market direction on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the predictions
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
