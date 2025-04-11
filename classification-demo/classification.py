from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def run_classification(data, features, target, test_split, random_state=42):
    # Select features and target
    X = data[features].values
    y = data[target].values

    # Split the data with shuffling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train an SVM with a linear kernel
    model = SVC(probability=False, kernel='linear')  # decision_function available
    model.fit(X_train_scaled, y_train)

    # Get decision scores for the test set
    y_scores = model.decision_function(X_test_scaled)

    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_scores


def predict_with_threshold(decision_scores, threshold):
    return (decision_scores > threshold).astype(int)
