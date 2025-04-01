import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

def handle_child_model(X, y):
    model_file = "asd_model_children.pkl"
    try:
        model = joblib.load(model_file)
        if any(leak in model.feature_names_in_ for leak in ["id", "result"]):
            raise ValueError("Invalid features")
        return model, False
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)
        return model, True

def handle_adult_model(X, y):
    model_file = "asd_model_adults.pkl"
    try:
        model = joblib.load(model_file)
        if any(leak in model.feature_names_in_ for leak in ["id", "result"]):
            raise ValueError("Invalid features")
        return model, False
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [3, 5, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        joblib.dump(model, model_file)
        return model, True