from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def data_preprocessing(df):
    atttributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df.columns = atttributes
    df.replace('?', np.nan, inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    X = imputed_data.drop('target', axis=1)
    y = imputed_data['target']
    imputed_data.to_csv('preprocessed_heartdisease.csv', index=False) 
    return X, y

def training_model(X, y):
    stratified_setup = StratifiedKFold(n_splits=20, shuffle=True, random_state=64)

    models_accuracy = {'GaussianNB': [], 'RandomForest': [], 'VotingClassifier': []}
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    mean_auc = 0.0
    fold = 0

    for train_index, test_index in stratified_setup.split(X, y):
        fold += 1
        print("\n\n\nFold ", fold, "\n\n\n")
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        naive_bayes_generation = GaussianNB()
        naive_bayes_generation.fit(x_train, y_train)
        naive_bayes_generation_accuracy = metrics.accuracy_score(y_test, naive_bayes_generation.predict(x_test))
        models_accuracy['GaussianNB'].append(naive_bayes_generation_accuracy)
        print("GaussianNB accuracy(in %):", naive_bayes_generation_accuracy * 100)

        random_forest_generation = RandomForestClassifier(n_estimators=128, random_state=77, max_depth=3, min_samples_split=2, min_samples_leaf=1)

        random_forest_generation.fit(x_train, y_train)
        random_forest_accuracy = metrics.accuracy_score(y_test, random_forest_generation.predict(x_test))
        models_accuracy['RandomForest'].append(random_forest_accuracy)
        print("RandomForest accuracy(in %):", random_forest_accuracy * 100)
        eclf = VotingClassifier(estimators=[('naive_bayes_generation', naive_bayes_generation), ('random_forest_generation', random_forest_generation)], voting='soft')
        eclf.fit(x_train, y_train)
        eclf_accuracy = metrics.accuracy_score(y_test, eclf.predict(x_test))
        models_accuracy['VotingClassifier'].append(eclf_accuracy)
        print("Ensemble model accuracy(in %):", eclf_accuracy * 100)

        print("\n\n\n\nConfusion Matrix")
        cf_matrix = confusion_matrix(y_test, eclf.predict(x_test))
        print(cf_matrix)

        print("\n\n\n\nF1 Score")
        f_score = f1_score(y_test, eclf.predict(x_test))
        print(f_score)

        # ROC Curve generation
        fpr, tpr, _ = roc_curve(y_test, eclf.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_auc += roc_auc

    mean_tpr /= stratified_setup.get_n_splits(X, y)
    mean_auc /= stratified_setup.get_n_splits(X, y)
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2, label='Mean ROC (area = %0.2f)' % mean_auc)

    # Finalize and save the ROC curve
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    # Print average accuracy of each model
    for model, accuracies in models_accuracy.items():
        print(f"Average {model} accuracy: {np.mean(accuracies)}")

    return eclf  # Return the trained ensemble model

def predict_input(user_input, eclf):
    # Load the dataset
    df = pd.read_csv('C:/Users/ramkr/OneDrive/Desktop/flask/.venv/heartdisease copy.csv', header=None)

    # Preprocess the data
    X, y = data_preprocessing(df)

   
    user_input_tuple = (
        user_input['age'],
        user_input['sex'],
        user_input['cp'],
        user_input['trestbps'],
        user_input['chol'],
        user_input['fbs'],
        user_input['restecg'],
        user_input['thalach'],
        user_input['exang'],
        user_input['oldpeak'],
        user_input['slope'],
        user_input['ca'],
        user_input['thal']
    )

    # Combine preprocessing steps for numerical features
    user_input_array = np.array([user_input_tuple])

    # Make prediction using the trained model
    prediction = eclf.predict(user_input_array)

    return prediction

if __name__ == "__main__":
    # User input (for testing purposes)
    user_input = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 3,
        'ca': 0,
        'thal': 1
    }

    # Load the dataset
    df = pd.read_csv('C:/Users/ramkr/OneDrive/Desktop/flask/.venv/heartdisease copy.csv', header=None)

    # Preprocess the data
    X, y =data_preprocessing(df)

    # Train the model
    trained_model = training_model(X, y)

    # Test the predict function
    prediction = predict_input(user_input, trained_model)
    print("Prediction:", prediction)
