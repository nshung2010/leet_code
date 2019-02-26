# Data analysis and wrangling package
import pandas as pd
import numpy as np
import random as rnd

# Visualization package
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning package

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

class TITATIC(object):
    def __init__(self, train_data_file='../input/train.csv',
                       test_data_file='../input/test.csv',
                       submission_output='../output/submission_v2.csv'):

        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.submission_output = submission_output
        self.models = [LogisticRegression(),
                      KNeighborsClassifier(n_neighbors=3),
                      GaussianNB(),
                      Perceptron(),
                      LinearSVC(),
                      SGDClassifier(),
                      DecisionTreeClassifier(),
                      RandomForestClassifier(n_estimators=100)]

    def read_data_(self):
        """
        Read the data from train_data_file and test_data_file
        INPUT:
        -self
        OUTPUT:
        - train_df
        - test_df
        """
        # get the data
        train_df = pd.read_csv(self.train_data_file)
        test_df = pd.read_csv(self.test_data_file)
        self.passenger_id = test_df['PassengerId']
        return train_df, test_df

    def clean_data(self, train_df, test_df):
        """
        Cleaning the data; add some features and delete some unnessesary
        features. This is summary of what have been investigated from
        notebook
        INPUT:
        - train_df: DataFrame of train data
        - test_df: DataFrame of test data

        OUTPUT: "clean" train data and test data

        """

        combine = [train_df, test_df]
        new_combine = []
        for dataset in combine:
            # drop Ticket and Cabin features (not important)
            dataset = dataset.drop(['Ticket', 'Cabin'], axis=1)

            # Build Title feature based on Name
            dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
            title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
            dataset = dataset.drop(['Name', 'PassengerId'], axis=1)

            # change Sex to categorical numbers
            dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
            guess_ages = np.zeros((2, 3))
            # fill age based on median value from same Sex and Pclass
            for i in range(0, 2):
                for j in range(0, 3):
                    guess_df = dataset[(dataset['Sex']==i) & \
                                      (dataset['Pclass']==j+1)]['Age'].dropna()

                    age_guess = guess_df.median()

                    # convert randome age float to nearest 0.5 age
                    guess_ages[i, j] = int(age_guess/0.5+0.5)*0.5

            for i in range(0, 2):
                for j in range(0, 3):
                    dataset.loc[(dataset.Age.isnull()) & \
                    (dataset.Sex ==i) & (dataset.Pclass==j+1), \
                    'Age'] = guess_ages[i, j]

            dataset['Age'] = dataset['Age'].astype(int)

            # Set 5 age band for the Age columns
            dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[ dataset['Age'] > 64, 'Age']

            # Create FamilySize feature
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
            dataset = dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

            # Creat feature Age*Class
            dataset['Age*Class'] = dataset.Age * dataset.Pclass

            # Fill missing values for Embarked columns
            freq_port = dataset.Embarked.dropna().mode()[0]
            dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
            # convert Embarked to Categorical
            # dataset['Embarked'] = pd.Categorical(dataset['Embarked']).codes
            dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

            # Fill missing values for Fare (using median values)
            dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
            # create Fare band
            dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
            dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
            dataset['Fare'] = dataset['Fare'].astype(int)
            new_combine.append(dataset)

        self.train_df = new_combine[0]
        self.test_df = new_combine[1]
        return self

    def get_accuracy_model(self, X, Y):
        """
        Get the accuracy of the model and display it
        INPUT:
        - X: the train data
        - Y: the label of train data

        OUTPUT:
        - The best model: best_model (stored in self.best_model)
        - DataFrame the accuracy of model: model_accuracy_all
        """

        all_model_names = []
        score_all = []

        highest_score = 0

        for model in self.models:
            model_name = model.__class__.__name__
            all_model_names.append(model_name)
            model.fit(X, Y)
            accuracy = cross_val_score(model, X, Y, scoring='accuracy', cv=5, n_jobs=-1)
            accuracy = accuracy.mean()
            score_all.append(accuracy)
            if accuracy > highest_score:
                highest_score = accuracy
                best_model = model

        model_accuracy_all = pd.DataFrame({'Model': all_model_names, 'Score': score_all})
        print('_'*70)
        print('The model accuracies of all models:')
        print('_'*70)
        print(model_accuracy_all.sort_values(by='Score', ascending=False))
        self.best_model_ = best_model
        self.highest_score_ = highest_score
        return model_accuracy_all

    def analyze_(self):
        """
        fit the train data using all the models in the self.models
        take the accuracy score to pick the best model.
        """
        train_df, test_df = self.read_data_()
        self.clean_data(train_df, test_df)
        X = self.train_df.drop("Survived",axis=1).copy()
        Y = self.train_df["Survived"]

        model_accuracy_all = self.get_accuracy_model(X, Y)
        # Fit the best model for all of the data
        self.best_model_.fit(X, Y)
        print('_'*70)
        print(f'The best model is {self.best_model_.__class__.__name__}' \
              f' with the accuracy of {self.highest_score_:0.2f} %')
        print('_'*70)
        return self


    def write_output(self):
        """
        Write the submision output:
        INPUT:
        - self
        OUTPUT:
        - the submission csv file
        """
        X_test = self.test_df.copy()
        Y_pred = self.best_model_.predict(X_test)
        submission = pd.DataFrame({
        "PassengerId": self.passenger_id,
        "Survived": Y_pred
        })

        submission.to_csv(self.submission_output, index=False)

def main():
    """
    The main analysis
    """

    titanic = TITATIC()
    titanic.analyze_()
    titanic.write_output()

if __name__ == '__main__':
    main()














