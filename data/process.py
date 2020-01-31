import pandas as pd
import numpy as np

class Data():
    
    def __init__(self, location):
        
        self.data = pd.DataFrame(pd.read_csv(location))
        
        self.passenger = self.data['PassengerId']
        
        self.droppers = ['Ticket', 'PassengerId']
        
        self.data = self.process_titles()
        
        self.data = self.impute_missing()
        
        self.data = self.process_cabins()
        
        self.data = self.process_dummies()
        
        self.data = self.engineer_features()

        self.data = self.process_age()
        
        self.data = self.drop_cols()
        
    
    def process_titles(self):
        
        df = self.data
        
        df['Titles'] = df['Name'].str.extract('(\w*\.)')
        df.loc[~df['Titles'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.']), 'Titles'] = 'Other'
        df = pd.concat([df, pd.get_dummies(df['Titles'])], axis=1)
        
        self.droppers.extend(['Name', 'Titles'])
        
        return df
    
    def impute_missing(self):
        
        df = self.data
            
        most_common_embarked = df['Embarked'].value_counts().keys()[0]
        
        df['Embarked'] = df['Embarked'].fillna(most_common_embarked)
        
        df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].mean()
        
        return df
    
    def process_age(self):
        
        df = self.data
        
        #mean_age_by_gender = df.groupby('Sex').mean()['Age'].to_dict()
        #null_age = df.loc[df['Age'].isna(), :].Sex.map(mean_age_by_gender)
        #df.loc[null_age.index, 'Age'] = null_age

        t = pd.read_csv('ages.csv')

        t = t.set_index(['Sex', 'Pclass', 'Titles'])
        
        for index, row in df.loc[df['Age'].isna(), :].iterrows():
            
            df.loc[index, 'Age'] = t.loc[(row['Sex'], row['Pclass'], row['Titles']), 'Age']
        
        return df
    
    def process_cabins(self):
        
        df = self.data
        
        cabin_extract = df['Cabin'].str.extract('(\w)\d*$')
        cabin_extract = pd.get_dummies(cabin_extract[0])
        z = dict(zip(cabin_extract.columns.tolist(), [x + '_CABIN' for x in cabin_extract.columns.tolist()]))
        cabin_extract.rename(columns=z, inplace=True)
        df = pd.concat([df, cabin_extract], axis=1)
        
        if 'T_CABIN' in df.columns:
            
            self.droppers.extend(['T_CABIN'])
        
        return df
    
    def process_dummies(self):
        
        df = self.data
        
        pclass = pd.get_dummies(df['Pclass'])
        mapper = dict(zip(pclass.columns.tolist(), ['class_' + str(x) for x in pclass.columns.tolist()]))
        pclass.rename(columns=mapper, inplace=True)
        df = pd.concat([df, pclass], axis=1)
        
        df[['female', 'male']] = pd.get_dummies(df['Sex'])
        
        df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
        
        self.droppers.extend(['Sex', 'Embarked', 'Pclass'])
        
        return df
    
    def engineer_features(self):
        
        df = self.data
        
        df['FamilyAboard'] = df['Parch'] + df['SibSp']
        df['IsAlone'] = df['FamilyAboard'] == 0

        df.loc[:, 'InCabin'] = ~df['Cabin'].isna()

        self.droppers.extend(['Parch', 'SibSp', 'Cabin'])
        
        return df
    
    def drop_cols(self):
        
        return self.data.drop(self.droppers, axis=1)
        
    def return_data(self):
        
        return self.data
    
    def return_prediction_df(self, predictions):
        
        return pd.DataFrame({'PassengerId':self.passenger, 'Survived':predictions})
