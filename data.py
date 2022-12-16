import pandas as pd


class Data:
    
    def __init__(self):
        
        self.dataset = self._get_data()
    
    def __call__(self):
        
        self.dataset = self._select_data()
        return self.dataset
    
    def _get_data(self):
    
        #import data
        data_training = pd.read_csv('asap-aes/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')    
        valid_set = pd.read_csv('asap-aes/valid_set.tsv', sep='\t', encoding='ISO-8859-1')
        test_set = pd.read_csv('asap-aes/test_set.tsv', sep='\t', encoding='ISO-8859-1')
        
        #import valid_test_score 
        valid_test_score = pd.read_csv('asap-aes/valid_sample_submission_2_column.csv')
        
        #prepare_data
        data_training = data_training[["essay_id", "essay_set", "essay", "domain1_score"]]

        valid_test_score = valid_test_score.rename(columns={"prediction_id":"essay_id"})
        
        valid_set = pd.merge(valid_set, valid_test_score, on='essay_id', how='inner')
        test_set = pd.merge(test_set, valid_test_score, on='essay_id', how='inner')

        valid_test_data = pd.merge(valid_set, test_set, how='outer')

        valid_test_data = valid_test_data[['essay_id','essay_set', 'essay', 'predicted_score']]
        valid_test_data = valid_test_data.rename(columns={'predicted_score': "domain1_score"})
        
        #get final data
        data = pd.merge(data_training, valid_test_data, how='outer')

        data.to_csv('asap-aes/data.csv', index=False)
        
        return data
    
    def _select_data(self):
        essay_sets = list(set(self.dataset["essay_set"]))
        essay_ids = []
        for i in range(1, len(essay_sets)+1):
            score = self.dataset[(self.dataset['essay_set'] == i)]['domain1_score']
            if max(score)>6:
                essay_ids.append(i)
        self.dataset = self.dataset[~self.dataset['essay_set'].isin(essay_ids)]
        self.dataset.reset_index(drop = True, inplace = True)
        return self.dataset
        

if __name__ == '__main__':
    data = Data()()