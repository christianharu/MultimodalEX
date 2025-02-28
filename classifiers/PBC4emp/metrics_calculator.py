import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import CEM as cem

import test_BERT as bert_tester


def get_metrics(df, predictions,actual):
    modified_df = df.copy()
    modified_df = modified_df.rename(columns={actual: "empathy"})
    acc = accuracy_score(modified_df["empathy"],modified_df[predictions])
    cem_score = cem.get_cem(modified_df[predictions]-1,modified_df[["empathy"]])
    pre = precision_score(modified_df["empathy"],modified_df[predictions], average='weighted')
    f1 = f1_score(modified_df["empathy"],modified_df[predictions], average='weighted')
    rec = recall_score(modified_df["empathy"],modified_df[predictions], average='weighted')
    return [acc,cem_score,pre,f1,rec,predictions]


current_dir = os.getcwd() #get directory of the repository
data_loc = current_dir + '/empathy_classifier/EmpatheticExchanges/previous_training_predictions/'


testing_df = pd.read_csv(data_loc + 'EmpatheticExchanges_test_3.csv')

print(len(testing_df))

bert_predictions = pd.read_csv(data_loc + 'BERT_predictions_bert_classifier_3_le.txt',header=None)
#bert_predictions[0] += 1

print(bert_predictions)

pbc4cip_predictions = pd.read_csv(data_loc + 'PBC4cip_predictions.txt',header=None)
pbc4cip_predictions[0] += 1

print(len(pbc4cip_predictions))

label_array = testing_df['empathy'].unique()

predictions_bert = []

for pred in bert_predictions[0].tolist():
    predictions_bert.append(label_array[pred])




print(label_array)


testing_df['pbc4cip_predictions'] = pbc4cip_predictions[0]
testing_df['bert_predictions'] = predictions_bert

print(testing_df.head())

metrics_pbc4cip = get_metrics(testing_df, 'pbc4cip_predictions','empathy')
metrics_bert = get_metrics(testing_df,  'bert_predictions','empathy')




metric_to_num = {'accuracy': 0, 'cem': 1, 'precision': 2, 'f1': 3,'recall': 4, 'name': 5}




metrics_list = [metrics_pbc4cip,metrics_bert]

accuracies = []
cems = []
precisions = []
f1s = []
recalls = []
names = []

for item in metrics_list:
    accuracies.append(item[metric_to_num['accuracy']])
    cems.append(item[metric_to_num['cem']])
    precisions.append(item[metric_to_num['precision']])
    f1s.append(item[metric_to_num['f1']])
    recalls.append(item[metric_to_num['recall']])
    names.append(item[metric_to_num['name']])

data = {'name': names, 'accuracy': accuracies, 'cem': cems,'precision': precisions,'f1': f1s,'recall': recalls }
metrics_df = pd.DataFrame.from_dict(data)
metrics_df.to_csv(data_loc+'old_performance_metrics.csv', index=False)