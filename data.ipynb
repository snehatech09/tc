# pip3 install pandas
# pip3 install nltk
import nltk
import pandas as pd

INPUT_CSV_FILE = 'data.csv'

df = pd.read_csv(INPUT_CSV_FILE, sep=';')

## format price
df['Challenge Stats First Place Prize'] = df['Challenge Stats First Place Prize'].apply(lambda x: int(x.split(',')[0].replace('.','')))
df['Challenge Stats Total Prize']  = df['Challenge Stats Total Prize'].apply(lambda x: int(x.split(',')[0].replace('.','')))

lable = df[['Challenge Stats Technology List', 'Challenge Stats Challenge Name', 'Challenge Stats Challenge Copilot', 'Challenge Stats Project Category Name', 'Challenge Stats First Place Prize', 'Challenge Stats Total Prize']].to_dict(orient='records')
result = df['Challenge Stats Num Valid Submissions'].tolist()

data = zip(lable, result)
classifier = nltk.NaiveBayesClassifier.train(data)

SAMPLE_CHALLNGE = {'Challenge Stats Technology List': 'Node.js', 'Challenge Stats Challenge Name': '96hours', 'Challenge Stats Challenge Copilot': 'copilot-a', 'Challenge Stats Project Category Name': 'code', 'Challenge Stats First Place Prize': 1000, 'Challenge Stats Total Prize':2000}
print(classifier.classify(SAMPLE_CHALLNGE))