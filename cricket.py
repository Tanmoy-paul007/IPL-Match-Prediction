import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# ম্যাচ ও বল-বল ভিত্তিক ডেটাসেট লোড করা
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# প্রথম ইনিংসের মোট রান বের করা
total_score_df = delivery.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
total_score_df['total_runs'] += 1  # প্রতিটি ইনিংসে ১ রান যোগ

total_score_df = total_score_df[total_score_df['inning'] == 1]  # শুধু প্রথম ইনিংস রাখা

# ম্যাচের ডেটার সাথে প্রথম ইনিংসের রান যুক্ত করা
match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# টিমের নাম একরকম করা যাতে বিশ্লেষণ সহজ হয়
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

# পুরোনো টিমের নামকে নতুন নামে রূপান্তর
match_df['team1'] = match_df['team1'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})
match_df['team2'] = match_df['team2'].replace({'Delhi Daredevils': 'Delhi Capitals', 'Deccan Chargers': 'Sunrisers Hyderabad'})

# শুধু নির্ধারিত টিমগুলোর মধ্যকার ম্যাচ রাখা
match_df = match_df[match_df['team1'].isin(teams) & match_df['team2'].isin(teams)]

# DL মেথড প্রয়োগ না হওয়া ম্যাচগুলো রাখা
match_df = match_df[match_df['dl_applied'] == 0]

# প্রয়োজনীয় কলামগুলো রাখা
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

# ডেলিভারি ডেটার সাথে ম্যাচ ডেটা যুক্ত করা, এবং শুধু দ্বিতীয় ইনিংসের ডেটা রাখা
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

# দ্বিতীয় ইনিংসে প্রতি বলের পর মোট কত রান হয়েছে সেটা হিসেব করা
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

# কত বল বাকি আছে সেটা হিসাব করা
delivery_df['balls_left'] = 120 - ((delivery_df['over'] * 6) + delivery_df['ball'] - 1)

# আউট হয়েছে কিনা সেটা 0 বা 1 এ রূপান্তর করা
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

# প্রতিটি ম্যাচে এখন পর্যন্ত কতো উইকেট পড়েছে তা হিসাব করা
delivery_df['wickets'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - delivery_df['wickets']

# চূড়ান্ত আউটপুট
delivery_df.tail()

# লক্ষ্য থেকে কত রান বাকি আছে তা নির্ধারণ করা
delivery_df['runs_left'] = delivery_df['total_runs'] - delivery_df['current_score']

# CRR হিসাব
delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])

# RRR হিসাব
delivery_df['rrr'] = delivery_df.apply(
    lambda row: (row['runs_left'] * 6) / row['balls_left'] if row['balls_left'] != 0 else 0,
    axis=1
)

#  Batting team r winner team ak e hole result 1 hobe
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

# result delevary korbe tai 
delivery_df['result'] = delivery_df.apply(result,axis=1)

# amr proyojon moto column nam onushare shaajisy
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]

# ১৬. ডেটা ক্লিন ও র‍্যান্ডমভাবে সাজানো
final_df = final_df.sample(final_df.shape[0])
final_df.sample()
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

# ১৭. ইনপুট (X) ও আউটপুট (Y) ভাগ করা
X = final_df.iloc[:, :-1]
Y = final_df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# ১৮. কলাম ট্রান্সফরমার ও পাইপলাইন তৈরি
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

# ১৯. মডেল ট্রেইন করা
pipe.fit(X_train, Y_train)

y_pred = pipe.predict(X_test)

# ২০. মডেলের একিউরেসি চেক করা
accuracy_score(Y_test,y_pred)

pipe.predict_proba(X_test)[10]

def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    
temp_df,target = match_progression(delivery_df,7,pipe)
temp_df

plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))

delivery_df['city'].unique()

pickle.dump(pipe,open('pipe.pkl','wb'))
