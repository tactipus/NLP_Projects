import pandas as pd
import numpy as np

# misc
import datetime as dt
from pprint import pprint
from itertools import chain
import os
import emoji
import en_core_web_sm
import spacy

# reddit crawler
import praw
from praw.models import MoreComments

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.stem import WordNetLemmatizer
from nltk.stem import  PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer # tokenize words
from nltk.corpus import stopwords
from nltk import FreqDist

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='Dark2')
from wordcloud import WordCloud
import plotly.express as px

# initializing the crawler
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')

r = praw.Reddit(
    client_id='qfci4cABi7N-VdIw46gvpw', \
    client_secret='ZEQgPgJjExOUT-mK9SSqcuqxb4SKhQ', \
    user_agent='Sentiment_B/X.X.X', \
    username='l3ssthanher0', \
    password='Up$hut243546', \
    check_for_async=False)

# subreddit = r.subreddit('NPD')
# for submission in subreddit.hot(limit=10):
#     print(submission.title)
#     print("Submission ID = ", submission.id, '\n')


post1 = r.submission(id='lkodju')
comments_all = []
post1.comments.replace_more(limit=None)
for comments in post1.comments.list():
    comments_all.append(comments.body)

# print(comments_all, '\n')
print('Total Comments Scraped = ', (len(comments_all)))

list1 = comments_all
list1 = [str(i) for i in list1]
string_uncleaned = ','.join(list1)
# print(string_uncleaned)
string_emojiless = emoji.replace_emoji(string_uncleaned, replace='')
# print(string_emojiless)

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
tokenized_string = tokenizer.tokenize(string_emojiless)
# print(tokenized_string)
lower_string_tokenized = [word.lower() for word in tokenized_string]
# print(lower_string_tokenized)

# removing stopwords
nlp = en_core_web_sm.load()

all_stopwords = nlp.Defaults.stop_words

text = lower_string_tokenized
tokens_without_sw = [word for word in text if not word in all_stopwords]

# print(tokens_without_sw)

#lemmatize the words
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])
# print(lemmatized_tokens)

cleaned_output = lemmatized_tokens

#apply sentiment analysis
sia = sia()
results = []

for sentences in cleaned_output:
    pol_score = sia.polarity_scores(sentences)
    pol_score['words'] = sentences
    results.append(pol_score)

pd.set_option('display.max_columns', None, 'max_colwidth', None)
df = pd.DataFrame.from_records(results)
# pprint(df)

df['label'] = 0
df.loc[df['compound'] > 0.10, 'label'] = 1
df.loc[df['compound'] < -0.10, 'label'] = -1

#removal of neutral words
df_pos_neg = df.loc[df['label'] != 0]

# pprint(df.label.value_counts())

#representation of word sentiment
fig, ax = plt.subplots(figsize=(8, 8))
counts = df_pos_neg.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_ylabel("Percentage")

# plt.show()

#frequency distribution of most common and negative words
pos_words = list(df.loc[df['label'] == 1].words)
pos_freq = FreqDist(pos_words)
pos_freq = pos_freq.most_common(20)

neg_words = list(df.loc[df['label'] == -1].words)
neg_freq = FreqDist(neg_words)
neg_freq = neg_freq.most_common(20)

#visualize through WordCloud
pos_words1 = [str(p) for p in pos_freq]
pos_words_string = ",".join(pos_words1)

neg_words1 = [str(p) for p in neg_freq]
neg_words_string = ",".join(neg_words1)

wordcloud_pos = WordCloud(background_color='white').generate(pos_words_string)
wordcloud_neg = WordCloud().generate(neg_words_string)

plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.show()

plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.show()

#visualize through bar chart
pos_freq_df = pd.DataFrame(pos_freq)
pos_freq_df = pos_freq_df.rename(columns={0: "Bar Graph of Frequent Words", 1: "Count"}, inplace=False)

fig = px.bar(pos_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Positive Words')
fig.show()

neg_freq_df = pd.DataFrame(neg_freq)
neg_freq_df = neg_freq_df.rename(columns={0: "Bar Graph of Frequent Words", 1: "Count"}, inplace=False)

fig = px.bar(neg_freq_df, x='Bar Graph of Frequent Words', y='Count', title='Commonly Used Negative Words')
fig.show()



### From another tutorial ###
# subreddit = r.subreddit('NPD')
# top = [*subreddit.top(limit=None)]
# body = [top.selftext for top in top]
# top = pd.DataFrame({"body": body})
#
# sid = sia()
# res = [*top['body'].apply(sid.polarity_scores)]
#
# sentiment_df = pd.DataFrame.from_records(res)
# top = pd.concat([top, sentiment_df], axis=1, join='inner')
#
# THRESHOLD = 0.2
#
# conditions = [
#     (top['compound'] <= -THRESHOLD),
#     (top['compound'] > -THRESHOLD) & (top['compound'] < THRESHOLD),
#     (top['compound'] >= THRESHOLD),
# ]
#
# values = ['neg', 'neu', 'pos']
# top['label'] = np.select(conditions, values)
#
# # sentence0 = top.body.iloc[0]
# # print(sentence0)
# # words0 = top.body.iloc[0].split()
# # print(words0)
# #
# # pos_list, neg_list, neu_list = [], [], []
# #
# # for word in words0:
# #     if (sid.polarity_scores(word)['compound']) >= THRESHOLD:
# #         pos_list.append(word)
# #     elif (sid.polarity_scores(word)['compound']) <= -THRESHOLD:
# #         neg_list.append(word)
# #     else:
# #         neu_list.append(word)
# #
# # print('\nPositive:',pos_list)
# # print('Neutral:',neu_list)
# # print('Negative:',neg_list)
# # score = sid.polarity_scores(sentence0)
# #
# # print(f"\nThis sentence is {round(score['neg'] * 100, 2)}% negative")
# # print(f"This sentence is {round(score['neu'] * 100, 2)}% neutral")
# # print(f"This sentence is {round(score['pos'] * 100, 2)}% positive")
# # print(f"The compound value : {score['compound']} <= {-THRESHOLD}")
# # print(f"\nThis sentence is NEGATIVE")
#
# # pprint(top.label.value_counts())
#
#
# def top_title_output(df, label):
#     res = df[df['label'] == label].body.values
#     print(f'{"=" * 20}')
#     print("\n".join(title for title in res))
#
#
# # randomly sample
# top_sub = top.groupby('label').sample(n = 5, random_state = 7)
#
# # print("Positive news")
# # top_title_output(top_sub, "pos")
# #
# # print("\nNeutral news")
# # top_title_output(top_sub, "neu")
# #
# # print("\nNegative news")
# # top_title_output(top_sub, "neg")
#
#
#
# stop_words = stopwords.words('english')
#
#
# def custom_tokenize(text):
#     # remove single quote and dashes
#     # text = text.replace("'", "").replace("-", "").lower()
#
#     # split on words only
#     tk = nltk.tokenize.RegexpTokenizer(r'\w+')
#     tokens = tk.tokenize(text)
#
#     # remove stop words
#     words = [w for w in tokens if not w in stop_words]
#     return words
#
#
# def tokens_2_words(df, label):
#     # subset titles based on label
#     bodies = df[df['label'] == label].body
#     # apply our custom tokenize function to each title
#     tokens = bodies.apply(custom_tokenize)
#     # join nested lists into a single list
#     words = list(chain.from_iterable(tokens))
#     return words
#
#
# pos_words = tokens_2_words(top, 'pos')
# neg_words = tokens_2_words(top, 'neg')
#
# pos_freq = nltk.FreqDist(pos_words)
# # pprint(pos_freq.most_common(20))
#
# neg_freq = nltk.FreqDist(neg_words)
# # pprint(neg_freq.most_common(20))
#
# # pprint()
#
#
# wordcloud = WordCloud().generate(' '.join(pos_words))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()