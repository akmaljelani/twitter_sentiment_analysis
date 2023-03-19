import streamlit as st
import streamlit.components.v1 as components
import tweepy
import pandas as pd
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
import seaborn as sns
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import base64
from pathlib import Path
# from PIL import Image
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

## twitter authentication


consumer_key = "Your consumer key here"
consumer_secret = "Your consumer secret key here"
access_token = "Your access token here"
access_token_secret = "Your access token secret here"

# Authenticate the API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Initialize the API
api = tweepy.API(auth)

# Initialize the Vader Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

#Put your Bearer Token in the parenthesis below
client = tweepy.Client(bearer_token='Insert your bearer token here')

# Function to clean tweets
def clean_tweet(tweet):
    # Remove links, special characters, and numbers
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', "", tweet)
    tweet = re.sub(r'\d+', "", tweet)
    # Remove emojis
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')
    return tweet

# Function to get tweet sentiment using Vader
def get_tweet_sentiment(tweet):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(tweet)
    polarity, subjectivity = TextBlob(tweet).sentiment
    if score['compound'] <= -0.05:
        sentiment = 'negative'
    elif score['compound'] >= 0.05:
        sentiment = 'positive'
    else:
        sentiment = 'neutral'
    return pd.Series([polarity, subjectivity, sentiment, score['neg'], score['neu'], score['pos'], score['compound']])



# Function to plot word cloud
def plot_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(stopwords=stopwords, background_color="white", max_words=500, width=800, height=500)
    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    st.pyplot()
    return wc

# Set prevent_rerun=True in Streamlit app configuration
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Streamlit app
st.title("Twitter Search Sentiment Analysis")
st.markdown('This app analyzes the tweets from specified keyword/s. Sentiment Analysis could be used'
            ' as a social listening tool, brand monitoring, product review, customer reviews/complains, trend analysis, competitor analysis etc')




# Get user input
search_term = st.text_input("Enter a search term:")
language = st.selectbox("Select a language:", ("id", "en"))
retweet = st.checkbox("Include retweets")


if st.button("Search"):
    # Construct query
    query = search_term
    if not retweet:
        query += " -filter:retweets"
    if language == "id":
        query += " lang:id"
    else:
        query += " lang:en"
    # Get tweets
    tweets = tweepy.Cursor(api.search_tweets, q=query, tweet_mode="extended", lang=language, result_type="recent").items(500)
    # Create dataframe
    df = pd.DataFrame(columns=["tweet", "clean_tweet", "sentiment"])
    for tweet in tweets:
        df = df.append({"tweet": tweet.full_text, "clean_tweet": clean_tweet(tweet.full_text)}, ignore_index=True)
    # Get tweet sentiments into the df. combining tweets & sentiments
    df[['polarity', 'subjectivity', 'sentiment', 'neg', 'neu', 'pos', 'compound']] = df['clean_tweet'].apply(get_tweet_sentiment)

    num_tweets = len(df)
    # Display dataframe
    st.subheader(f"{num_tweets} tweets mentioning '{search_term}' over the last 7 days")
    st.write(df)

    # Display the results in 3 columns
    # first_column, second_column, third_column = st.columns([0.8, 1, 0.2])
    first_column, second_column = st.columns([1, 1])
    second_row_col_one, second_row_col_two, second_row_col_three = st.columns([0.3, 0.8 ,1])

    # Plot word cloud
    with second_row_col_three:
        st.subheader(f"Wordcloud for '{search_term}'")
        stopwords = set(STOPWORDS)
        # stopwords.update(["http", "https", "co", "amp"])  # add custom stopwords
        text = " ".join(tweet for tweet in df["clean_tweet"] if tweet not in stopwords)
        display_wc = plot_wordcloud(text)


    #number of sentiments
    # sentiment_freq = df['sentiment'].value_counts()
    # st.dataframe(sentiment_freq)

    with first_column:
        df_negative = df[df["sentiment"] == "negative"]
        df_positive = df[df["sentiment"] == "positive"]
        df_neutral = df[df["sentiment"] == "neutral"]


        # Function for count_values_in single columns

        def count_values_in_column(data, feature):
            total = data.loc[:, feature].value_counts(dropna=False)
            percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
            return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


        sentiment_count = count_values_in_column(df, "sentiment")
        # st.dataframe(sentiment_count)
        # fig, ax = plt.subplots()
        # ax.hist(df["sentiment"], bins=3)
        # ax.set_xticks(range(3))
        # ax.set_xticklabels(sentiment_count.index)
        # ax.set_xlabel("Sentiment")
        # ax.set_ylabel("Count")
        # st.pyplot(fig)
        colors = {
            'positive': '#8bc34a',  # green
            'negative': '#ffccbc',  # pastel red
            'neutral': '#fff59d'  # pastel yellow
        }

        st.subheader(f"Number of sentiments for '{search_term}'")

        fig = px.histogram(df, x='sentiment', color="sentiment", color_discrete_map=colors)
        fig.update_layout(height=500, width=400)
        fig.update_layout(xaxis_title='Sentiment', yaxis_title='No of comments')
        st.plotly_chart(fig)


    ##DOWNLOAD SENTIMENT COUNT


    #### CALCULATING WORDFREQUENCY
    # download the stopwords
    nltk.download('stopwords')

    ## IF GOT ADDITIONAL STOPWORDS
    # my_stopwords = {"hi", "pls", "please", "thank", "thanks", "thank u", "thank you", "like"}

    # get the list of stopwords
    # stop_words = set(stopwords.words('english'))

    ## IF GOT ADDITIONAL STOPWORDS
    # stop_words = stop_words.union(my_stopwords)

    with second_row_col_one:
        ""

    with second_row_col_two:
        # define a function to remove stopwords
        def remove_stopwords(text):
            words = [word for word in text.split() if word.lower() not in stopwords]
            return " ".join(words)


        df['clean_tweet'] = df['clean_tweet'].apply(remove_stopwords)

        # create a list of words from the clean_text column
        words = [word for row in df['clean_tweet'] for word in row.split()]

        # calculate the word frequency
        fdist = FreqDist(words)

        # get the top 30 most common words
        top_words = fdist.most_common(100)

        # create a DataFrame from the words and their frequency
        df_freq = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

        st.subheader(f"100 most common words gathered from '{search_term}'")
        st.dataframe(df_freq)

        with second_column:
            # get the top 30 most common words
            st.subheader(f"Top words for '{search_term}'")
            top_words2 = fdist.most_common(20)

            # create a DataFrame from the words and their frequency
            df_freq = pd.DataFrame(top_words2, columns=['Word', 'Frequency'])

            # create the bar plot using Plotly Express
            fig = px.bar(df_freq, x='Word', y='Frequency', color='Frequency', color_continuous_scale='blues')
            fig.update_layout(title='Top 30 Most Common Words')
            st.plotly_chart(fig)

    ### CLEAR ALL
    if st.button('Clear All'):
        st.session_state.pop('df', None)
        st.session_state.pop('plot_wordcloud', None)
        st.session_state.pop('sentiment_freq', None)
        st.session_state.pop('sentiment_count', None)
        st.session_state.pop('df_freq', None)



    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(df)

    st.download_button(
        "Press to Download",
        csv,
        "text_sentiment.csv",
        "text/csv",
        key='download-csv'
    )




