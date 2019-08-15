## NLP, Web APIs & Classification
--------
_By: Rachel Koenig_

**Problem Statement**
As a data scientist for the advertising department at reddit, I need to find the most predictive keywords and/or phrases to accurately classify the the dating advice and relationship advice subreddit pages so we can use them to determine which advertisements should populate on each page. Since this is a classification problem, I'll use Logistic Regression & Bayes models.  Misclassifications in this case would be fairly harmless so I will use the accuracy score and a baseline of 63.3% to rate success.
Using TFiDfVectorization, I'll find the feature importance to determine which words have the highest prediction power for the target variables.  If successful, this model could also be used to target other pages that have similar frequency of the same words and phrases. 

**Data Collection**

See dating-advice-scrape and relationship-advice-scrape notebooks for this part.  

I imported requests and used it to scrape two reddit pages.  
Relationship advice: https://www.reddit.com/r/relationship_advice/  
Dating advice: https://www.reddit.com/r/dating_advice/  
I scaped close to 1000 unique posts for each subreddit on my first scrape with the "Hot" filter on. The second scrape, I switched to the "New" filter for both pages and got a little more than 900 each.  
I made sure to discard of duplicate rows by using .drop_duplicates with the 'id' column as a subset.
I set time to 1 second for both scrapes so as not to overload the server.

After turning all the scrapes into DataFrames, I saved them as csvs which can be found in the dataset folder of this repo. 


**Data Cleaning and EDA**

- dropped rows with null self text column becuase those rows are useless to me. 
- combined title and selftext column in to one new all_text columns
- dummied subreddit column to be 'dating_advice': 0, 'relationship_advice': 1 
- exambined distributions of word counts for titles and selftext column per post and compared the two subreddit pages.

**Preprocessing and Modeling**

Found the baseline accuracy score 0.633 which means if I always pick the value that occurs most often, I'll be right 63.3% of the time. 

First attempt: 
logistic regression model with default CountVectorizer paramaters.
train score: 99 | test 75 | cross val 74  
Second attempt: 
tried CountVectorizer with Stemmatizer preprocessing on first set of scraping, pretty bad score with high variance. Train 99%, test 72%
- tried to decrease max features and score got even worse
- tried with lemmatizer preprocessing instead and test score went up to 74% 

Simply increasing the data and stratifying y in my test/train/split increased my cvec test score to 81 and cross val to 80.  
Adding 2 paramaters to my CountVectorizers helped quite a bit. A min_df of 3 and ngram_range of (1,2) increased my test score to 83.2 and cross val to 82.3  However, these score disappeared.

To further iterate on the above well scoring model, I tried a Gridsearch with Pipeline on Logistic Regression and CountVectorizer with this param grid:
'cvec__max_features': [2500, 3000, 4000, 5000],
    'cvec__min_df': [2, 3],
    'cvec__max_df': [.9, .95, .98],
    'cvec__ngram_range': [(1,1), (1,2), (2,3)]
 It found these to be the best params:
{'cvec__max_df': 0.95,
 'cvec__max_features': 4000,
 'cvec__min_df': 3,
 'cvec__ngram_range': (1, 2)}
 My accuracy scores:  
 cross val score 76%
 test of 78%
 
 
tfidf with all default paramaters 
lr.score(X_train_tfidf, y_train)
0.8854568854568855

lr.score(X_test_tfidf, y_test)
0.7699228791773779

cross_val_score(lr, X_train_tfidf, y_train, cv=5).mean()
0.7863572102875827


 TfidfVectorizer 
 (stop_words=['relationship', 'girlfriend', 
                                               'boyfriend', 'friends', 'friends', 
                                               'just', 'like', 'dating', 'know',
                                              'time', 'want', 'really', 'would',
                                              'get', 'feel', 'said', 'things', 'think'],
                                   ngram_range=(1,2),
                                   max_df=0.9,
                                   min_df= 2,
                                   max_features=5000
                                  )

I think Tfidf worked the best to decrease my overfitting due to variance problem because I customized the stop words to take away the ones that were actually too frequent to be predictive. This was a success, however, with more time I probably could've tweaked them a bit more to increase all scores.  
Looking at both the single words and words in groups of two (bigrams) was the best param that gridsearch suggested, however, all of my top most predictive words ended up being uni-grams.  
My original list of features had an abundance of jibberish words and typos.  Minimizing the # of times a word was required to show up to 2, helped get rid of those. 
Gridsearch also suggested 90% max df rate which helped to eliminate oversaturated words as well.
Lastly, setting max features to 5000 decreased cut down my columns to about a quarter of what they were to only focus the most frequently used words of what was left.

**Conclusion and Recommendations**

Even though I would like to have higher train and test scores, I was able to successfully lower the variance and there are definitely several words that have high predictive power, so I think the model is ready to launch a test.  If advertising engagement increases, the same key words could be used to find other potentially lucrative pages. I found it interesting that taking out the overly used words helped with overfitting, but brought the accuracy score down. I think there is probably still room to play around with the paramaters of the Tfidf Vectorizer to see if different stop words make a different or  

Sources:
visualizations: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a

