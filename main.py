import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics



df = pd.read_csv(r"D:\ds2\Tweets.csv")

print("Sample Tweets:")
print(df[['text', 'airline_sentiment']].head())

#Show sentiment distribution
print("\nSentiment Distribution:")
print(df['airline_sentiment'].value_counts())

df = pd.read_csv(r"D:\ds2\Tweets.csv") 

# Split the dataset into training and testing sets
X = df['text']
y = df['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

import joblib
# Save the model to a file
joblib.dump(model, 'sentiment_model.pkl')
