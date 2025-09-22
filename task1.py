import pandas as pd
import math
from sklearn.metrics import classification_report, accuracy_score

train_path = "sentiment_dataset/twitter_training.csv"
valid_path = "sentiment_dataset/twitter_validation.csv"

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

train_df = train_df[train_df.iloc[:, 2].isin(["Positive", "Negative", "Neutral"])]
valid_df = valid_df[valid_df.iloc[:, 2].isin(["Positive", "Negative", "Neutral"])]

X_train, y_train = train_df.iloc[:, 3], train_df.iloc[:, 2]
X_valid, y_valid = valid_df.iloc[:, 3], valid_df.iloc[:, 2]

short_lexicon = {
    "good": 2.0, "great": 3.0, "happy": 2.5, "love": 3.2, "excellent": 3.5,
    "fantastic": 3.5, "amazing": 3.2, "wonderful": 3.4, "awesome": 3.0, "nice": 2.0,
    "bad": -2.0, "terrible": -3.5, "hate": -3.2, "awful": -3.4, "horrible": -3.5,
    "worst": -3.6, "sad": -2.5, "angry": -2.8, "disappointing": -2.7, "poor": -2.0,
}

class SimpleSentimentIntensityAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon
    
    def polarity_scores(self, text):
        words = str(text).lower().split()
        score = 0.0
        for w in words:
            score += self.lexicon.get(w, 0.0)
        if len(words) > 0:
            score = score / math.sqrt(len(words))
        return {"compound": score}

simple_sia = SimpleSentimentIntensityAnalyzer(short_lexicon)

def simple_vader_sentiment(text):
    score = simple_sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

y_pred_simple = X_valid.apply(simple_vader_sentiment)

print("Accuracy:", accuracy_score(y_valid, y_pred_simple))
print(classification_report(y_valid, y_pred_simple))
