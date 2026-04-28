import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

FEEDS = [
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://news.bitcoin.com/feed/",
]

def get_news_sentiment(limit=30):
    scores = []
    headlines = []
    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:limit]:
            title = entry.title
            s = analyzer.polarity_scores(title)['compound']
            scores.append(s)
            headlines.append((title, s))
    avg = sum(scores) / len(scores) if scores else 0
    return avg, headlines

if __name__ == "__main__":
    avg, heads = get_news_sentiment()
    print(f"Avg sentiment: {avg:.3f}")
    for h, s in heads[:5]:
        print(f"{s:+.2f} | {h}")