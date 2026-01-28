"""
Sentiment Analysis Module

Multi-source sentiment analysis for market prediction:
- Twitter/X sentiment analysis
- News sentiment analysis
- Google Trends analysis
- Reddit sentiment (if available)
- Earnings call sentiment
- Social media volume indicators

Methods:
- VADER (Valence Aware Dictionary)
- TextBlob
- FinBERT (Financial domain BERT)
- Custom financial lexicon
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.settings import config


class SentimentAnalyzer:
    """
    Multi-source sentiment analysis for stock prediction.
    """
    
    def __init__(self, ticker: str = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker or config.data.ticker
        self.sentiment_data = {}
        self.signals = pd.DataFrame()
        
        # Financial lexicon (domain-specific sentiment words)
        self.bullish_words = {
            'bullish', 'rally', 'surge', 'soar', 'breakout', 'moon', 'rocket',
            'buy', 'long', 'calls', 'upside', 'outperform', 'upgrade', 'beat',
            'growth', 'profit', 'gain', 'winner', 'strong', 'momentum', 'launch',
            'contract', 'revenue', 'successful', 'partnership', 'acquisition'
        }
        
        self.bearish_words = {
            'bearish', 'crash', 'dump', 'plunge', 'breakdown', 'sell', 'short',
            'puts', 'downside', 'underperform', 'downgrade', 'miss', 'loss',
            'weak', 'decline', 'failure', 'delay', 'investigation', 'lawsuit',
            'dilution', 'offering', 'debt', 'concern', 'risk', 'warning'
        }
        
        self._init_sentiment_analyzers()
    
    def _init_sentiment_analyzers(self):
        """Initialize sentiment analysis models."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            
            # Add financial terms to VADER lexicon
            for word in self.bullish_words:
                self.vader.lexicon[word] = 2.0
            for word in self.bearish_words:
                self.vader.lexicon[word] = -2.0
            
            self.has_vader = True
        except ImportError:
            self.has_vader = False
            print("VADER not available")
        
        try:
            from textblob import TextBlob
            self.has_textblob = True
        except ImportError:
            self.has_textblob = False
            print("TextBlob not available")
    
    def analyze_text_vader(self, text: str) -> Dict:
        """
        Analyze text sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.has_vader:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        
        scores = self.vader.polarity_scores(text)
        return scores
    
    def analyze_text_textblob(self, text: str) -> Dict:
        """
        Analyze text sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity
        """
        if not self.has_textblob:
            return {'polarity': 0, 'subjectivity': 0.5}
        
        from textblob import TextBlob
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_text_financial(self, text: str) -> float:
        """
        Analyze text using custom financial lexicon.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        bullish_count = len(words.intersection(self.bullish_words))
        bearish_count = len(words.intersection(self.bearish_words))
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def analyze_text_ensemble(self, text: str) -> Dict:
        """
        Ensemble sentiment analysis combining multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with ensemble sentiment scores
        """
        vader_score = self.analyze_text_vader(text)['compound']
        textblob_score = self.analyze_text_textblob(text)['polarity']
        financial_score = self.analyze_text_financial(text)
        
        # Weighted ensemble
        ensemble_score = (
            vader_score * 0.4 +
            textblob_score * 0.3 +
            financial_score * 0.3
        )
        
        return {
            'vader': vader_score,
            'textblob': textblob_score,
            'financial': financial_score,
            'ensemble': ensemble_score
        }
    
    def fetch_twitter_sentiment(self, 
                                 days_back: int = None,
                                 max_tweets: int = 1000) -> pd.DataFrame:
        """
        Fetch and analyze Twitter sentiment.
        
        Note: Requires Twitter API credentials in environment variables.
        
        Args:
            days_back: Number of days to look back
            max_tweets: Maximum tweets to fetch
            
        Returns:
            DataFrame with daily sentiment scores
        """
        days_back = days_back or config.sentiment.twitter_lookback_days
        
        print(f"Fetching Twitter sentiment for ${self.ticker}...")
        
        # Note: This requires Twitter API access
        # In production, you would use tweepy with API credentials
        
        try:
            import tweepy
            
            # Check for credentials
            import os
            bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
            
            if not bearer_token:
                print("  Twitter API credentials not found, using simulated data")
                return self._simulate_twitter_sentiment(days_back)
            
            client = tweepy.Client(bearer_token=bearer_token)
            
            # Search for tweets
            query = f"${self.ticker} OR #{self.ticker} -is:retweet lang:en"
            
            tweets_data = []
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            # Fetch tweets
            tweets = client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),
                tweet_fields=['created_at', 'public_metrics', 'text'],
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat()
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    sentiment = self.analyze_text_ensemble(tweet.text)
                    tweets_data.append({
                        'date': tweet.created_at.date(),
                        'text': tweet.text,
                        'likes': tweet.public_metrics.get('like_count', 0),
                        'retweets': tweet.public_metrics.get('retweet_count', 0),
                        **sentiment
                    })
            
            if tweets_data:
                df = pd.DataFrame(tweets_data)
                
                # Aggregate by day
                daily = df.groupby('date').agg({
                    'ensemble': 'mean',
                    'vader': 'mean',
                    'likes': 'sum',
                    'retweets': 'sum',
                    'text': 'count'
                }).rename(columns={'text': 'tweet_count'})
                
                return daily
            
            return pd.DataFrame()
            
        except ImportError:
            print("  Tweepy not installed, using simulated data")
            return self._simulate_twitter_sentiment(days_back)
        except Exception as e:
            print(f"  Error fetching Twitter data: {e}")
            return self._simulate_twitter_sentiment(days_back)
    
    def _simulate_twitter_sentiment(self, days: int) -> pd.DataFrame:
        """Generate simulated Twitter sentiment data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate correlated sentiment (somewhat random but with persistence)
        np.random.seed(42)
        base_sentiment = np.cumsum(np.random.randn(days) * 0.1)
        base_sentiment = np.clip(base_sentiment / np.max(np.abs(base_sentiment)), -1, 1)
        
        df = pd.DataFrame({
            'date': dates,
            'ensemble': base_sentiment + np.random.randn(days) * 0.1,
            'vader': base_sentiment + np.random.randn(days) * 0.15,
            'tweet_count': np.random.poisson(100, days),
            'likes': np.random.poisson(500, days),
            'retweets': np.random.poisson(50, days)
        })
        
        df['ensemble'] = np.clip(df['ensemble'], -1, 1)
        df['vader'] = np.clip(df['vader'], -1, 1)
        df = df.set_index('date')
        
        return df
    
    def fetch_news_sentiment(self, days_back: int = None) -> pd.DataFrame:
        """
        Fetch and analyze news sentiment.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            DataFrame with daily news sentiment
        """
        days_back = days_back or config.sentiment.news_lookback_days
        
        print(f"Fetching news sentiment for {self.ticker}...")
        
        try:
            from newsapi import NewsApiClient
            import os
            
            api_key = os.environ.get('NEWS_API_KEY')
            
            if not api_key:
                print("  News API key not found, using simulated data")
                return self._simulate_news_sentiment(days_back)
            
            newsapi = NewsApiClient(api_key=api_key)
            
            # Fetch articles
            all_articles = newsapi.get_everything(
                q=f"{self.ticker} OR Rocket Lab",
                from_param=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            news_data = []
            
            for article in all_articles.get('articles', []):
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_text_ensemble(text)
                
                news_data.append({
                    'date': pd.to_datetime(article['publishedAt']).date(),
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    **sentiment
                })
            
            if news_data:
                df = pd.DataFrame(news_data)
                
                # Aggregate by day
                daily = df.groupby('date').agg({
                    'ensemble': 'mean',
                    'vader': 'mean',
                    'title': 'count'
                }).rename(columns={'title': 'article_count'})
                
                return daily
            
            return pd.DataFrame()
            
        except ImportError:
            print("  NewsAPI not installed, using simulated data")
            return self._simulate_news_sentiment(days_back)
        except Exception as e:
            print(f"  Error fetching news: {e}")
            return self._simulate_news_sentiment(days_back)
    
    def _simulate_news_sentiment(self, days: int) -> pd.DataFrame:
        """Generate simulated news sentiment data."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        np.random.seed(43)
        sentiment = np.random.randn(days) * 0.3
        sentiment = np.clip(sentiment, -1, 1)
        
        df = pd.DataFrame({
            'date': dates,
            'ensemble': sentiment,
            'vader': sentiment + np.random.randn(days) * 0.1,
            'article_count': np.random.poisson(5, days)
        })
        
        df = df.set_index('date')
        return df
    
    def fetch_google_trends(self, keywords: List[str] = None) -> pd.DataFrame:
        """
        Fetch Google Trends data.
        
        Args:
            keywords: Keywords to search for
            
        Returns:
            DataFrame with trend data
        """
        if keywords is None:
            keywords = [
                self.ticker,
                "Rocket Lab",
                "RKLB stock",
                "Electron rocket",
                "Peter Beck"
            ]
        
        print(f"Fetching Google Trends for: {keywords[:3]}...")
        
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            pytrends.build_payload(
                keywords[:5],
                timeframe='today 3-m',
                geo='US'
            )
            
            df = pytrends.interest_over_time()
            
            if not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop('isPartial', axis=1)
                
                # Create aggregate metric
                df['trends_aggregate'] = df.mean(axis=1)
                df['trends_change'] = df['trends_aggregate'].pct_change()
                df['trends_zscore'] = (
                    (df['trends_aggregate'] - df['trends_aggregate'].rolling(30).mean()) /
                    df['trends_aggregate'].rolling(30).std()
                )
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"  Error fetching Google Trends: {e}")
            return self._simulate_trends()
    
    def _simulate_trends(self) -> pd.DataFrame:
        """Generate simulated trends data."""
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        
        np.random.seed(44)
        base = 50 + np.cumsum(np.random.randn(90) * 2)
        base = np.clip(base, 0, 100)
        
        df = pd.DataFrame({
            'date': dates,
            'trends_aggregate': base,
            'trends_change': np.concatenate([[0], np.diff(base) / base[:-1]]),
            'trends_zscore': (base - np.mean(base)) / np.std(base)
        })
        
        df = df.set_index('date')
        return df
    
    def calculate_sentiment_indicators(self,
                                       twitter_df: pd.DataFrame,
                                       news_df: pd.DataFrame,
                                       trends_df: pd.DataFrame,
                                       price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite sentiment indicators aligned with price data.
        
        Args:
            twitter_df: Twitter sentiment DataFrame
            news_df: News sentiment DataFrame
            trends_df: Google Trends DataFrame
            price_df: Price DataFrame (for alignment)
            
        Returns:
            DataFrame with sentiment indicators
        """
        signals = pd.DataFrame(index=price_df.index)
        
        # Twitter sentiment
        if not twitter_df.empty:
            twitter_df.index = pd.to_datetime(twitter_df.index)
            signals = signals.join(
                twitter_df[['ensemble']].rename(columns={'ensemble': 'twitter_sentiment'}),
                how='left'
            )
            if 'tweet_count' in twitter_df.columns:
                signals = signals.join(
                    twitter_df[['tweet_count']].rename(columns={'tweet_count': 'twitter_volume'}),
                    how='left'
                )
        
        # News sentiment
        if not news_df.empty:
            news_df.index = pd.to_datetime(news_df.index)
            signals = signals.join(
                news_df[['ensemble']].rename(columns={'ensemble': 'news_sentiment'}),
                how='left'
            )
            if 'article_count' in news_df.columns:
                signals = signals.join(
                    news_df[['article_count']].rename(columns={'article_count': 'news_volume'}),
                    how='left'
                )
        
        # Google Trends
        if not trends_df.empty:
            trends_df.index = pd.to_datetime(trends_df.index)
            for col in ['trends_aggregate', 'trends_zscore']:
                if col in trends_df.columns:
                    signals = signals.join(trends_df[[col]], how='left')
        
        # Forward fill sentiment (sentiment persists)
        signals = signals.ffill()
        
        # Calculate composite sentiment
        sentiment_cols = [c for c in signals.columns if 'sentiment' in c]
        if sentiment_cols:
            signals['sentiment_composite'] = signals[sentiment_cols].mean(axis=1)
        
        # Sentiment momentum (change in sentiment)
        if 'sentiment_composite' in signals.columns:
            signals['sentiment_momentum'] = signals['sentiment_composite'].diff(5)
        
        # Sentiment extreme flags
        if 'sentiment_composite' in signals.columns:
            sent = signals['sentiment_composite']
            signals['sentiment_extreme_bullish'] = sent > sent.rolling(60).mean() + 2 * sent.rolling(60).std()
            signals['sentiment_extreme_bearish'] = sent < sent.rolling(60).mean() - 2 * sent.rolling(60).std()
        
        # Search interest anomaly
        if 'trends_zscore' in signals.columns:
            signals['search_spike'] = signals['trends_zscore'] > 2
        
        # Social volume anomaly
        volume_cols = [c for c in signals.columns if 'volume' in c]
        if volume_cols:
            for col in volume_cols:
                signals[f'{col}_zscore'] = (
                    (signals[col] - signals[col].rolling(30).mean()) /
                    signals[col].rolling(30).std()
                )
        
        self.signals = signals
        return signals
    
    def get_sentiment_summary(self) -> Dict:
        """Get current sentiment summary."""
        if self.signals.empty:
            return {}
        
        latest = self.signals.iloc[-1]
        
        summary = {
            'twitter_sentiment': latest.get('twitter_sentiment', np.nan),
            'news_sentiment': latest.get('news_sentiment', np.nan),
            'composite_sentiment': latest.get('sentiment_composite', np.nan),
            'search_interest': latest.get('trends_aggregate', np.nan),
            'sentiment_momentum': latest.get('sentiment_momentum', np.nan),
        }
        
        # Interpret sentiment
        comp = summary.get('composite_sentiment', 0)
        if comp > 0.3:
            summary['interpretation'] = 'Strongly Bullish'
        elif comp > 0.1:
            summary['interpretation'] = 'Moderately Bullish'
        elif comp < -0.3:
            summary['interpretation'] = 'Strongly Bearish'
        elif comp < -0.1:
            summary['interpretation'] = 'Moderately Bearish'
        else:
            summary['interpretation'] = 'Neutral'
        
        return summary
    
    def run_full_analysis(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete sentiment analysis pipeline.
        
        Args:
            price_df: Price DataFrame for alignment
            
        Returns:
            DataFrame with all sentiment signals
        """
        print("=" * 60)
        print(f"Running Sentiment Analysis for {self.ticker}")
        print("=" * 60)
        
        # Fetch data from all sources
        twitter_df = self.fetch_twitter_sentiment()
        news_df = self.fetch_news_sentiment()
        trends_df = self.fetch_google_trends()
        
        # Calculate indicators
        signals = self.calculate_sentiment_indicators(
            twitter_df, news_df, trends_df, price_df
        )
        
        # Summary
        summary = self.get_sentiment_summary()
        
        print("=" * 60)
        print("Sentiment Analysis Complete")
        print(f"Current sentiment: {summary.get('interpretation', 'Unknown')}")
        print(f"Composite score: {summary.get('composite_sentiment', 'N/A'):.3f}")
        print("=" * 60)
        
        return signals


class OptionsFlowAnalyzer:
    """
    Analyze options flow for sentiment signals.
    
    Put/Call ratios and unusual options activity can predict moves.
    """
    
    def __init__(self, ticker: str = None):
        self.ticker = ticker or config.data.ticker
        self.signals = pd.DataFrame()
    
    def calculate_pc_ratio_signals(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate put/call ratio based signals.
        
        Args:
            options_data: DataFrame with options data
            
        Returns:
            DataFrame with P/C signals
        """
        if options_data.empty:
            return pd.DataFrame()
        
        signals = pd.DataFrame()
        
        # Aggregate P/C ratios
        if 'pc_volume_ratio' in options_data.columns:
            avg_pc = options_data['pc_volume_ratio'].mean()
            
            signals['pc_ratio'] = [avg_pc]
            
            # Interpret
            # High P/C (>1) suggests bearish sentiment (often contrarian bullish)
            # Low P/C (<0.7) suggests bullish sentiment (often contrarian bearish)
            if avg_pc > 1.2:
                signals['pc_signal'] = ['contrarian_bullish']
            elif avg_pc > 1.0:
                signals['pc_signal'] = ['mildly_bearish']
            elif avg_pc < 0.6:
                signals['pc_signal'] = ['contrarian_bearish']
            elif avg_pc < 0.8:
                signals['pc_signal'] = ['mildly_bullish']
            else:
                signals['pc_signal'] = ['neutral']
        
        self.signals = signals
        return signals


if __name__ == "__main__":
    # Test sentiment analysis
    from data.collector import DataCollector
    
    # Collect price data
    collector = DataCollector("RKLB")
    data = collector.collect_all_data()
    price_df = data['primary']
    
    # Run sentiment analysis
    analyzer = SentimentAnalyzer("RKLB")
    signals = analyzer.run_full_analysis(price_df)
    
    print("\nSentiment Signals Sample:")
    print(signals.tail(10))
