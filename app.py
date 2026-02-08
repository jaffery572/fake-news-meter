"""
FAKE NEWS DETECTOR 2.0
Advanced AI-powered misinformation detection system
Combines multiple verification techniques for high accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import re
import json
import datetime
import time
from textblob import TextBlob
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    import nltk
    nltk.download('vader_lexicon')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .credibility-excellent {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .credibility-good {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .credibility-warning {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .credibility-danger {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .analysis-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4A90E2;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        color: #856404;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(255, 65, 108, 0.3);
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px 15px;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 15px;
    }
    
    .tab-content {
        padding: 20px 0;
    }
    
    .news-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .news-card:hover {
        border-color: #4A90E2;
        box-shadow: 0 5px 20px rgba(74, 144, 226, 0.1);
    }
    
    .source-trustworthy {
        background: #d4edda;
        color: #155724;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    
    .source-unreliable {
        background: #f8d7da;
        color: #721c24;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    
    .source-unknown {
        background: #fff3cd;
        color: #856404;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    """Main AI-powered fake news detection system"""
    
    def __init__(self):
        # Initialize components
        self.trusted_domains = self._load_trusted_domains()
        self.unreliable_domains = self._load_unreliable_domains()
        self.sia = SentimentIntensityAnalyzer()
        
        # Load sentence transformer for semantic analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.sentence_model = None
        
        # Load sensational phrases
        self.sensational_phrases = [
            'shocking', 'you won\'t believe', 'mind-blowing', 'breaking',
            'urgent', 'alert', 'must read', 'viral', 'secret',
            'they don\'t want you to know', 'exposed', 'leaked',
            'finally revealed', 'the truth about', 'what they\'re not telling you'
        ]
        
        # Load emotional manipulation phrases
        self.emotional_phrases = [
            'outrageous', 'disgusting', 'heartbreaking', 'terrifying',
            'enraging', 'shameful', 'appalling', 'horrifying'
        ]
        
        # Initialize fact-check cache
        self.fact_check_cache = {}
    
    def _load_trusted_domains(self):
        """Load list of trusted news domains"""
        trusted = [
            'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'wsj.com', 'economist.com', 'bloomberg.com',
            'npr.org', 'pbs.org', 'c-span.org',
            'sciencemag.org', 'nature.com', 'lancet.com',
            'who.int', 'cdc.gov', 'nih.gov', 'europa.eu',
            'un.org', 'worldbank.org', 'imf.org'
        ]
        return set(trusted)
    
    def _load_unreliable_domains(self):
        """Load list of known unreliable domains"""
        unreliable = [
            'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
            'worldtruth.tv', 'humansarefree.com', 'veteranstoday.com',
            'yournewswire.com', 'wakingtimes.com', 'activistpost.com',
            'collective-evolution.com'
        ]
        return set(unreliable)
    
    def analyze_text(self, text):
        """Comprehensive text analysis for fake news indicators"""
        if not text or len(text.strip()) < 50:
            return {'error': 'Text too short for analysis'}
        
        analysis = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentences': len(re.findall(r'[.!?]+', text)),
            'readability_score': self._calculate_readability(text),
            'sensational_score': 0,
            'emotional_score': 0,
            'clickbait_score': 0,
            'subjectivity_score': 0,
            'polarity_score': 0,
            'all_caps_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'common_phrases_found': [],
            'emotional_phrases_found': []
        }
        
        text_lower = text.lower()
        
        # Check for sensational phrases
        for phrase in self.sensational_phrases:
            if phrase in text_lower:
                analysis['common_phrases_found'].append(phrase)
                analysis['sensational_score'] += 1
        
        # Check for emotional manipulation
        for phrase in self.emotional_phrases:
            if phrase in text_lower:
                analysis['emotional_phrases_found'].append(phrase)
                analysis['emotional_score'] += 1
        
        # Count excessive punctuation
        analysis['all_caps_count'] = len(re.findall(r'\b[A-Z]{3,}\b', text))
        analysis['exclamation_count'] = text.count('!')
        analysis['question_count'] = text.count('?')
        
        # Calculate clickbait score
        analysis['clickbait_score'] = self._calculate_clickbait_score(text)
        
        # Sentiment analysis
        blob = TextBlob(text)
        analysis['subjectivity_score'] = blob.sentiment.subjectivity
        analysis['polarity_score'] = blob.sentiment.polarity
        
        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)
        analysis['vader_compound'] = vader_scores['compound']
        analysis['vader_neg'] = vader_scores['neg']
        analysis['vader_neu'] = vader_scores['neu']
        analysis['vader_pos'] = vader_scores['pos']
        
        # Normalize scores
        analysis['sensational_score'] = min(10, analysis['sensational_score']) / 10
        analysis['emotional_score'] = min(10, analysis['emotional_score']) / 10
        analysis['clickbait_score'] = min(1.0, analysis['clickbait_score'])
        
        return analysis
    
    def _calculate_readability(self, text):
        """Calculate Flesch Reading Ease score"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = len(re.findall(r'[aeiouy]+', text.lower()))
        
        if sentences == 0 or words == 0:
            return 0
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    
    def _calculate_clickbait_score(self, text):
        """Calculate likelihood of clickbait"""
        score = 0
        
        # Check for common clickbait patterns
        patterns = [
            (r'you won\'t believe', 0.3),
            (r'shocking', 0.2),
            (r'this will blow your mind', 0.4),
            (r'what happens next', 0.3),
            (r'number \d+ will shock you', 0.5),
            (r'viral', 0.2),
            (r'breaking', 0.1),
            (r'urgent', 0.2),
            (r'alert', 0.2),
            (r'secret', 0.3)
        ]
        
        text_lower = text.lower()
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                score += weight
        
        # Check for excessive punctuation
        if text.count('!') > 3:
            score += min(0.5, text.count('!') * 0.1)
        
        # Check for ALL CAPS words
        all_caps = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(all_caps) > 2:
            score += min(0.3, len(all_caps) * 0.05)
        
        return min(1.0, score)
    
    def analyze_source(self, url):
        """Analyze the credibility of the news source"""
        if not url:
            return {'error': 'No URL provided'}
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            analysis = {
                'domain': domain,
                'is_trusted': domain in self.trusted_domains,
                'is_unreliable': domain in self.unreliable_domains,
                'domain_age': 'unknown',
                'ssl_secured': url.startswith('https'),
                'url_structure_score': self._analyze_url_structure(url),
                'source_reputation': 0.5
            }
            
            # Determine source reputation
            if analysis['is_trusted']:
                analysis['source_reputation'] = 0.9
                analysis['source_category'] = 'Trusted Source'
            elif analysis['is_unreliable']:
                analysis['source_reputation'] = 0.1
                analysis['source_category'] = 'Known Unreliable'
            elif '.gov' in domain or '.edu' in domain:
                analysis['source_reputation'] = 0.8
                analysis['source_category'] = 'Government/Education'
            elif '.org' in domain:
                analysis['source_reputation'] = 0.6
                analysis['source_category'] = 'Organization'
            elif '.com' in domain or '.net' in domain:
                analysis['source_reputation'] = 0.4
                analysis['source_category'] = 'Commercial'
            else:
                analysis['source_reputation'] = 0.3
                analysis['source_category'] = 'Unknown'
            
            # Additional URL analysis
            analysis['url_length'] = len(url)
            analysis['has_subdomain'] = len(domain.split('.')) > 2
            
            return analysis
            
        except Exception as e:
            return {'error': f'URL analysis failed: {str(e)}'}
    
    def _analyze_url_structure(self, url):
        """Analyze URL structure for suspicious patterns"""
        score = 0.5  # Neutral start
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\d{8,}',  # Many numbers
            r'[a-f0-9]{16,}',  # Hex-like strings
            r'click', r'share', r'viral', r'breaking',
            r'free', r'win', r'prize', r'offer'
        ]
        
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower):
                score -= 0.1
        
        # Check URL length
        if len(url) > 100:
            score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def check_fact_claims(self, text):
        """Check for verifiable fact claims in text"""
        # This is a simplified version. In production, you'd use fact-checking APIs
        # like Google Fact Check Tools API, ClaimBuster API, etc.
        
        # Common misinformation patterns
        misinformation_patterns = [
            (r'\bcure\b.*\bcancer\b', 0.8, 'Medical misinformation'),
            (r'\bsecret\b.*\bgovernment\b', 0.7, 'Conspiracy theory'),
            (r'\b100%\b.*\beffective\b', 0.6, 'Exaggerated claim'),
            (r'\bthey\b.*\bhiding\b', 0.7, 'Conspiracy language'),
            (r'\bproven\b.*\bscientifically\b', 0.5, 'Overstated evidence'),
            (r'\beveryone knows\b', 0.6, 'Bandwagon fallacy'),
            (r'\bindisputable fact\b', 0.7, 'False certainty'),
            (r'\bdo your own research\b', 0.5, 'Anti-expert rhetoric')
        ]
        
        claims = []
        text_lower = text.lower()
        
        for pattern, confidence, label in misinformation_patterns:
            if re.search(pattern, text_lower):
                claims.append({
                    'claim': re.search(pattern, text_lower).group(0),
                    'confidence': confidence,
                    'label': label,
                    'type': 'potential_misinformation'
                })
        
        # Check for verifiable statements (simplified)
        verifiable_patterns = [
            r'\b\d+\b.*percent', r'\b\d+\b.*years', r'\bstudies show\b',
            r'\bresearch proves\b', r'\bscientists say\b', r'\baccording to\b'
        ]
        
        for pattern in verifiable_patterns:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower).group(0)
                if match not in [c['claim'] for c in claims]:
                    claims.append({
                        'claim': match,
                        'confidence': 0.3,
                        'label': 'Verifiable claim',
                        'type': 'verifiable_statement'
                    })
        
        return {
            'total_claims': len(claims),
            'misinformation_claims': len([c for c in claims if c['type'] == 'potential_misinformation']),
            'verifiable_claims': len([c for c in claims if c['type'] == 'verifiable_statement']),
            'claims': claims
        }
    
    def get_cross_references(self, keywords, max_results=5):
        """Get cross-references from reliable sources (simulated)"""
        # In production, use NewsAPI, Google News API, or similar
        
        # Simulated reliable sources for demo
        reliable_sources = [
            {
                'title': 'Reuters Fact Check',
                'url': 'https://www.reuters.com/fact-check/',
                'snippet': 'Reuters maintains a dedicated fact-checking team that verifies viral claims.',
                'relevance': 0.9
            },
            {
                'title': 'AP News Fact Check',
                'url': 'https://apnews.com/hub/fact-checking',
                'snippet': 'Associated Press fact-checking of political claims and viral misinformation.',
                'relevance': 0.9
            },
            {
                'title': 'BBC Reality Check',
                'url': 'https://www.bbc.com/news/reality_check',
                'snippet': 'BBC team dedicated to verifying claims, statistics, and viral stories.',
                'relevance': 0.8
            },
            {
                'title': 'PolitiFact',
                'url': 'https://www.politifact.com/',
                'snippet': 'Pulitzer Prize-winning fact-checking website rating claims accuracy.',
                'relevance': 0.8
            },
            {
                'title': 'Snopes',
                'url': 'https://www.snopes.com/',
                'snippet': 'Oldest and largest fact-checking site investigating urban legends and rumors.',
                'relevance': 0.7
            }
        ]
        
        # Filter by keyword relevance (simplified)
        keyword_str = ' '.join(keywords).lower()
        results = []
        
        for source in reliable_sources:
            # Simple keyword matching
            relevance = 0.5
            for keyword in keywords:
                if keyword.lower() in source['title'].lower() or keyword.lower() in source['snippet'].lower():
                    relevance = min(1.0, relevance + 0.2)
            
            if relevance > 0.4:
                results.append({**source, 'relevance': relevance})
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True)[:max_results]
    
    def calculate_overall_credibility(self, text_analysis, source_analysis, fact_analysis):
        """Calculate overall credibility score from all analyses"""
        
        weights = {
            'source_reputation': 0.4,
            'sensational_score': 0.2,
            'clickbait_score': 0.15,
            'emotional_score': 0.1,
            'misinformation_claims': 0.15
        }
        
        # Extract scores
        source_score = source_analysis.get('source_reputation', 0.5)
        sensational_score = 1 - text_analysis.get('sensational_score', 0)
        clickbait_score = 1 - text_analysis.get('clickbait_score', 0)
        emotional_score = 1 - text_analysis.get('emotional_score', 0)
        
        # Normalize misinformation claims
        misinfo_penalty = min(1.0, fact_analysis.get('misinformation_claims', 0) / 5)
        misinfo_score = 1 - misinfo_penalty
        
        # Calculate weighted score
        overall_score = (
            weights['source_reputation'] * source_score +
            weights['sensational_score'] * sensational_score +
            weights['clickbait_score'] * clickbait_score +
            weights['emotional_score'] * emotional_score +
            weights['misinformation_claims'] * misinfo_score
        )
        
        # Adjust based on additional factors
        if 'error' not in text_analysis:
            # Penalize excessive punctuation
            if text_analysis.get('exclamation_count', 0) > 5:
                overall_score -= 0.1
            
            # Penalize ALL CAPS
            if text_analysis.get('all_caps_count', 0) > 3:
                overall_score -= 0.05
        
        # Ensure score is within bounds
        overall_score = max(0.1, min(0.99, overall_score))
        
        # Determine credibility level
        if overall_score >= 0.8:
            credibility_level = 'Highly Credible'
            color = 'green'
        elif overall_score >= 0.6:
            credibility_level = 'Mostly Credible'
            color = 'blue'
        elif overall_score >= 0.4:
            credibility_level = 'Questionable'
            color = 'orange'
        else:
            credibility_level = 'Likely False/Misleading'
            color = 'red'
        
        return {
            'overall_score': overall_score,
            'credibility_level': credibility_level,
            'color': color,
            'component_scores': {
                'source_reputation': source_score,
                'sensational_score': sensational_score,
                'clickbait_score': clickbait_score,
                'emotional_score': emotional_score,
                'misinformation_score': misinfo_score
            }
        }
    
    def extract_article_from_url(self, url):
        """Extract article text from URL (simplified version)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.decompose()
            
            # Try to find article content
            article = soup.find('article')
            if article:
                text = article.get_text()
            else:
                # Fallback to body text
                text = soup.find('body').get_text()
            
            # Clean text
            text = ' '.join(text.split())
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else "No title found"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content') if meta_desc else ""
            
            return {
                'success': True,
                'title': title_text,
                'description': description,
                'text': text[:5000],  # Limit text length
                'url': url,
                'extracted_at': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

class NewsFeedMonitor:
    """Monitor live news feed for analysis"""
    
    def __init__(self):
        # News API endpoints (free tiers)
        self.news_sources = [
            'https://newsapi.org/v2/top-headlines?country=us&apiKey=demo',
            'https://newsapi.org/v2/everything?q=news&apiKey=demo'
        ]
        
        # RSS feeds of trusted sources
        self.rss_feeds = [
            'http://feeds.reuters.com/reuters/topNews',
            'http://feeds.bbci.co.uk/news/rss.xml',
            'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml'
        ]
    
    def get_live_headlines(self, count=10):
        """Get live headlines from various sources (simulated for demo)"""
        
        # Simulated headlines for demo (in production, use actual API)
        headlines = [
            {
                'title': 'Scientists Confirm Climate Change Accelerating',
                'source': 'Reuters',
                'url': 'https://www.reuters.com/climate-change',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.92
            },
            {
                'title': 'BREAKING: Miracle Cure Discovered for Cancer!',
                'source': 'UnreliableNews.com',
                'url': 'https://unreliablenews.com/miracle-cure',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.15
            },
            {
                'title': 'New Economic Policy Announced by Government',
                'source': 'BBC News',
                'url': 'https://www.bbc.com/news/economy',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.88
            },
            {
                'title': 'You Won\'t Believe What This Celebrity Did!',
                'source': 'Clickbait Central',
                'url': 'https://clickbait.com/celebrity-news',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.25
            },
            {
                'title': 'Global Summit Addresses AI Regulation',
                'source': 'The Guardian',
                'url': 'https://www.theguardian.com/ai-summit',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.85
            },
            {
                'title': 'Secret Government Documents Leaked Online',
                'source': 'Conspiracy Times',
                'url': 'https://conspiracytimes.com/leaks',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.18
            },
            {
                'title': 'Stock Market Reaches All-Time High',
                'source': 'Wall Street Journal',
                'url': 'https://www.wsj.com/markets',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.89
            },
            {
                'title': 'ALIENS VISITED EARTH, GOVERNMENT COVER-UP!',
                'source': 'AlienTruth.org',
                'url': 'https://alientruth.org/coverup',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.08
            },
            {
                'title': 'Healthcare Reforms Pass Key Committee',
                'source': 'AP News',
                'url': 'https://apnews.com/healthcare',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.91
            },
            {
                'title': 'This Simple Trick Will Make You Rich Overnight!',
                'source': 'GetRichQuick.com',
                'url': 'https://getrichquick.com/trick',
                'published_at': datetime.datetime.now().isoformat(),
                'credibility_score': 0.12
            }
        ]
        
        return sorted(headlines, key=lambda x: x['credibility_score'], reverse=True)[:count]

def create_credibility_gauge(score, level, color):
    """Create a gauge chart for credibility score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': f"Credibility Score: {level}"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_analysis_radar(component_scores):
    """Create radar chart for component analysis"""
    categories = list(component_scores.keys())
    values = [component_scores[cat] * 100 for cat in categories]
    
    # Format category names
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=display_categories,
        fill='toself',
        fillcolor='rgba(74, 144, 226, 0.3)',
        line=dict(color='#4A90E2', width=3),
        name='Component Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def main():
    """Main application function"""
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = FakeNewsDetector()
    
    if 'monitor' not in st.session_state:
        st.session_state.monitor = NewsFeedMonitor()
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Header
    st.markdown('<h1 class="main-header">🔍 FAKE NEWS DETECTOR AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Advanced AI-powered misinformation detection with live news monitoring</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔎 Analysis Options")
        
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📝 Analyze Text", "🔗 Analyze URL", "📰 Live News Feed", "📊 Dashboard"]
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth:",
            options=["Basic", "Standard", "Advanced", "Comprehensive"],
            value="Standard"
        )
        
        # Include cross-references
        include_cross_ref = st.checkbox("Include Cross-References", value=True)
        
        # Show technical details
        show_technical = st.checkbox("Show Technical Details", value=False)
        
        st.markdown("---")
        st.markdown("## 📈 Quick Stats")
        
        total_analyses = len(st.session_state.analysis_history)
        st.metric("Total Analyses", total_analyses)
        
        if total_analyses > 0:
            avg_credibility = np.mean([a.get('overall_score', 0) for a in st.session_state.analysis_history])
            st.metric("Avg. Credibility", f"{avg_credibility*100:.0f}%")
        
        st.markdown("---")
        st.markdown("### 🎯 Tips for Detection:")
        st.markdown("""
        1. Check the source reputation
        2. Look for emotional language
        3. Verify facts with trusted sources
        4. Be wary of clickbait headlines
        5. Check publication date
        """)
    
    # Main content area
    if analysis_type == "📝 Analyze Text":
        render_text_analysis()
    elif analysis_type == "🔗 Analyze URL":
        render_url_analysis()
    elif analysis_type == "📰 Live News Feed":
        render_news_feed()
    elif analysis_type == "📊 Dashboard":
        render_dashboard()

def render_text_analysis():
    """Render text analysis interface"""
    st.markdown("## 📝 Text Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        news_text = st.text_area(
            "Enter news text to analyze:",
            height=200,
            placeholder="Paste the news article text here...",
            help="The AI will analyze the text for sensationalism, emotional manipulation, clickbait indicators, and fact claims."
        )
        
        # Quick analysis button
        if st.button("🚀 Analyze Text", type="primary"):
            if news_text and len(news_text.strip()) > 50:
                with st.spinner("Analyzing text with AI..."):
                    # Perform analysis
                    text_analysis = st.session_state.detector.analyze_text(news_text)
                    fact_analysis = st.session_state.detector.check_fact_claims(news_text)
                    source_analysis = {'source_reputation': 0.5, 'source_category': 'Text Input'}
                    
                    # Calculate overall credibility
                    credibility = st.session_state.detector.calculate_overall_credibility(
                        text_analysis, source_analysis, fact_analysis
                    )
                    
                    # Get cross-references
                    keywords = news_text.split()[:10]  # First 10 words as keywords
                    cross_refs = st.session_state.detector.get_cross_references(keywords)
                    
                    # Store in history
                    analysis_result = {
                        'type': 'text',
                        'timestamp': datetime.datetime.now().isoformat(),
                        'text_preview': news_text[:100] + '...' if len(news_text) > 100 else news_text,
                        'text_analysis': text_analysis,
                        'fact_analysis': fact_analysis,
                        'credibility': credibility,
                        'cross_references': cross_refs
                    }
                    
                    st.session_state.analysis_history.append(analysis_result)
                    
                    # Display results
                    display_analysis_results(analysis_result, news_text)
            else:
                st.warning("Please enter at least 50 characters of text to analyze.")
    
    with col2:
        st.markdown("### 📋 Sample Texts")
        st.markdown("""
        Try analyzing these sample texts:
        
        **Credible Example:**
        > "The World Health Organization reported a 15% decrease in malaria cases globally over the past five years, according to data published in their annual report."
        
        **Questionable Example:**
        > "SHOCKING: Doctors HATE this one simple trick that cures cancer overnight! Big Pharma doesn't want you to know!"
        
        **Clickbait Example:**
        > "You won't BELIEVE what this celebrity did at the awards show! The video will go VIRAL!"
        """)
        
        # Quick analyze buttons for samples
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("Credible Sample"):
                st.session_state.sample_text = "The World Health Organization reported a 15% decrease in malaria cases globally over the past five years, according to data published in their annual report. The decline is attributed to improved prevention measures and increased access to treatment in affected regions."
                st.rerun()
        
        with sample_col2:
            if st.button("Clickbait Sample"):
                st.session_state.sample_text = "BREAKING: You won't BELIEVE what scientists just discovered! This one SECRET food burns belly fat overnight! Doctors are SHOCKED!"
                st.rerun()
        
        if 'sample_text' in st.session_state:
            news_text = st.session_state.sample_text
            st.text_area("Sample text loaded:", value=news_text, height=100)

def render_url_analysis():
    """Render URL analysis interface"""
    st.markdown("## 🔗 URL Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # URL input
        url = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article",
            help="The AI will extract and analyze the article content from the URL."
        )
        
        # Also accept text
        st.markdown("---")
        additional_text = st.text_area(
            "Or add additional text for context:",
            height=100,
            placeholder="Optional: Add your own summary or context..."
        )
        
        if st.button("🌐 Analyze URL", type="primary"):
            if url:
                with st.spinner("Extracting and analyzing article..."):
                    # Extract article
                    extraction_result = st.session_state.detector.extract_article_from_url(url)
                    
                    if extraction_result['success']:
                        article_text = extraction_result['text']
                        if additional_text:
                            article_text = additional_text + "\n\n" + article_text
                        
                        # Perform analysis
                        text_analysis = st.session_state.detector.analyze_text(article_text)
                        source_analysis = st.session_state.detector.analyze_source(url)
                        fact_analysis = st.session_state.detector.check_fact_claims(article_text)
                        
                        # Calculate overall credibility
                        credibility = st.session_state.detector.calculate_overall_credibility(
                            text_analysis, source_analysis, fact_analysis
                        )
                        
                        # Get cross-references
                        keywords = article_text.split()[:10]
                        cross_refs = st.session_state.detector.get_cross_references(keywords)
                        
                        # Store in history
                        analysis_result = {
                            'type': 'url',
                            'timestamp': datetime.datetime.now().isoformat(),
                            'url': url,
                            'title': extraction_result.get('title', 'No title'),
                            'text_preview': article_text[:100] + '...',
                            'text_analysis': text_analysis,
                            'source_analysis': source_analysis,
                            'fact_analysis': fact_analysis,
                            'credibility': credibility,
                            'cross_references': cross_refs
                        }
                        
                        st.session_state.analysis_history.append(analysis_result)
                        
                        # Display results
                        display_analysis_results(analysis_result, article_text)
                    else:
                        st.error(f"Failed to extract article: {extraction_result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a URL to analyze.")
    
    with col2:
        st.markdown("### 🌐 Sample URLs")
        st.markdown("""
        Try these sample URLs:
        
        **Trusted Sources:**
        - `https://www.reuters.com/` (Reuters)
        - `https://www.bbc.com/news` (BBC News)
        - `https://apnews.com/` (Associated Press)
        
        **Note:** For demo purposes, actual URL extraction is limited. Use the text analysis for full functionality.
        """)
        
        # Quick URL buttons
        url_col1, url_col2 = st.columns(2)
        with url_col1:
            if st.button("Trusted Example"):
                st.session_state.sample_url = "https://www.reuters.com/business/environment/"
                st.rerun()
        
        with url_col2:
            if st.button("Clickbait Example"):
                st.session_state.sample_url = "https://clickbait-example.com/viral-news"
                st.rerun()
        
        if 'sample_url' in st.session_state:
            url = st.session_state.sample_url
            st.text_input("Sample URL loaded:", value=url)

def render_news_feed():
    """Render live news feed monitor"""
    st.markdown("## 📰 Live News Feed Monitor")
    
    # Get live headlines
    with st.spinner("Fetching latest headlines..."):
        headlines = st.session_state.monitor.get_live_headlines(15)
    
    st.markdown(f"### 🔄 Latest Headlines (Updated: {datetime.datetime.now().strftime('%H:%M:%S')})")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        min_credibility = st.slider("Min. Credibility", 0.0, 1.0, 0.3, 0.1)
    with col2:
        max_results = st.slider("Max Results", 5, 25, 10)
    with col3:
        sort_by = st.selectbox("Sort By", ["Credibility", "Recency", "Source"])
    
    # Filter and sort headlines
    filtered_headlines = [h for h in headlines if h['credibility_score'] >= min_credibility]
    
    if sort_by == "Credibility":
        filtered_headlines.sort(key=lambda x: x['credibility_score'], reverse=True)
    elif sort_by == "Source":
        filtered_headlines.sort(key=lambda x: x['source'])
    
    # Display headlines
    for headline in filtered_headlines[:max_results]:
        credibility = headline['credibility_score']
        
        if credibility >= 0.8:
            credibility_badge = '<span class="source-trustworthy">High Credibility</span>'
        elif credibility >= 0.6:
            credibility_badge = '<span class="source-trustworthy">Good Credibility</span>'
        elif credibility >= 0.4:
            credibility_badge = '<span class="source-unknown">Questionable</span>'
        else:
            credibility_badge = '<span class="source-unreliable">Low Credibility</span>'
        
        st.markdown(f"""
        <div class="news-card">
            <h4>{headline['title']}</h4>
            <p><strong>Source:</strong> {headline['source']} | {credibility_badge}</p>
            <p><strong>Credibility Score:</strong> {credibility*100:.0f}%</p>
            <p><strong>Published:</strong> {headline['published_at'][:10]}</p>
            <a href="{headline['url']}" target="_blank">🔗 Read Article</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis of feed
    st.markdown("---")
    st.markdown("### 📊 Feed Analysis")
    
    if headlines:
        credibility_scores = [h['credibility_score'] for h in headlines]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Credibility", f"{np.mean(credibility_scores)*100:.1f}%")
        with col2:
            st.metric("High Credibility", f"{sum(1 for s in credibility_scores if s >= 0.8)} articles")
        with col3:
            st.metric("Low Credibility", f"{sum(1 for s in credibility_scores if s < 0.4)} articles")
        
        # Distribution chart
        fig = px.histogram(
            x=credibility_scores,
            nbins=10,
            title="Credibility Score Distribution",
            labels={'x': 'Credibility Score', 'y': 'Number of Articles'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_dashboard():
    """Render analytics dashboard"""
    st.markdown("## 📊 Analysis Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Analyze some news to see your dashboard.")
        return
    
    # Convert history to dataframe
    history_df = pd.DataFrame([
        {
            'timestamp': h['timestamp'],
            'type': h['type'],
            'credibility_score': h['credibility']['overall_score'],
            'credibility_level': h['credibility']['credibility_level'],
            'text_preview': h.get('text_preview', '')
        }
        for h in st.session_state.analysis_history
    ])
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(history_df))
    with col2:
        avg_score = history_df['credibility_score'].mean() * 100
        st.metric("Avg. Credibility", f"{avg_score:.1f}%")
    with col3:
        high_cred = sum(history_df['credibility_score'] >= 0.7)
        st.metric("Highly Credible", high_cred)
    with col4:
        low_cred = sum(history_df['credibility_score'] < 0.4)
        st.metric("Low Credibility", low_cred)
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Credibility over time
        if len(history_df) > 1:
            history_df['timestamp_dt'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp_dt')
            
            fig = px.line(
                history_df,
                x='timestamp_dt',
                y='credibility_score',
                title='Credibility Trend Over Time',
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        # Credibility distribution
        fig = px.histogram(
            history_df,
            x='credibility_score',
            nbins=10,
            title='Credibility Score Distribution',
            color_discrete_sequence=['#FF416C']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent analyses
    st.markdown("### 📋 Recent Analyses")
    
    for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
        cred = analysis['credibility']
        
        if cred['credibility_level'] == 'Highly Credible':
            badge = '<span class="source-trustworthy">Highly Credible</span>'
        elif cred['credibility_level'] == 'Mostly Credible':
            badge = '<span class="source-trustworthy">Mostly Credible</span>'
        elif cred['credibility_level'] == 'Questionable':
            badge = '<span class="source-unknown">Questionable</span>'
        else:
            badge = '<span class="source-unreliable">Likely False</span>'
        
        st.markdown(f"""
        <div class="news-card">
            <h4>Analysis #{len(st.session_state.analysis_history)-i}</h4>
            <p>{badge} | Score: {cred['overall_score']*100:.0f}%</p>
            <p><small>{analysis.get('text_preview', 'No preview')}</small></p>
            <p><small>Type: {analysis['type'].upper()} | {analysis['timestamp'][:16]}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export data
    st.markdown("---")
    if st.button("📥 Export Analysis Data"):
        export_data = {
            'analyses': st.session_state.analysis_history,
            'exported_at': datetime.datetime.now().isoformat(),
            'total_analyses': len(st.session_state.analysis_history)
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"fake_news_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def display_analysis_results(analysis_result, full_text=None):
    """Display comprehensive analysis results"""
    
    credibility = analysis_result['credibility']
    text_analysis = analysis_result.get('text_analysis', {})
    source_analysis = analysis_result.get('source_analysis', {})
    fact_analysis = analysis_result.get('fact_analysis', {})
    cross_refs = analysis_result.get('cross_references', [])
    
    # Display credibility score
    st.markdown(f"## 🎯 Analysis Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge chart
        fig = create_credibility_gauge(
            credibility['overall_score'],
            credibility['credibility_level'],
            credibility['color']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Credibility box
        if credibility['credibility_level'] == 'Highly Credible':
            st.markdown(f"""
            <div class="credibility-excellent">
                <h3>✅ {credibility['credibility_level']}</h3>
                <p>This content appears to be from a credible source with minimal sensationalism or misinformation indicators.</p>
            </div>
            """, unsafe_allow_html=True)
        elif credibility['credibility_level'] == 'Mostly Credible':
            st.markdown(f"""
            <div class="credibility-good">
                <h3>⚠️ {credibility['credibility_level']}</h3>
                <p>This content is mostly credible but may contain some sensational language or requires verification of specific claims.</p>
            </div>
            """, unsafe_allow_html=True)
        elif credibility['credibility_level'] == 'Questionable':
            st.markdown(f"""
            <div class="credibility-warning">
                <h3>⚠️ {credibility['credibility_level']}</h3>
                <p>This content has significant credibility issues. Verify claims with trusted sources before sharing.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="credibility-danger">
                <h3>❌ {credibility['credibility_level']}</h3>
                <p>This content shows strong indicators of misinformation, sensationalism, or unreliable sourcing.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Component Analysis", "🔍 Text Analysis", "📰 Source Analysis", "✅ Verification"])
    
    with tab1:
        # Component scores
        st.markdown("### Component Analysis")
        
        fig = create_analysis_radar(credibility['component_scores'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Component breakdown
        for component, score in credibility['component_scores'].items():
            component_name = component.replace('_', ' ').title()
            progress_value = score
            
            st.markdown(f"**{component_name}:** {score*100:.0f}%")
            st.progress(progress_value)
    
    with tab2:
        # Text analysis details
        st.markdown("### 📝 Text Analysis Details")
        
        if 'error' not in text_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Word Count", text_analysis.get('word_count', 0))
                st.metric("Readability Score", f"{text_analysis.get('readability_score', 0):.0f}")
                st.metric("Clickbait Score", f"{text_analysis.get('clickbait_score', 0)*100:.0f}%")
            
            with col2:
                st.metric("Sensationalism", f"{text_analysis.get('sensational_score', 0)*100:.0f}%")
                st.metric("Emotional Score", f"{text_analysis.get('emotional_score', 0)*100:.0f}%")
                st.metric("Subjectivity", f"{text_analysis.get('subjectivity_score', 0)*100:.0f}%")
            
            # Detected phrases
            if text_analysis.get('common_phrases_found'):
                st.markdown("#### 🚨 Sensational Phrases Detected:")
                for phrase in text_analysis['common_phrases_found']:
                    st.markdown(f"- `{phrase}`")
            
            if text_analysis.get('emotional_phrases_found'):
                st.markdown("#### 😠 Emotional Manipulation Phrases:")
                for phrase in text_analysis['emotional_phrases_found']:
                    st.markdown(f"- `{phrase}`")
            
            # Excessive punctuation
            if text_analysis.get('exclamation_count', 0) > 5:
                st.warning(f"⚠️ High exclamation count: {text_analysis['exclamation_count']} (may indicate sensationalism)")
            
            if text_analysis.get('all_caps_count', 0) > 3:
                st.warning(f"⚠️ Excessive ALL CAPS: {text_analysis['all_caps_count']} instances")
    
    with tab3:
        # Source analysis
        st.markdown("### 📰 Source Analysis")
        
        if source_analysis and 'error' not in source_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Source Reputation", f"{source_analysis.get('source_reputation', 0)*100:.0f}%")
                st.metric("Source Category", source_analysis.get('source_category', 'Unknown'))
                st.metric("SSL Secured", "✅ Yes" if source_analysis.get('ssl_secured') else "❌ No")
            
            with col2:
                st.metric("Domain", source_analysis.get('domain', 'Unknown'))
                st.metric("URL Structure", f"{source_analysis.get('url_structure_score', 0)*100:.0f}%")
            
            # Source trust badge
            if source_analysis.get('is_trusted'):
                st.success("✅ This domain is on our trusted sources list.")
            elif source_analysis.get('is_unreliable'):
                st.error("❌ This domain is on our unreliable sources list.")
            else:
                st.warning("⚠️ This domain is not in our trusted sources database. Verify with additional sources.")
        else:
            st.info("No URL source analysis available for text input.")
    
    with tab4:
        # Fact checking and verification
        st.markdown("### ✅ Fact Verification")
        
        # Fact claims analysis
        if fact_analysis.get('total_claims', 0) > 0:
            st.metric("Total Claims Identified", fact_analysis['total_claims'])
            st.metric("Potential Misinformation", fact_analysis['misinformation_claims'])
            st.metric("Verifiable Claims", fact_analysis['verifiable_claims'])
            
            # Display claims
            if fact_analysis['claims']:
                st.markdown("#### 📋 Identified Claims:")
                for claim in fact_analysis['claims'][:5]:  # Show first 5
                    if claim['type'] == 'potential_misinformation':
                        st.markdown(f"""
                        <div class="warning-box">
                        <strong>⚠️ {claim['label']}</strong><br>
                        Claim: "{claim['claim']}"<br>
                        Confidence: {claim['confidence']*100:.0f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="analysis-box">
                        <strong>📄 {claim['label']}</strong><br>
                        Claim: "{claim['claim']}"<br>
                        This claim should be verified with reliable sources.
                        </div>
                        """, unsafe_allow_html=True)
        
        # Cross-references
        if cross_refs:
            st.markdown("#### 🔗 Cross-Reference with Trusted Sources:")
            for ref in cross_refs:
                st.markdown(f"""
                <div class="news-card">
                    <h4>{ref['title']}</h4>
                    <p>{ref['snippet']}</p>
                    <p><strong>Relevance:</strong> {ref['relevance']*100:.0f}%</p>
                    <a href="{ref['url']}" target="_blank">🔗 Visit Source</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific cross-references found. Consider checking general fact-checking websites.")

if __name__ == "__main__":
    main()
