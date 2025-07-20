"""
NLP Plugin for AETHERION
Provides text analysis, sentiment analysis, and language processing capabilities
"""

import re
import string
from typing import Dict, List, Any
from collections import Counter
import logging

from .plugin_base import PluginBase, PluginMetadata, PluginConfig

logger = logging.getLogger(__name__)


class NLPPlugin(PluginBase):
    """Natural Language Processing plugin for text analysis"""
    
    name = "nlp_analyzer"
    version = "1.0.0"
    description = "Advanced NLP capabilities for text analysis and processing"
    
    metadata = PluginMetadata(
        name=name,
        version=version,
        description=description,
        author="AETHERION Core",
        category="analysis",
        tags=["nlp", "text", "analysis", "sentiment"],
        dependencies=[],
        config_schema={
            "max_text_length": {"type": "integer", "default": 10000},
            "enable_sentiment": {"type": "boolean", "default": True},
            "enable_keywords": {"type": "boolean", "default": True},
            "enable_summary": {"type": "boolean", "default": True}
        }
    )
    
    def __init__(self):
        super().__init__()
        self.config = PluginConfig(self.metadata.config_schema)
        self._initialized = False
        
        # Common stop words for English
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Sentiment word lists (simplified)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'brilliant', 'outstanding', 'perfect', 'beautiful', 'love', 'happy',
            'joy', 'success', 'win', 'victory', 'hope', 'dream', 'inspire'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'sad', 'angry', 'fear', 'pain', 'death', 'kill', 'destroy',
            'fail', 'lose', 'defeat', 'despair', 'hopeless', 'nightmare'
        }
    
    def initialize(self):
        """Initialize the NLP plugin"""
        logger.info("Initializing NLP Plugin")
        self._initialized = True
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract keywords from text"""
        if not self.config.get("enable_keywords", True):
            return []
            
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Split into words
        words = clean_text.split()
        
        # Filter out stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        keywords = []
        for word, count in word_counts.most_common(top_n):
            keywords.append({
                "word": word,
                "frequency": count,
                "percentage": round((count / len(filtered_words)) * 100, 2)
            })
            
        return keywords
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not self.config.get("enable_sentiment", True):
            return {"sentiment": "neutral", "score": 0.0}
            
        # Preprocess text
        clean_text = self.preprocess_text(text)
        words = clean_text.split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate sentiment score
        total_words = len(words)
        if total_words == 0:
            return {"sentiment": "neutral", "score": 0.0}
            
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Determine sentiment category
        if sentiment_score > 0.05:
            sentiment = "positive"
        elif sentiment_score < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "sentiment": sentiment,
            "score": round(sentiment_score, 3),
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_words": total_words
        }
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple summary of text"""
        if not self.config.get("enable_summary", True):
            return ""
            
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
            
        # Simple summary: take first few sentences
        summary_sentences = sentences[:max_sentences]
        return '. '.join(summary_sentences) + '.'
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity metrics"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = text.split()
        characters = len(text.replace(' ', ''))
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = characters / len(words) if words else 0
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        return {
            "sentences": len(sentences),
            "words": len(words),
            "characters": characters,
            "unique_words": unique_words,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "lexical_diversity": round(lexical_diversity, 3)
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NLP analysis on input text"""
        if not self._initialized:
            self.initialize()
            
        text = input_data.get("text", "")
        if not text:
            return {"error": "No text provided for analysis"}
            
        # Check text length limit
        max_length = self.config.get("max_text_length", 10000)
        if len(text) > max_length:
            text = text[:max_length]
            
        try:
            # Perform analysis
            sentiment = self.analyze_sentiment(text)
            keywords = self.extract_keywords(text)
            summary = self.generate_summary(text)
            complexity = self.analyze_text_complexity(text)
            
            return {
                "success": True,
                "analysis": {
                    "sentiment": sentiment,
                    "keywords": keywords,
                    "summary": summary,
                    "complexity": complexity,
                    "text_length": len(text),
                    "processed_length": len(text)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup plugin resources"""
        logger.info("Cleaning up NLP Plugin")
        self._initialized = False 