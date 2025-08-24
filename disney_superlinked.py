"""
Disney Review RAG System - Using Superlinked + LangChain
========================================================

This module creates a RAG system to answer questions about Disney reviews
using the analyzed data from DisneyReviewAnalyzer with Superlinked's 
multi-space vector indexing and LangChain integration.

Based on official Superlinked documentation and examples.
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict
from pathlib import Path

# Superlinked imports - Fixed based on documentation
from superlinked import framework as sl

# LangChain imports
from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import Field


class DisneyReviewSchema(sl.Schema):
    """Superlinked schema for Disney review data - properly inheriting from sl.Schema."""
    
    review_id: sl.IdField
    review_text: sl.String
    overall_sentiment: sl.String
    sentiment_confidence: sl.Float
    
    # Aspect sentiments
    attractions_sentiment: sl.String
    food_sentiment: sl.String
    hotels_sentiment: sl.String
    staff_sentiment: sl.String
    price_sentiment: sl.String
    crowd_sentiment: sl.String
    cleanliness_sentiment: sl.String
    accessibility_sentiment: sl.String
    
    # Demographics and context
    visitor_type: sl.String
    visitor_origin: sl.String
    visit_frequency: sl.String
    special_occasion: sl.String
    trip_duration: sl.String
    
    # Time dimensions - matching your DataFrame
    season: sl.String
    month: sl.String
    day_type: sl.String
    
    # Engagement signals
    time_spent: sl.String
    popular_attractions: sl.String
    recommendation_intent: sl.String
    
    # Pain points
    main_complaint: sl.String
    journey_stage: sl.String
    pain_severity: sl.String
    
    # Meta
    review_length: sl.String
    language_quality: sl.String
    
    # Numerical scores
    rating: sl.Integer  # From your DataFrame
    satisfaction_score: sl.Float  # Derived metric
    
    # Boolean fields from DataFrame (keeping as strings for categorical spaces)
    has_children: sl.String
    repeat_visits: sl.String
    mentions_tickets: sl.String
    mentions_fast_passes: sl.String
    mentions_souvenirs: sl.String
    mentions_food_purchases: sl.String
    mentions_hotel_booking: sl.String
    mentions_parking: sl.String


class DisneyReviewRAG:
    """RAG system for Disney review analysis using Superlinked + LangChain."""
    
    def __init__(self):
        """Initialize the Disney Review RAG system."""
        
        # Initialize schema instance
        self.disney_review = DisneyReviewSchema()
        
        # Create vector spaces
        self._create_vector_spaces()
        
        # Create index
        self._create_index()
        
        # Create query
        self._create_query()
        
        # Setup data parser
        self._create_data_parser()
        
        # Initialize Superlinked components (will be set up in setup_superlinked)
        self.source = None
        self.executor = None
        self.app = None
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _create_vector_spaces(self):
        """Create specialized vector spaces for Disney review features."""
        
        # 1. Text Similarity Space - for semantic search over review content
        self.text_space = sl.TextSimilaritySpace(
            text=sl.chunk(
                self.disney_review.review_text, 
                chunk_size=300, 
                chunk_overlap=75
            ),
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 2. Overall Sentiment Space
        self.overall_sentiment_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.overall_sentiment,
            categories=["positive", "negative", "neutral"],
            uncategorized_as_category=True
        )
        
        # 3. Aspect Sentiment Spaces
        sentiment_categories = ["positive", "negative", "neutral", "not_mentioned"]
        
        self.attractions_sentiment_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.attractions_sentiment,
            categories=sentiment_categories,
            uncategorized_as_category=True
        )
        
        self.food_sentiment_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.food_sentiment,
            categories=sentiment_categories,
            uncategorized_as_category=True
        )
        
        self.staff_sentiment_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.staff_sentiment,
            categories=sentiment_categories,
            uncategorized_as_category=True
        )
        
        self.price_sentiment_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.price_sentiment,
            categories=sentiment_categories,
            uncategorized_as_category=True
        )
        
        # 4. Visitor Demographics Spaces
        self.visitor_type_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.visitor_type,
            categories=["families", "couples", "solo", "group", "unknown"],
            uncategorized_as_category=True
        )
        
        self.visitor_origin_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.visitor_origin,
            categories=["international", "local", "unknown"],
            uncategorized_as_category=True
        )
        
        # 5. Visit Context Spaces
        self.visit_frequency_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.visit_frequency,
            categories=["first_time", "returning", "frequent", "unknown"],
            uncategorized_as_category=True
        )
        
        self.trip_duration_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.trip_duration,
            categories=["day_trip", "weekend", "week", "extended", "unknown"],
            uncategorized_as_category=True
        )
        
        # 6. Temporal Spaces
        self.season_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.season,
            categories=["spring", "summer", "fall", "winter", "unknown"],
            uncategorized_as_category=True
        )
        
        self.month_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.month,
            categories=["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"],
            uncategorized_as_category=True
        )
        
        # 7. Engagement Spaces
        self.recommendation_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.recommendation_intent,
            categories=["yes", "no", "conditional", "unknown"],
            uncategorized_as_category=True
        )
        
        # 8. Numerical Spaces
        self.sentiment_confidence_space = sl.NumberSpace(
            number=self.disney_review.sentiment_confidence,
            min_value=0.0,
            max_value=1.0,
            mode=sl.Mode.MAXIMUM
        )
        
        self.rating_space = sl.NumberSpace(
            number=self.disney_review.rating,
            min_value=1,
            max_value=5,
            mode=sl.Mode.MAXIMUM
        )
        
        self.satisfaction_score_space = sl.NumberSpace(
            number=self.disney_review.satisfaction_score,
            min_value=0.0,
            max_value=10.0,
            mode=sl.Mode.MAXIMUM
        )
        
        # 9. Pain Points Space
        self.pain_severity_space = sl.CategoricalSimilaritySpace(
            category_input=self.disney_review.pain_severity,
            categories=["minor", "moderate", "major", "trip_ruining", "unknown"],
            uncategorized_as_category=True
        )
    
    def _create_index(self):
        """Create the Superlinked index with all vector spaces."""
        
        self.disney_features = [
            # Core spaces - prioritize text and sentiment
            self.text_space,
            self.overall_sentiment_space,
            
            # Aspect sentiments - high importance for Disney analysis
            self.attractions_sentiment_space,
            self.food_sentiment_space,
            self.staff_sentiment_space,
            self.price_sentiment_space,
            
            # Demographics
            self.visitor_type_space,
            self.visitor_origin_space,
            
            # Context
            self.visit_frequency_space,
            self.trip_duration_space,
            
            # Temporal
            self.season_space,
            self.month_space,
            
            # Engagement
            self.recommendation_space,
            
            # Numerical
            self.sentiment_confidence_space,
            self.rating_space,
            self.satisfaction_score_space,
            
            # Pain points
            self.pain_severity_space
        ]
        
        # Create index with additional fields for selection
        self.disney_index = sl.Index(
            self.disney_features,
            fields=[
                self.disney_review.rating,
                self.disney_review.review_length,
                self.disney_review.language_quality,
                self.disney_review.has_children,
                self.disney_review.repeat_visits
            ]
        )
    
    def _create_query(self):
        """Create the parameterized query for flexible retrieval."""
        
        # Define weights for different aspects - optimized for Disney analysis
        self.weights = {
            self.text_space: sl.Param("text_weight"),
            self.overall_sentiment_space: sl.Param("sentiment_weight"),
            self.attractions_sentiment_space: sl.Param("attractions_weight"),
            self.food_sentiment_space: sl.Param("food_weight"),
            self.staff_sentiment_space: sl.Param("staff_weight"),
            self.price_sentiment_space: sl.Param("price_weight"),
            self.visitor_type_space: sl.Param("visitor_type_weight"),
            self.visit_frequency_space: sl.Param("visit_frequency_weight"),
            self.season_space: sl.Param("season_weight"),
            self.month_space: sl.Param("month_weight"),
            self.recommendation_space: sl.Param("recommendation_weight"),
            self.sentiment_confidence_space: sl.Param("confidence_weight"),
            self.rating_space: sl.Param("rating_weight"),
            self.satisfaction_score_space: sl.Param("satisfaction_weight"),
            self.pain_severity_space: sl.Param("pain_weight")
        }
        
        # Create the query following Superlinked best practices
        self.disney_query = (
            sl.Query(self.disney_index, weights=self.weights)
            .find(self.disney_review)
            .similar(self.text_space.text, sl.Param("query_text"))
            .filter(self.disney_review.overall_sentiment == sl.Param("sentiment_filter"))
            .filter(self.disney_review.visitor_type == sl.Param("visitor_type_filter"))
            .filter(self.disney_review.visit_frequency == sl.Param("visit_frequency_filter"))
            .filter(self.disney_review.attractions_sentiment == sl.Param("attractions_filter"))
            .filter(self.disney_review.recommendation_intent == sl.Param("recommendation_filter"))
            .select([
                self.disney_review.review_text,
                self.disney_review.overall_sentiment,
                self.disney_review.sentiment_confidence,
                self.disney_review.attractions_sentiment,
                self.disney_review.food_sentiment,
                self.disney_review.staff_sentiment,
                self.disney_review.price_sentiment,
                self.disney_review.visitor_type,
                self.disney_review.visit_frequency,
                self.disney_review.recommendation_intent,
                self.disney_review.main_complaint,
                self.disney_review.popular_attractions,
                self.disney_review.satisfaction_score,
                self.disney_review.rating,
                self.disney_review.season,
                self.disney_review.month,
                self.disney_review.has_children,
                self.disney_review.repeat_visits
            ])
            .limit(sl.Param("limit"))
        )
    
    def _create_data_parser(self):
        """Create DataFrameParser for mapping DataFrame columns to schema fields."""
        
        self.dataframe_parser = sl.DataFrameParser(
            schema=self.disney_review,
            mapping={
                self.disney_review.review_id: "Review_ID",
                self.disney_review.review_text: "Review_Text",
                self.disney_review.overall_sentiment: "overall_sentiment",
                self.disney_review.sentiment_confidence: "sentiment_confidence",
                
                # Aspect sentiments
                self.disney_review.attractions_sentiment: "attractions_sentiment",
                self.disney_review.food_sentiment: "food_sentiment",
                self.disney_review.hotels_sentiment: "hotels_sentiment",
                self.disney_review.staff_sentiment: "staff_sentiment",
                self.disney_review.price_sentiment: "price_sentiment",
                self.disney_review.crowd_sentiment: "crowd_sentiment",
                self.disney_review.cleanliness_sentiment: "cleanliness_sentiment",
                self.disney_review.accessibility_sentiment: "accessibility_sentiment",
                
                # Demographics
                self.disney_review.visitor_type: "visitor_type",
                self.disney_review.visitor_origin: "visitor_origin",
                self.disney_review.visit_frequency: "visit_frequency",
                self.disney_review.special_occasion: "special_occasion",
                self.disney_review.trip_duration: "trip_duration",
                
                # Temporal
                self.disney_review.season: "season",
                self.disney_review.month: "month",
                self.disney_review.day_type: "day_type",
                
                # Engagement
                self.disney_review.time_spent: "time_spent",
                self.disney_review.popular_attractions: "popular_attractions",
                self.disney_review.recommendation_intent: "recommendation_intent",
                
                # Pain points
                self.disney_review.main_complaint: "main_complaint",
                self.disney_review.journey_stage: "journey_stage",
                self.disney_review.pain_severity: "pain_severity",
                
                # Meta
                self.disney_review.review_length: "review_length",
                self.disney_review.language_quality: "language_quality",
                
                # Numerical fields
                self.disney_review.rating: "Rating",
                self.disney_review.satisfaction_score: "satisfaction_score",  # Will be calculated
                
                # Boolean fields converted to strings
                self.disney_review.has_children: "has_children",
                self.disney_review.repeat_visits: "repeat_visits",
                self.disney_review.mentions_tickets: "mentions_tickets",
                self.disney_review.mentions_fast_passes: "mentions_fast_passes",
                self.disney_review.mentions_souvenirs: "mentions_souvenirs",
                self.disney_review.mentions_food_purchases: "mentions_food_purchases",
                self.disney_review.mentions_hotel_booking: "mentions_hotel_booking",
                self.disney_review.mentions_parking: "mentions_parking"
            }
        )
    
    def setup_superlinked(self):
        """Setup Superlinked components following the official pattern."""
        
        # Create source with schema and parser
        self.source = sl.InMemorySource(self.disney_review, parser=self.dataframe_parser)
        
        # Create executor with source and index
        self.executor = sl.InMemoryExecutor(sources=[self.source], indices=[self.disney_index])
        
        # Run the application
        self.app = self.executor.run()
        
        print("✅ Superlinked application initialized successfully")
        return self.app
    
    def prepare_data_for_indexing(self, analyzed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare analyzed Disney review data for Superlinked indexing.
        Updated to match your actual DataFrame structure and add calculated fields.
        
        Args:
            analyzed_df: DataFrame with analyzed Disney review data
            
        Returns:
            Prepared DataFrame ready for indexing
        """
        # Create a copy to avoid modifying original
        prepared_df = analyzed_df.copy()
        
        # Filter for successful analyses only
        prepared_df = prepared_df[prepared_df.get('analysis_successful', True) == True]
        
        # Calculate satisfaction score
        prepared_df['satisfaction_score'] = prepared_df.apply(
            lambda row: self._calculate_satisfaction_score(
                row.get('overall_sentiment', 'neutral'),
                row.get('sentiment_confidence', 0.5),
                row.get('Rating', 3)
            ), axis=1
        )
        
        # Convert boolean fields to strings for categorical spaces
        boolean_fields = ['has_children', 'repeat_visits', 'mentions_tickets', 
                         'mentions_fast_passes', 'mentions_souvenirs', 
                         'mentions_food_purchases', 'mentions_hotel_booking', 
                         'mentions_parking']
        
        for field in boolean_fields:
            if field in prepared_df.columns:
                prepared_df[field] = prepared_df[field].astype(str)
        
        # Ensure all required fields exist with defaults
        required_fields = {
            'Review_ID': lambda x: str(x.name),  # Use index as fallback
            'Review_Text': '',
            'overall_sentiment': 'neutral',
            'sentiment_confidence': 0.5,
            'Rating': 3,
            'satisfaction_score': 5.0
        }
        
        for field, default_value in required_fields.items():
            if field not in prepared_df.columns:
                if callable(default_value):
                    prepared_df[field] = prepared_df.apply(default_value, axis=1)
                else:
                    prepared_df[field] = default_value
        
        return prepared_df
    
    def _calculate_satisfaction_score(self, sentiment: str, confidence: float, rating: int) -> float:
        """Calculate a satisfaction score from sentiment, confidence, and rating."""
        # Base score from sentiment
        base_scores = {
            'positive': 7.0,
            'neutral': 5.0,
            'negative': 3.0
        }
        
        base_score = base_scores.get(sentiment.lower(), 5.0)
        
        # Adjust based on confidence (higher confidence = more extreme scores)
        confidence_adjustment = confidence * 1.5
        if sentiment.lower() == 'positive':
            sentiment_score = min(10.0, base_score + confidence_adjustment)
        elif sentiment.lower() == 'negative':
            sentiment_score = max(0.0, base_score - confidence_adjustment)
        else:
            sentiment_score = base_score
        
        # Incorporate actual rating (1-5 scale) - weight 0.6 sentiment, 0.4 rating
        rating_normalized = (rating - 1) * 2.5  # Convert 1-5 to 0-10 scale
        final_score = (sentiment_score * 0.6) + (rating_normalized * 0.4)
        
        return round(min(10.0, max(0.0, final_score)), 2)
    
    def index_data(self, analyzed_df: pd.DataFrame):
        """Index the analyzed Disney review data using the official Superlinked pattern."""
        
        if self.app is None:
            raise ValueError("Superlinked app not initialized. Call setup_superlinked() first.")
        
        # Prepare data
        prepared_df = self.prepare_data_for_indexing(analyzed_df)
        
        # Index data using the source.put method as shown in examples
        self.source.put([prepared_df])
        
        print(f"✅ Indexed {len(prepared_df)} Disney reviews successfully")
        return len(prepared_df)


class DisneyReviewRetriever(BaseRetriever):
    """Custom LangChain retriever for Disney reviews using Superlinked."""
    
    sl_client: Any = Field(...)
    disney_query: Any = Field(...)
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve relevant Disney review documents."""
        
        # Default query parameters optimized for Disney reviews
        query_params = {
            "query_text": query,
            # Core weights
            "text_weight": kwargs.get("text_weight", 1.0),
            "sentiment_weight": kwargs.get("sentiment_weight", 0.7),
            
            # Aspect weights - higher for Disney-specific aspects
            "attractions_weight": kwargs.get("attractions_weight", 0.9),
            "food_weight": kwargs.get("food_weight", 0.6),
            "staff_weight": kwargs.get("staff_weight", 0.6),
            "price_weight": kwargs.get("price_weight", 0.5),
            
            # Demographics
            "visitor_type_weight": kwargs.get("visitor_type_weight", 0.4),
            "visit_frequency_weight": kwargs.get("visit_frequency_weight", 0.3),
            
            # Temporal
            "season_weight": kwargs.get("season_weight", 0.2),
            "month_weight": kwargs.get("month_weight", 0.2),
            
            # Engagement
            "recommendation_weight": kwargs.get("recommendation_weight", 0.7),
            
            # Numerical
            "confidence_weight": kwargs.get("confidence_weight", 0.2),
            "rating_weight": kwargs.get("rating_weight", 0.4),
            "satisfaction_weight": kwargs.get("satisfaction_weight", 0.5),
            
            # Pain points
            "pain_weight": kwargs.get("pain_weight", 0.6),
            
            "limit": kwargs.get("limit", 5),
            
            # Optional filters - set to None if not specified
            "sentiment_filter": kwargs.get("sentiment_filter"),
            "visitor_type_filter": kwargs.get("visitor_type_filter"),
            "visit_frequency_filter": kwargs.get("visit_frequency_filter"),
            "attractions_filter": kwargs.get("attractions_filter"),
            "recommendation_filter": kwargs.get("recommendation_filter")
        }
        
        try:
            results = self.sl_client.query(
                self.disney_query,
                **query_params
            )
            
            documents = []
            for entry in results.entries:
                try:
                    fields = entry.fields or {}
                    
                    if fields.get("review_text"):
                        doc_content = fields["review_text"]
                        
                        # Create rich metadata
                        metadata = {
                            "review_id": getattr(entry, 'id', 'unknown'),
                            "overall_sentiment": fields.get("overall_sentiment", "unknown"),
                            "sentiment_confidence": fields.get("sentiment_confidence", 0.0),
                            "attractions_sentiment": fields.get("attractions_sentiment", "not_mentioned"),
                            "food_sentiment": fields.get("food_sentiment", "not_mentioned"),
                            "staff_sentiment": fields.get("staff_sentiment", "not_mentioned"),
                            "price_sentiment": fields.get("price_sentiment", "not_mentioned"),
                            "visitor_type": fields.get("visitor_type", "unknown"),
                            "visit_frequency": fields.get("visit_frequency", "unknown"),
                            "recommendation_intent": fields.get("recommendation_intent", "unknown"),
                            "main_complaint": fields.get("main_complaint", ""),
                            "popular_attractions": fields.get("popular_attractions", ""),
                            "satisfaction_score": fields.get("satisfaction_score", 0.0),
                            "rating": fields.get("rating", 0),
                            "season": fields.get("season", "unknown"),
                            "month": fields.get("month", "unknown"),
                            "has_children": fields.get("has_children", "False"),
                            "repeat_visits": fields.get("repeat_visits", "False")
                        }
                        
                        documents.append(Document(page_content=doc_content, metadata=metadata))
                
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
            
            return documents
        
        except Exception as e:
            print(f"Error executing Disney review query: {e}")
            return []


def create_disney_rag_system(analyzed_df: pd.DataFrame) -> RetrievalQA:
    """
    Create a complete Disney Review RAG system.
    
    Args:
        analyzed_df: DataFrame with analyzed Disney review data
        
    Returns:
        RetrievalQA chain ready for questioning
    """
    
    # Initialize RAG system
    disney_rag = DisneyReviewRAG()
    
    # Setup Superlinked components
    app = disney_rag.setup_superlinked()
    
    # Index the data
    indexed_count = disney_rag.index_data(analyzed_df)
    print(f"✅ Disney RAG system ready with {indexed_count} reviews")
    
    # Create retriever
    retriever = DisneyReviewRetriever(
        sl_client=app,
        disney_query=disney_rag.disney_query
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=disney_rag.llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": create_disney_qa_prompt()
        }
    )
    
    return qa_chain


def create_disney_qa_prompt():
    """Create a specialized prompt for Disney review Q&A."""
    
    template = """You are an expert Disney theme park analyst. Use the following Disney review excerpts to answer questions about visitor experiences, sentiment, and insights.

Context from Disney Reviews:
{context}

Question: {question}

Please provide a comprehensive answer based on the review data. Include specific insights about:
- Overall sentiment patterns
- Visitor demographics when relevant  
- Specific aspects mentioned (attractions, food, staff, pricing, etc.)
- Common pain points or positive highlights
- Recommendations based on the data

If the question asks for trends or patterns, analyze across multiple reviews.
If asking about specific experiences, focus on relevant review details.

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Example usage functions
def example_disney_rag_workflow(analyzed_df: pd.DataFrame):
    """
    Example workflow for using the Disney RAG system with pre-analyzed data.
    
    Args:
        analyzed_df: DataFrame with already analyzed Disney review data
    """
    
    # Create the RAG system with pre-analyzed data
    qa_chain = create_disney_rag_system(analyzed_df)
    
    # Ask questions about the Disney reviews
    sample_questions = [
        "What do families with children think about Disney attractions?",
        "What are the main complaints about food at Disney parks?",
        "How do first-time visitors rate their experience compared to returning visitors?",
        "What attractions get mentioned most positively?",
        "What pricing concerns do visitors have?",
        "How satisfied are international visitors compared to local ones?"
    ]
    
    print("Disney Review RAG System - Ready to answer questions!")
    print("=" * 60)
    
    for question in sample_questions:
        try:
            result = qa_chain(question)
            print(f"Q: {question}")
            print(f"A: {result['result']}")
            print(f"Sources: {len(result['source_documents'])} reviews")
            print("-" * 60)
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            print("-" * 60)


def load_and_create_rag_system(csv_path: str) -> RetrievalQA:
    """
    Load pre-analyzed Disney review data and create RAG system.
    
    Args:
        csv_path: Path to CSV file with analyzed Disney review data
        
    Returns:
        RetrievalQA chain ready for questioning
    """
    
    # Load pre-analyzed data
    print(f"📊 Loading analyzed Disney review data from {csv_path}")
    analyzed_df = pd.read_csv(csv_path)
    
    # Filter for successful analyses only
    successful_df = analyzed_df[analyzed_df.get('analysis_successful', True) == True]
    print(f"✅ Found {len(successful_df)} successfully analyzed reviews out of {len(analyzed_df)} total")
    
    # Create RAG system
    qa_chain = create_disney_rag_system(successful_df)
    
    return qa_chain


