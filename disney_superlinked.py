"""
Disney Review RAG System - Using Superlinked + LangChain
========================================================

This module creates a RAG system to answer questions about Disney reviews
using the analyzed data from DisneyReviewAnalyzer with Superlinked's 
multi-space vector indexing and LangChain integration.

Based on official Superlinked documentation and examples.
"""

import os
import pandas as pd
from typing import List, Any

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
        
        # Setup OpenAI configuration for natural language querying
        self._setup_openai_config()
        
        # Create natural language query (will be set up after _create_query)
        self._create_natural_language_query()
    
    def _setup_openai_config(self):
        """Setup OpenAI configuration for natural language querying."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not found. Natural language querying will be disabled.")
            self.openai_config = None
        else:
            self.openai_config = sl.OpenAIClientConfig(
                api_key=api_key, 
                model="gpt-4o"
            )
            print("✅ OpenAI configuration set up for natural language querying")
    
    def _create_natural_language_query(self):
        """Create a natural language query version of the Disney query."""
        if self.openai_config is None:
            print("⚠️  OpenAI configuration not available. Skipping natural language query setup.")
            self.disney_nlq_query = None
            return
        
        # Create enhanced query with natural language support and parameter descriptions
        self.disney_nlq_query = (
            sl.Query(self.disney_index, weights=self.weights)
            .find(self.disney_review)
            .similar(
                self.text_space.text, 
                sl.Param(
                    "query_text",
                    description="Main text content from the user's question about Disney reviews, attractions, experiences, or specific aspects like food, staff, pricing, etc."
                )
            )
            .filter(
                self.disney_review.overall_sentiment == sl.Param(
                    "sentiment_filter",
                    description="Filter reviews by overall sentiment: positive, negative, or neutral",
                    options=["positive", "negative", "neutral"]
                )
            )
            .filter(
                self.disney_review.visitor_type == sl.Param(
                    "visitor_type_filter",
                    description="Filter by type of visitors: families, couples, solo, group, or unknown",
                    options=["families", "couples", "solo", "group", "unknown"]
                )
            )
            .filter(
                self.disney_review.visit_frequency == sl.Param(
                    "visit_frequency_filter",
                    description="Filter by visit frequency: first_time, returning, frequent, or unknown",
                    options=["first_time", "returning", "frequent", "unknown"]
                )
            )
            .filter(
                self.disney_review.attractions_sentiment == sl.Param(
                    "attractions_filter",
                    description="Filter by sentiment about attractions: positive, negative, neutral, or not_mentioned",
                    options=["positive", "negative", "neutral", "not_mentioned"]
                )
            )
            .filter(
                self.disney_review.recommendation_intent == sl.Param(
                    "recommendation_filter",
                    description="Filter by recommendation intent: yes, no, conditional, or unknown",
                    options=["yes", "no", "conditional", "unknown"]
                )
            )
            .filter(
                self.disney_review.season == sl.Param(
                    "season_filter",
                    description="Filter by season of visit: spring, summer, fall, winter, or unknown",
                    options=["spring", "summer", "fall", "winter", "unknown"]
                )
            )
            .filter(
                self.disney_review.month == sl.Param(
                    "month_filter",
                    description="Filter by month of visit",
                    options=["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
                )
            )
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
            .limit(sl.Param("limit", description="Maximum number of reviews to return", default=20))
            .with_natural_query(sl.Param("natural_query"), self.openai_config)
        )
        print("✅ Natural language query created successfully")
    
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
    disney_nlq_query: Any = Field(default=None)
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve relevant Disney review documents using traditional or natural language queries."""
        
        # Check if natural language querying is requested and available
        use_natural_language = kwargs.get("use_natural_language", False) and self.disney_nlq_query is not None
        
        if use_natural_language:
            return self._get_documents_with_nlq(query, **kwargs)
        else:
            return self._get_documents_traditional(query, **kwargs)
    
    def _get_documents_with_nlq(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve documents using natural language query."""
        
        # Simple parameters for natural language query
        query_params = {
            "natural_query": query,
            "limit": kwargs.get("limit", 20)
        }
        
        # Allow override of specific parameters if provided
        if "sentiment_filter" in kwargs:
            query_params["sentiment_filter"] = kwargs["sentiment_filter"]
        if "visitor_type_filter" in kwargs:
            query_params["visitor_type_filter"] = kwargs["visitor_type_filter"]
        if "season_filter" in kwargs:
            query_params["season_filter"] = kwargs["season_filter"]
        
        try:
            results = self.sl_client.query(
                self.disney_nlq_query,
                **query_params
            )
            
            return self._process_results(results)
        
        except Exception as e:
            print(f"Error executing Disney natural language query: {e}")
            # Fallback to traditional query
            print("Falling back to traditional parameterized query...")
            return self._get_documents_traditional(query, **kwargs)
    
    def _get_documents_traditional(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve documents using traditional parameterized query."""
        
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
            
            "limit": kwargs.get("limit", 20),
            
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
            
            return self._process_results(results)
        
        except Exception as e:
            print(f"Error executing Disney review query: {e}")
            return []
    
    def _process_results(self, results) -> List[Document]:
        """Process Superlinked results into LangChain Document format."""
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
    
    # Create retriever with both traditional and natural language query support
    retriever = DisneyReviewRetriever(
        sl_client=app,
        disney_query=disney_rag.disney_query,
        disney_nlq_query=disney_rag.disney_nlq_query
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


def create_disney_rag_system_with_nlq(analyzed_df: pd.DataFrame) -> tuple[RetrievalQA, Any]:
    """
    Create a Disney Review RAG system with natural language query support.
    
    Args:
        analyzed_df: DataFrame with analyzed Disney review data
        
    Returns:
        Tuple of (RetrievalQA chain, Superlinked app) for direct querying
    """
    
    # Initialize RAG system
    disney_rag = DisneyReviewRAG()
    
    # Setup Superlinked components
    app = disney_rag.setup_superlinked()
    
    # Index the data
    indexed_count = disney_rag.index_data(analyzed_df)
    print(f"✅ Disney RAG system ready with {indexed_count} reviews")
    
    # Create retriever with natural language query support
    retriever = DisneyReviewRetriever(
        sl_client=app,
        disney_query=disney_rag.disney_query,
        disney_nlq_query=disney_rag.disney_nlq_query
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
    
    return qa_chain, app


def query_disney_with_natural_language(
    app, 
    disney_nlq_query, 
    natural_query: str, 
    limit: int = 20, 
    **override_params
) -> dict:
    """
    Query Disney reviews using natural language directly with Superlinked.
    
    Args:
        app: Superlinked app instance
        disney_nlq_query: Natural language query object
        natural_query: Natural language query string
        limit: Maximum number of results to return
        **override_params: Optional parameter overrides (e.g., sentiment_filter="positive")
        
    Returns:
        Dictionary with results and metadata
    """
    
    if disney_nlq_query is None:
        raise ValueError("Natural language querying not available. Check OpenAI API key configuration.")
    
    # Build query parameters
    query_params = {
        "natural_query": natural_query,
        "limit": limit,
        **override_params
    }
    
    try:
        # Execute the natural language query
        result = app.query(disney_nlq_query, **query_params)
        
        # Convert to pandas for easier viewing
        df_result = sl.PandasConverter.to_pandas(result)
        
        return {
            "results": df_result,
            "metadata": result.metadata,
            "extracted_params": result.metadata.search_params if hasattr(result.metadata, 'search_params') else {},
            "num_results": len(df_result)
        }
        
    except Exception as e:
        print(f"Error executing natural language query: {e}")
        return {
            "results": None,
            "metadata": None,
            "extracted_params": {},
            "num_results": 0,
            "error": str(e)
        }


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


