"""
Disney Review Analyzer - Comprehensive Sentiment and Content Analysis
====================================================================

This module uses LangChain with structured output to analyze Disney reviews across multiple dimensions:
- Overall sentiment analysis (positive, negative, neutral)
- Aspect-based sentiment for 8 key areas 
- Visitor demographics and trip context
- Time dimension analysis
- Purchase mentions and engagement signals
- Pain point identification

Features:
- Async batch processing for large datasets (40k+ reviews)
- Structured output with Pydantic models
- Progress tracking and error handling
- Memory-efficient processing
- Configurable concurrency control
- OpenAI LLM provider support
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_MESSAGE = """You are an expert analyst specializing in customer experience and sentiment analysis for theme parks and entertainment venues. Your task is to analyze Disney theme park reviews comprehensively.

ANALYSIS REQUIREMENTS:

1. OVERALL SENTIMENT: Determine if the review is positive, negative, or neutral with a confidence score.

2. ASPECT-BASED SENTIMENT: Analyze sentiment for each aspect ONLY if mentioned in the review:
   - Attractions/rides
   - Food & restaurants  
   - Hotels & resorts
   - Staff friendliness
   - Price & value for money
   - Crowd management & waiting times
   - Cleanliness & safety
   - Accessibility (mobility, languages, dietary needs)
   
   Use "not_mentioned" if an aspect isn't discussed.

3. VISITOR DEMOGRAPHICS: Infer from text clues:
   - Visitor type (families, couples, solo travelers, groups)
   - Origin (international vs local visitors)
   - Presence of children

4. TRIP CONTEXT:
   - Visit frequency (first-time, returning, frequent visitor)
   - Special occasions (birthday, honeymoon, holidays, etc.)
   - Trip duration

5. TIME DIMENSION:
   - Season/month if mentioned
   - Day type (weekday/weekend/holiday)

6. PURCHASE MENTIONS: Note any mentions of:
   - Tickets, fast passes, souvenirs, food, hotels, parking

7. ENGAGEMENT SIGNALS:
   - Time spent
   - Specific attractions mentioned
   - Intent to return or recommend

8. PAIN POINTS:
   - Main complaints
   - Where in the journey problems occurred
   - Severity of issues

IMPORTANT GUIDELINES:
- Base analysis ONLY on what's explicitly stated or clearly implied
- Use "unknown" or "not_mentioned" when information isn't available
- Be conservative with inferences
- Pay attention to context and nuance
- Consider sarcasm and mixed sentiments"""

HUMAN_MESSAGE = """Please analyze this Disney theme park review:

REVIEW TEXT:
{review_text}

Provide a comprehensive analysis following the structured format. Ensure all fields are filled appropriately based on the content."""


class AspectSentiment(BaseModel):
    """Sentiment analysis for specific aspects of the Disney experience."""
    
    attractions_rides: Optional[str] = Field(
        None, 
        description="Sentiment about attractions and rides (positive/negative/neutral/not_mentioned)"
    )
    food_restaurants: Optional[str] = Field(
        None, 
        description="Sentiment about food and restaurants (positive/negative/neutral/not_mentioned)"
    )
    hotels_resorts: Optional[str] = Field(
        None, 
        description="Sentiment about hotels and resorts (positive/negative/neutral/not_mentioned)"
    )
    staff_friendliness: Optional[str] = Field(
        None, 
        description="Sentiment about staff friendliness (positive/negative/neutral/not_mentioned)"
    )
    price_value: Optional[str] = Field(
        None, 
        description="Sentiment about price and value for money (positive/negative/neutral/not_mentioned)"
    )
    crowd_waiting: Optional[str] = Field(
        None, 
        description="Sentiment about crowd management and waiting times (positive/negative/neutral/not_mentioned)"
    )
    cleanliness_safety: Optional[str] = Field(
        None, 
        description="Sentiment about cleanliness and safety (positive/negative/neutral/not_mentioned)"
    )
    accessibility: Optional[str] = Field(
        None, 
        description="Sentiment about accessibility (mobility, languages, dietary needs) (positive/negative/neutral/not_mentioned)"
    )


class VisitorDemographics(BaseModel):
    """Inferred visitor demographics from review text."""
    
    visitor_type: Optional[str] = Field(
        None, 
        description="Type of visitors (families/couples/solo/group/unknown)"
    )
    visitor_origin: Optional[str] = Field(
        None, 
        description="Visitor origin (international/local/unknown)"
    )
    has_children: Optional[bool] = Field(
        None, 
        description="Whether the review mentions traveling with children"
    )


class TripContext(BaseModel):
    """Context about the trip and visit."""
    
    visit_frequency: Optional[str] = Field(
        None, 
        description="Visit frequency (first_time/returning/frequent/unknown)"
    )
    special_occasion: Optional[str] = Field(
        None, 
        description="Special occasion mentioned (birthday/honeymoon/anniversary/holiday/graduation/none/unknown)"
    )
    trip_duration: Optional[str] = Field(
        None, 
        description="Duration of stay (day_trip/weekend/week/extended/unknown)"
    )


class TimeDimension(BaseModel):
    """Time-related information about the visit."""
    
    season: Optional[str] = Field(
        None, 
        description="Season of visit (spring/summer/fall/winter/unknown)"
    )
    month: Optional[str] = Field(
        None, 
        description="Month of visit if mentioned (january through december or unknown)"
    )
    day_type: Optional[str] = Field(
        None, 
        description="Type of day (weekday/weekend/holiday/unknown)"
    )


class PurchaseMentions(BaseModel):
    """Mentions of purchases and spending."""
    
    tickets: bool = Field(False, description="Whether tickets are mentioned")
    fast_passes: bool = Field(False, description="Whether fast passes/priority access is mentioned")
    souvenirs: bool = Field(False, description="Whether souvenirs are mentioned")
    food_purchases: bool = Field(False, description="Whether food purchases are mentioned")
    hotel_booking: bool = Field(False, description="Whether hotel booking is mentioned")
    parking: bool = Field(False, description="Whether parking fees are mentioned")


class EngagementSignals(BaseModel):
    """Signals indicating visitor engagement and priorities."""
    
    time_spent: Optional[str] = Field(
        None, 
        description="Duration mentioned for the visit (hours/full_day/multiple_days/unknown)"
    )
    popular_attractions: List[str] = Field(
        default_factory=list, 
        description="Specific attractions, rides, or shows mentioned by name"
    )
    repeat_visits: bool = Field(
        False, 
        description="Whether the visitor mentions wanting to return or has visited before"
    )
    recommendation_intent: Optional[str] = Field(
        None, 
        description="Whether they would recommend to others (yes/no/conditional/unknown)"
    )


class PainPoints(BaseModel):
    """Identified pain points and negative experiences."""
    
    main_complaint: Optional[str] = Field(
        None, 
        description="Primary complaint or issue mentioned"
    )
    journey_stage: Optional[str] = Field(
        None, 
        description="Stage where enthusiasm dropped (arrival/entry/during_visit/exit/unknown)"
    )
    severity: Optional[str] = Field(
        None, 
        description="Severity of pain point (minor/moderate/major/trip_ruining/unknown)"
    )


class DisneyReviewAnalysis(BaseModel):
    """Complete analysis of a Disney review."""
    
    # Core sentiment
    overall_sentiment: str = Field(
        description="Overall sentiment of the review (positive/negative/neutral)"
    )
    sentiment_confidence: float = Field(
        description="Confidence score for overall sentiment (0.0 to 1.0)"
    )
    
    # Aspect-based analysis
    aspect_sentiments: AspectSentiment = Field(
        description="Sentiment analysis for specific aspects"
    )
    
    # Demographics and context
    visitor_demographics: VisitorDemographics = Field(
        description="Inferred visitor demographics"
    )
    trip_context: TripContext = Field(
        description="Trip context and circumstances"
    )
    time_dimension: TimeDimension = Field(
        description="Time-related information"
    )
    
    # Behavioral insights
    purchase_mentions: PurchaseMentions = Field(
        description="Mentions of purchases and spending"
    )
    engagement_signals: EngagementSignals = Field(
        description="Engagement and priority signals"
    )
    pain_points: PainPoints = Field(
        description="Pain points and negative experiences"
    )
    
    # Meta information
    review_length: str = Field(
        description="Length category of the review (short/medium/long)"
    )
    language_quality: str = Field(
        description="Quality of language used (poor/fair/good/excellent)"
    )

    @validator('sentiment_confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class DisneyReviewAnalyzer:
    """Main analyzer class for processing Disney reviews."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrency: int = 5
    ):
        """
        Initialize the Disney Review Analyzer.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for LLM responses
            max_retries: Maximum retry attempts for failed requests
            batch_size: Number of reviews to process in each batch
            max_concurrency: Maximum number of concurrent async requests
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # Setup structured output
        self.structured_llm = self.llm.with_structured_output(DisneyReviewAnalysis)
        
        # Create prompt template
        self.prompt = self._create_prompt_template()
        
        # Create processing chain
        self.analysis_chain = self.prompt | self.structured_llm
        
        logger.info(f"DisneyReviewAnalyzer initialized with OpenAI {model_name}")
    
    def _setup_llm(self):
        """Setup the OpenAI LLM."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for analysis."""

        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            ("human", HUMAN_MESSAGE)
        ])
    
    async def analyze_single_review_async(self, review_text: str) -> Optional[DisneyReviewAnalysis]:
        """
        Analyze a single review asynchronously with retry logic.
        
        Args:
            review_text: The review text to analyze
            
        Returns:
            DisneyReviewAnalysis object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                result = await self.analysis_chain.ainvoke({"review_text": review_text})
                return result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to analyze review after {self.max_retries} attempts")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    async def analyze_batch_async(self, reviews: List[str]) -> List[Optional[DisneyReviewAnalysis]]:
        """
        Analyze a batch of reviews using LangChain's async batch functionality.
        
        Args:
            reviews: List of review texts
            
        Returns:
            List of DisneyReviewAnalysis objects (or None for failed analyses)
        """
        try:
            # Prepare inputs for batch processing
            inputs = [{"review_text": review} for review in reviews]
            
            # Use LangChain's abatch method with max_concurrency control
            config = {"max_concurrency": self.max_concurrency}
            results = await self.analysis_chain.abatch(inputs, config=config)
            return results
            
        except Exception as e:
            logger.error(f"Async batch processing failed: {str(e)}")
            # Fallback to individual async processing
            logger.info("Falling back to individual async processing...")
            tasks = [self.analyze_single_review_async(review) for review in reviews]
            return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def process_reviews_async(
        self, 
        reviews: List[str], 
        progress_callback: Optional[callable] = None
    ) -> List[Optional[DisneyReviewAnalysis]]:
        """
        Process reviews in async batches for maximum efficiency.
        
        Args:
            reviews: List of review texts
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of DisneyReviewAnalysis objects
        """
        total_reviews = len(reviews)
        results = [None] * total_reviews
        
        # Create batches
        batches = [
            reviews[i:i + self.batch_size] 
            for i in range(0, total_reviews, self.batch_size)
        ]
        
        logger.info(f"Processing {total_reviews} reviews in {len(batches)} batches with max concurrency {self.max_concurrency}")
        
        # Process batches with progress tracking using tqdm
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    batch_results = await self.analyze_batch_async(batch)
                    
                    # Store results in correct positions
                    start_idx = batch_idx * self.batch_size
                    for i, result in enumerate(batch_results):
                        if start_idx + i < total_reviews:
                            results[start_idx + i] = result
                    
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(batch_idx + 1, len(batches))
                        
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    # Fill with None for failed batch
                    start_idx = batch_idx * self.batch_size
                    for i in range(len(batch)):
                        if start_idx + i < total_reviews:
                            results[start_idx + i] = None
                    pbar.update(1)
        
        return results
    
    async def analyze_df_async(
        self, 
        df: pd.DataFrame, 
        review_column: str = 'Review_Text',
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyze reviews from a DataFrame asynchronously.
        
        Args:
            df: DataFrame containing the reviews
            review_column: Name of the column containing review text
            output_path: Path to save results (optional)
            save_interval: Save progress every N reviews
            
        Returns:
            DataFrame with original data plus analysis results
        """
        
        if review_column not in df.columns:
            raise ValueError(f"Column '{review_column}' not found in DataFrame")
        
        # Clean and prepare reviews
        reviews = df[review_column].astype(str).tolist()
        
        logger.info(f"Found {len(reviews)} reviews to analyze")
        
        # Process reviews
        def progress_callback(completed_batches, total_batches):
            logger.info(f"Completed {completed_batches}/{total_batches} batches")
        
        results = await self.process_reviews_async(reviews, progress_callback)
        
        # Convert results to DataFrame columns
        analysis_data = []
        
        for i, result in enumerate(results):
            if result is not None:
                analysis_data.append({
                    'index': i,
                    'overall_sentiment': result.overall_sentiment,
                    'sentiment_confidence': result.sentiment_confidence,
                    'attractions_sentiment': result.aspect_sentiments.attractions_rides,
                    'food_sentiment': result.aspect_sentiments.food_restaurants,
                    'hotels_sentiment': result.aspect_sentiments.hotels_resorts,
                    'staff_sentiment': result.aspect_sentiments.staff_friendliness,
                    'price_sentiment': result.aspect_sentiments.price_value,
                    'crowd_sentiment': result.aspect_sentiments.crowd_waiting,
                    'cleanliness_sentiment': result.aspect_sentiments.cleanliness_safety,
                    'accessibility_sentiment': result.aspect_sentiments.accessibility,
                    'visitor_type': result.visitor_demographics.visitor_type,
                    'visitor_origin': result.visitor_demographics.visitor_origin,
                    'has_children': result.visitor_demographics.has_children,
                    'visit_frequency': result.trip_context.visit_frequency,
                    'special_occasion': result.trip_context.special_occasion,
                    'trip_duration': result.trip_context.trip_duration,
                    # 'season': result.time_dimension.season,
                    # 'month': result.time_dimension.month,
                    'day_type': result.time_dimension.day_type,
                    'mentions_tickets': result.purchase_mentions.tickets,
                    'mentions_fast_passes': result.purchase_mentions.fast_passes,
                    'mentions_souvenirs': result.purchase_mentions.souvenirs,
                    'mentions_food_purchases': result.purchase_mentions.food_purchases,
                    'mentions_hotel_booking': result.purchase_mentions.hotel_booking,
                    'mentions_parking': result.purchase_mentions.parking,
                    'time_spent': result.engagement_signals.time_spent,
                    'popular_attractions': ';'.join(result.engagement_signals.popular_attractions),
                    'repeat_visits': result.engagement_signals.repeat_visits,
                    'recommendation_intent': result.engagement_signals.recommendation_intent,
                    'main_complaint': result.pain_points.main_complaint,
                    'journey_stage': result.pain_points.journey_stage,
                    'pain_severity': result.pain_points.severity,
                    'review_length': result.review_length,
                    'language_quality': result.language_quality,
                    'analysis_successful': True
                })
            else:
                analysis_data.append({
                    'index': i,
                    'analysis_successful': False
                })
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame(analysis_data)
        
        # Merge with original data
        result_df = df.copy()
        result_df = result_df.reset_index(drop=True)
        
        # Add analysis columns
        for col in analysis_df.columns:
            if col != 'index':
                result_df[col] = analysis_df[col]
        
        # Save results if path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        # Log summary statistics
        successful_analyses = analysis_df['analysis_successful'].sum()
        logger.info(f"Successfully analyzed {successful_analyses}/{len(reviews)} reviews")
        
        return result_df
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary report from analyzed reviews.
        
        Args:
            df: DataFrame with analysis results
            
        Returns:
            Dictionary containing summary statistics
        """
        successful_df = df[df['analysis_successful'] == True]
        
        if len(successful_df) == 0:
            return {"error": "No successful analyses to summarize"}
        
        report = {
            "overview": {
                "total_reviews": len(df),
                "successful_analyses": len(successful_df),
                "success_rate": len(successful_df) / len(df)
            },
            "sentiment_distribution": {
                "positive": (successful_df['overall_sentiment'] == 'positive').sum(),
                "negative": (successful_df['overall_sentiment'] == 'negative').sum(),
                "neutral": (successful_df['overall_sentiment'] == 'neutral').sum()
            },
            "aspect_sentiments": {},
            "visitor_insights": {},
            "temporal_patterns": {},
            "engagement_metrics": {}
        }
        
        # Aspect sentiment analysis
        aspect_columns = [col for col in successful_df.columns if col.endswith('_sentiment')]
        for col in aspect_columns:
            aspect_name = col.replace('_sentiment', '')
            aspect_data = successful_df[col].value_counts()
            report["aspect_sentiments"][aspect_name] = aspect_data.to_dict()
        
        # Visitor insights
        if 'visitor_type' in successful_df.columns:
            report["visitor_insights"]["visitor_types"] = successful_df['visitor_type'].value_counts().to_dict()
        
        if 'visit_frequency' in successful_df.columns:
            report["visitor_insights"]["visit_frequency"] = successful_df['visit_frequency'].value_counts().to_dict()
        
        # Temporal patterns
        if 'season' in successful_df.columns:
            report["temporal_patterns"]["seasons"] = successful_df['season'].value_counts().to_dict()
        
        # Engagement metrics
        if 'recommendation_intent' in successful_df.columns:
            report["engagement_metrics"]["recommendation_intent"] = successful_df['recommendation_intent'].value_counts().to_dict()
        
        return report


async def main():
    """Example usage of the DisneyReviewAnalyzer."""
    # Example reviews for testing
    sample_reviews = [
        "First of all, I am a Disney nut! Multiple trips to Disneyland and Disney World, Multiple Disney cruises, and a Disney Vacation Club member. Never having been to a Park outside the US, I did a lot of reading of reviews and comments. I realized that it wouldn't be Disney World, and I really downgraded my expectations based on what I read. I was completely surprised by my experience at Disneyland Paris. The Park is beautiful. It's smaller than Disney World, but what isn't? I thought the landscaping, the buildings, the restrooms and the rides were extremely well kept up. And in the Parks and hotel, I had nothing but positive interactions with Cast members. So much for the negativity I anticipated.The rides and attractions were a combination of familiar and new. And many of the familiar had different twists. Big Thunder Mountain is much more of a thrill ride, starting with the initial rush under the lake to get to the  mountain . Pirates of the Caribbean is a longer ride with more theming. Star Wars Hyperspace Mountain seemed a bigger thrill ride here. Buzz Lightyear, Star Tours, and It's a Small World were all familiar and well done. A couple walk thrus that were unique were the Sleeping Beauty Castle with Stained glass windows on the balcony, and a dragon in the basement. I also really enjoyed the walk through of Alice's Curious Labyrinth. My only disappointment was that Phantom Manor and the Indiana Jones ride were closed for refurbishment. Lots of the normal Disney option for dining. We enjoyed the Moroccan buffet at the Agrabah Cafe.This Disney Park did not let me down. Go with a positive attitude. Oh, and if you go in November, there will still be lines, but it will be cold.",
        "a fantasy world and the attractions are too good. since we went in Aug, it was crowded everywhere and maximum time we waited was about 45min for each attraction.a day is not enough, i would recommend people to stay in the hotel to enjoy all the parks.indiana jones, space mountain are thrilling but above all tower of terror is so scary but fun at the same time. pirates des caraibes and small world have been perfectlydesigned and animated.the parade show at 17h is spectacular and the fireworks show in the end (around 23h) is breathtaking. not to be missed at all!only big problem is there is no halal food in the parks, we had to survive on chips and veg foods. the gift shop has a variety of souvenir but the prices are quite expensive."
    ]
    
    # Initialize analyzer
    analyzer = DisneyReviewAnalyzer(
        model_name="gpt-4o-mini",
        batch_size=5,
        max_concurrency=3
    )
    
    # Analyze sample reviews
    print("Analyzing sample reviews...")
    results = await analyzer.process_reviews_async(sample_reviews)
    
    # Print results
    for i, result in enumerate(results):
        if result:
            print(f"\n=== Review {i+1} Analysis ===")
            print(f"Overall Sentiment: {result.overall_sentiment} (confidence: {result.sentiment_confidence:.2f})")
            print(f"Visitor Type: {result.visitor_demographics.visitor_type}")
            print(f"Visit Frequency: {result.trip_context.visit_frequency}")
            print(f"Season: {result.time_dimension.season}")
            print(f"Main Complaint: {result.pain_points.main_complaint}")
        else:
            print(f"Review {i+1}: Analysis failed")


if __name__ == "__main__":
    asyncio.run(main()) 