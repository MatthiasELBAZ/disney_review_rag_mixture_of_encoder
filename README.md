# Disney Review Analyzer 🏰

A comprehensive LangChain-powered tool for analyzing Disney theme park reviews with structured output and batch processing capabilities. Designed to process large datasets (40k+ reviews) efficiently while extracting detailed insights across multiple dimensions.

## Features ✨

### 🎯 Comprehensive Analysis Dimensions

1. **Overall Sentiment Analysis**
   - Positive, negative, neutral classification
   - Confidence scoring

2. **Aspect-Based Sentiment** 
   - Attractions/rides
   - Food & restaurants
   - Hotels & resorts
   - Staff friendliness
   - Price & value for money
   - Crowd management & waiting times
   - Cleanliness & safety
   - Accessibility (mobility, languages, dietary needs)

3. **Visitor Demographics** (inferred from text)
   - Families, couples, solo travelers, groups
   - International vs. local visitors
   - Presence of children

4. **Trip Context**
   - First-time visit, returning guest, frequent visitor
   - Special events (birthday, honeymoon, holidays)
   - Trip duration

5. **Time Dimension**
   - Season/month of visit
   - Day type (weekday/weekend/holiday)

6. **Purchase Mentions**
   - Tickets, fast passes, souvenirs, food, hotels, parking

7. **Engagement Signals**
   - Time spent, popular attractions mentioned
   - Intent to return or recommend

8. **Pain Points**
   - Journey stage where issues occurred
   - Severity assessment

### 🚀 Technical Features

- **Structured Output**: Pydantic models ensure consistent, validated results
- **Batch Processing**: LangChain batch API for efficient processing
- **Parallel Processing**: Multi-threaded execution for large datasets
- **Error Handling**: Retry logic and graceful failure handling
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Memory Efficient**: Processes data in configurable batches
- **Multiple LLM Support**: OpenAI, Google AI, and extensible to others

## Installation 📦

1. **Clone or download the files to your project directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
# or
GOOGLE_API_KEY=your_google_api_key_here
```

## Quick Start 🚀

### Option 1: Demo with Sample Reviews

```python
from disney_review_analyzer import DisneyReviewAnalyzer

# Initialize analyzer
analyzer = DisneyReviewAnalyzer(
    model_provider="openai",
    model_name="gpt-4o-mini",
    batch_size=10,
    max_workers=4
)

# Sample reviews
reviews = [
    "Amazing experience! The rides were fantastic and staff was very friendly...",
    "Too crowded and overpriced. Food was terrible but kids loved the parade..."
]

# Analyze reviews
results = analyzer.process_reviews_parallel(reviews)

# View results
for i, result in enumerate(results):
    if result:
        print(f"Review {i+1}: {result.overall_sentiment} "
              f"(confidence: {result.sentiment_confidence:.2f})")
```

### Option 2: Analyze Your CSV Dataset

```python
from disney_review_analyzer import DisneyReviewAnalyzer

# Initialize analyzer
analyzer = DisneyReviewAnalyzer(
    model_provider="openai",
    model_name="gpt-4o-mini",
    batch_size=20,  # Adjust based on your API limits
    max_workers=6   # Adjust based on your system
)

# Process CSV file
results_df = analyzer.analyze_csv_file(
    csv_path="path/to/your/reviews.csv",
    review_column="Review_Text",  # Adjust column name
    output_path="analysis_results.csv"
)

# Generate summary report
report = analyzer.generate_summary_report(results_df)
print(f"Analyzed {report['overview']['successful_analyses']} reviews")
print(f"Sentiment distribution: {report['sentiment_distribution']}")
```

### Option 3: Using the Example Script

```bash
python example_usage.py
```

Choose option 1 for a quick demo or option 2 to analyze your full dataset.

## Configuration ⚙️

### LLM Providers

**OpenAI (Recommended)**
```python
analyzer = DisneyReviewAnalyzer(
    model_provider="openai",
    model_name="gpt-4o-mini",  # Cost-effective
    # model_name="gpt-4o",     # Higher quality
)
```

**Google AI**
```python
analyzer = DisneyReviewAnalyzer(
    model_provider="google",
    model_name="gemini-1.5-flash",  # Fast and cost-effective
    # model_name="gemini-1.5-pro",   # Higher quality
)
```

### Performance Tuning

```python
analyzer = DisneyReviewAnalyzer(
    batch_size=50,      # Larger batches = faster, but more memory
    max_workers=8,      # More workers = faster, but more API usage
    max_retries=3,      # Retry failed requests
    temperature=0.1     # Lower = more consistent results
)
```

## Data Schema 📊

The analyzer produces structured output with the following main categories:

### Core Results
- `overall_sentiment`: positive/negative/neutral
- `sentiment_confidence`: 0.0 to 1.0

### Aspect Sentiments
- `attractions_sentiment`: Sentiment about rides/attractions
- `food_sentiment`: Sentiment about food and restaurants
- `staff_sentiment`: Sentiment about staff interactions
- `price_sentiment`: Sentiment about value for money
- ... (8 total aspects)

### Visitor Insights
- `visitor_type`: families/couples/solo/group
- `visitor_origin`: international/local
- `visit_frequency`: first_time/returning/frequent

### Behavioral Data
- `mentions_fast_passes`: Boolean
- `mentions_souvenirs`: Boolean
- `popular_attractions`: List of mentioned attractions
- `recommendation_intent`: yes/no/conditional

### Pain Points
- `main_complaint`: Primary issue mentioned
- `journey_stage`: Where problems occurred
- `pain_severity`: minor/moderate/major/trip_ruining

## Performance & Costs 💰

### Estimated Processing Times (40k reviews)

| Configuration | Time | Cost (OpenAI gpt-4o-mini) |
|---------------|------|---------------------------|
| batch_size=10, workers=4 | ~6 hours | ~$40-60 |
| batch_size=20, workers=6 | ~3 hours | ~$40-60 |
| batch_size=50, workers=8 | ~2 hours | ~$40-60 |

### Cost Optimization Tips

1. **Use gpt-4o-mini**: 85% cheaper than GPT-4, excellent for structured tasks
2. **Process samples first**: Test with 100-1000 reviews before full dataset
3. **Batch processing**: Use larger batch sizes when possible
4. **Monitor API limits**: Stay within rate limits to avoid delays

## Example Output 📋

```json
{
  "overall_sentiment": "positive",
  "sentiment_confidence": 0.85,
  "aspect_sentiments": {
    "attractions_rides": "positive",
    "food_restaurants": "negative", 
    "staff_friendliness": "positive",
    "price_value": "negative"
  },
  "visitor_demographics": {
    "visitor_type": "families",
    "has_children": true,
    "visitor_origin": "international"
  },
  "trip_context": {
    "visit_frequency": "first_time",
    "special_occasion": "birthday",
    "trip_duration": "weekend"
  },
  "engagement_signals": {
    "popular_attractions": ["Space Mountain", "Pirates of the Caribbean"],
    "recommendation_intent": "conditional"
  },
  "pain_points": {
    "main_complaint": "Long wait times and expensive food",
    "journey_stage": "during_visit",
    "severity": "moderate"
  }
}
```

## Advanced Usage 🔧

### Custom Processing Pipeline

```python
import pandas as pd
from disney_review_analyzer import DisneyReviewAnalyzer

def process_large_dataset(csv_path, chunk_size=1000):
    """Process large datasets in chunks to manage memory."""
    analyzer = DisneyReviewAnalyzer(batch_size=25, max_workers=6)
    
    all_results = []
    
    # Process in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        reviews = chunk['review_text'].tolist()
        results = analyzer.process_reviews_parallel(reviews)
        all_results.extend(results)
        
        print(f"Processed {len(all_results)} reviews...")
    
    return all_results
```

### Error Handling and Monitoring

```python
def analyze_with_monitoring(reviews):
    analyzer = DisneyReviewAnalyzer()
    
    def progress_callback(completed_batches, total_batches):
        progress = completed_batches / total_batches
        print(f"Progress: {progress:.1%} ({completed_batches}/{total_batches} batches)")
    
    results = analyzer.process_reviews_parallel(
        reviews, 
        progress_callback=progress_callback
    )
    
    # Check success rate
    successful = sum(1 for r in results if r is not None)
    print(f"Success rate: {successful/len(results):.1%}")
    
    return results
```

## Troubleshooting 🛠️

### Common Issues

1. **API Key Not Found**
   ```bash
   Error: OPENAI_API_KEY environment variable is required
   ```
   Solution: Set up your `.env` file with valid API keys

2. **Rate Limit Errors**
   ```
   Rate limit exceeded
   ```
   Solution: Reduce `batch_size` and `max_workers`, or upgrade API plan

3. **Memory Issues**
   ```
   Memory error processing large dataset
   ```
   Solution: Reduce `batch_size` or process data in chunks

4. **Column Not Found**
   ```
   Column 'Review_Text' not found in CSV
   ```
   Solution: Check your CSV column names and specify the correct `review_column`

### Performance Optimization

- Start with small samples (100-1000 reviews) to test configuration
- Monitor API usage and costs in your provider dashboard
- Use `gpt-4o-mini` for cost-effective processing
- Adjust batch size based on your API tier and rate limits

## Contributing 🤝

Feel free to submit issues, feature requests, or pull requests to improve the analyzer.

## License 📄

This project is open source. Please check the specific license file for details.

---

**Happy Analyzing! 🎢✨** 