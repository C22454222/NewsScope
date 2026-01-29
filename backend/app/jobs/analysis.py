# app/jobs/analysis.py
import os
import time
from app.db.supabase import supabase
from huggingface_hub import InferenceClient


HF_API_TOKEN = os.getenv("HF_API_TOKEN")

SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
).strip()

BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "premsa/political-bias-prediction-allsides-BERT"
).strip()

client = InferenceClient(token=HF_API_TOKEN)


def _call_classification(model: str, text: str):
    """
    Safe wrapper around InferenceClient.text_classification.
    Retries on model loading errors (503).
    """
    truncated_text = text[:512]

    for attempt in range(3):
        try:
            return client.text_classification(
                truncated_text,
                model=model
            )

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "loading" in error_str.lower():
                print(
                    f"Model {model} loading... "
                    f"waiting 10s (attempt {attempt + 1}/3)"
                )
                time.sleep(10)
                continue

            print(f"Analysis error for {model}: {e}")
            return None

    print(f"Model {model} failed after 3 retries")
    return None


def _sentiment_score(text: str):
    """
    Analyze sentiment using RoBERTa sentiment model.
    Returns float: -1 (negative) to +1 (positive).
    """
    try:
        results = _call_classification(SENTIMENT_MODEL, text)
        if not results:
            return None

        sentiment_map = {}
        for item in results:
            label = item.label.lower()
            score = item.score

            if 'negative' in label:
                sentiment_map['negative'] = score
            elif 'neutral' in label:
                sentiment_map['neutral'] = score
            elif 'positive' in label:
                sentiment_map['positive'] = score

        neg = sentiment_map.get('negative', 0)
        pos = sentiment_map.get('positive', 0)

        return pos - neg

    except Exception as e:
        print(f"Sentiment parsing error: {e}")
        return None


def _get_source_political_leaning(source_name: str):
    """
    Retrieve source's political leaning from database.
    Uses AllSides/Ground News methodology.
    Returns float: -1 (Left) to +1 (Right).
    """
    try:
        response = (
            supabase.table("sources")
            .select("bias_rating")
            .eq("name", source_name)
            .single()
            .execute()
        )

        rating = response.data.get("bias_rating")

        bias_map = {
            "Left": -1.0,
            "Center-Left": -0.5,
            "Center": 0.0,
            "Center-Right": 0.5,
            "Right": 1.0
        }

        return bias_map.get(rating, 0.0)

    except Exception:
        return 0.0


def _detect_political_bias_ai(text: str):
    """
    Article-level political bias detection using politicalBiasBERT.

    Returns tuple: (bias_score: float, confidence: float)
    - bias_score: -1.0 (Left) to +1.0 (Right)
    - confidence: 0.0 to 1.0 (model confidence)

    Returns (None, None) if AI fails.
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return (None, None)

        # politicalBiasBERT returns: left, center, right
        # Find the prediction with highest score
        predictions = {item.label.lower(): item.score for item in results}

        # Get the top prediction
        top_label = max(predictions, key=predictions.get)
        confidence = predictions[top_label]

        # Map to -1.0 to +1.0 scale
        bias_map = {
            'left': -1.0,
            'center': 0.0,
            'right': 1.0,
        }

        bias_score = bias_map.get(top_label, 0.0)

        return (bias_score, confidence)

    except Exception as e:
        print(f"Political bias detection error: {e}")
        return (None, None)


def _hybrid_bias_analysis(text: str, source_name: str):
    """
    Hybrid bias detection methodology combining:
    1. Article-level AI political bias (politicalBiasBERT)
    2. Source-level political leaning (AllSides methodology)

    Returns tuple: (bias_score, bias_intensity)
    - bias_score: -1.0 (Left) to +1.0 (Right)
    - bias_intensity: 0.0 (neutral) to 1.0 (strong bias)

    Logic:
    - Try AI first (article-level political classification)
    - If AI succeeds: use AI score, intensity = distance from center
    - If AI fails: fallback to source rating, moderate intensity
    """
    # Step 1: Try AI-based article-level bias detection
    ai_bias, ai_confidence = _detect_political_bias_ai(text)

    if ai_bias is not None:
        # AI succeeded - use article-level classification
        bias_intensity = abs(ai_bias)  # Distance from center (0.0 to 1.0)

        label = "LEFT" if ai_bias < -0.3 else "RIGHT" if ai_bias > 0.3 else "CENTER"

        print(
            f"   ✅ AI: {label} (score={ai_bias:.2f}, "
            f"confidence={ai_confidence:.2%}, intensity={bias_intensity:.2f})"
        )
        return (ai_bias, bias_intensity)

    # Step 2: Fallback to source-level bias
    source_bias = _get_source_political_leaning(source_name)
    fallback_intensity = 0.5  # Moderate intensity for fallback

    print(
        f"   ⚠️  AI failed → fallback to source: "
        f"{source_name} = {source_bias:.2f}"
    )
    return (source_bias, fallback_intensity)


def analyze_unscored_articles():
    """
    Analyze articles using hybrid transformer-based approach.

    Sentiment Analysis:
      - Model: RoBERTa sentiment (CardiffNLP)
      - Method: Article-level sentiment classification
      - Output: sentiment_score (-1 to +1)

    Political Bias Detection:
      - Model: politicalBiasBERT (bucketresearch)
      - Method: Hybrid article-level AI + source-level fallback
      - Output: bias_score (-1 to +1), bias_intensity (0 to 1)

    Runs every hour at :15 (15 minutes after ingestion).
    """
    print("Starting analysis job...")

    response = (
        supabase.table("articles")
        .select("id, content, title, source")
        .is_("sentiment_score", "null")
        .limit(500)
        .execute()
    )
    articles = response.data

    if not articles:
        print("No unscored articles found.")
        return

    print(
        f"Analyzing {len(articles)} articles "
        f"(hybrid methodology)..."
    )

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""
        source = article.get("source") or ""

        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(f"Skipping article {article['id']} - no content")
            continue

        # Sentiment analysis (AI)
        sentiment = _sentiment_score(text)
        time.sleep(1.0)

        # Hybrid political bias detection (AI + source fallback)
        bias_score, bias_intensity = _hybrid_bias_analysis(text, source)

        print(
            f"Article {article['id']}: "
            f"Sentiment={sentiment:.3f}, "
            f"Bias={bias_score:.2f}, "
            f"Intensity={bias_intensity:.2f} "
            f"({source})"
        )

        # Update database
        update_data = {}
        if sentiment is not None:
            update_data["sentiment_score"] = sentiment
        if bias_score is not None:
            update_data["bias_score"] = bias_score
        if bias_intensity is not None:
            update_data["bias_intensity"] = bias_intensity

        if update_data:
            supabase.table("articles").update(
                update_data
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
