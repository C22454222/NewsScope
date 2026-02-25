import os
import time
from app.db.supabase import supabase
from huggingface_hub import InferenceClient


HF_API_TOKEN = os.getenv("HF_API_TOKEN")

SENTIMENT_MODEL = os.getenv(
    "HF_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
).strip()

# matous-volf/political-leaning-politics:
#   Trained on 12 combined datasets | Accuracy: ~84.7% AllSides
#   Outputs: LABEL_0=Left, LABEL_1=Center, LABEL_2=Right
BIAS_MODEL = os.getenv(
    "HF_BIAS_MODEL",
    "matous-volf/political-leaning-politics",
).strip()

GENERAL_BIAS_MODEL = os.getenv(
    "HF_GENERAL_BIAS_MODEL",
    "valurank/distilroberta-bias",
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
            result = client.text_classification(
                truncated_text,
                model=model,
            )
            return result

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

            if "negative" in label:
                sentiment_map["negative"] = score
            elif "neutral" in label:
                sentiment_map["neutral"] = score
            elif "positive" in label:
                sentiment_map["positive"] = score

        neg = sentiment_map.get("negative", 0)
        pos = sentiment_map.get("positive", 0)

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
            "Right": 1.0,
        }

        return bias_map.get(rating, 0.0)

    except Exception:
        return 0.0


def _detect_political_bias_ai(text: str):
    """
    Political bias detection using political-leaning-politics model.

    Model outputs: LABEL_0=Left, LABEL_1=Center, LABEL_2=Right.

    Returns tuple: (bias_score: float, confidence: float)
    - bias_score: -1.0 (Left) to +1.0 (Right)
    - confidence: 0.0 to 1.0 (model confidence in top prediction)

    Returns (None, None) if AI fails.
    """
    try:
        results = _call_classification(BIAS_MODEL, text)
        if not results:
            return (None, None)

        predictions = [(item.label, item.score) for item in results]
        top_label, confidence = max(predictions, key=lambda x: x[1])

        # E241 fix: no alignment spaces after ':'
        # LABEL_0=Left, LABEL_1=Center, LABEL_2=Right
        label_map = {
            "LABEL_0": -1.0,
            "LABEL_1": 0.0,
            "LABEL_2": 1.0,
            "LEFT": -1.0,
            "CENTER": 0.0,
            "RIGHT": 1.0,
            "0": -1.0,
            "1": 0.0,
            "2": 1.0,
        }

        top_norm = top_label.strip()
        bias_score = label_map.get(
            top_norm,
            label_map.get(top_norm.upper(), 0.0),
        )

        # W503 fix: if/elif/else instead of multiline ternary
        if bias_score < -0.3:
            label_str = "LEFT"
        elif bias_score > 0.3:
            label_str = "RIGHT"
        else:
            label_str = "CENTER"

        print(
            f"Political bias: {label_str} "
            f"(score={bias_score:.2f}, confidence={confidence:.2%})"
        )

        return (bias_score, confidence)

    except Exception as e:
        print(f"Political bias detection error: {e}")
        return (None, None)


def _detect_general_bias(text: str):
    """
    General/lexical bias detection using valurank/distilroberta-bias.

    Detects whether article language is biased or unbiased
    regardless of political direction — framing, loaded language,
    and sensationalism per Spinde et al. (2023).

    Model returns NEUTRAL or BIASED labels.

    Returns tuple: (label: str, score: float)
    - label: 'BIASED' or 'UNBIASED'
    - score: 0.0 to 1.0 (confidence)

    Returns (None, None) if model fails.
    """
    try:
        results = _call_classification(GENERAL_BIAS_MODEL, text)
        if not results:
            return (None, None)

        print(f"   🔍 General bias raw results: {results}")

        predictions = [(item.label, item.score) for item in results]
        top_label, confidence = max(predictions, key=lambda x: x[1])
        top_upper = top_label.upper()

        if "BIASED" in top_upper and "UN" not in top_upper:
            label = "BIASED"
        elif (
            "UNBIASED" in top_upper or "NON" in top_upper or "NEUTRAL" in top_upper
        ):
            label = "UNBIASED"
        elif top_upper in ("LABEL_1", "1"):
            label = "BIASED"
        elif top_upper in ("LABEL_0", "0"):
            label = "UNBIASED"
        else:
            print(
                f"Unknown general bias label: '{top_label}' "
                "— defaulting to UNBIASED"
            )
            label = "UNBIASED"

        print(f"General bias: {label} (confidence={confidence:.2%})")

        return (label, confidence)

    except Exception as e:
        print(f"General bias detection error: {e}")
        return (None, None)


def _hybrid_bias_analysis(text: str, source_name: str):
    """
    Hybrid bias detection methodology.

    Combines:
    1. Article-level AI political bias
       (matous-volf/political-leaning-politics)
    2. Source-level political leaning (AllSides methodology fallback)

    Returns tuple: (bias_score, bias_intensity)
    - bias_score:     -1.0 (Left) to +1.0 (Right)
    - bias_intensity:  0.0 to 1.0 (model confidence)
    """
    ai_bias, ai_confidence = _detect_political_bias_ai(text)

    if ai_bias is not None and ai_confidence is not None:
        bias_intensity = round(ai_confidence, 4)

        # W503 fix: if/elif/else instead of multiline ternary
        if ai_bias < -0.3:
            label = "LEFT"
        elif ai_bias > 0.3:
            label = "RIGHT"
        else:
            label = "CENTER"

        print(
            f"Article AI result: {label} (score={ai_bias:.2f}, "
            f"confidence={ai_confidence:.2%}, "
            f"intensity={bias_intensity:.2f})"
        )
        return (ai_bias, bias_intensity)

    source_bias = _get_source_political_leaning(source_name)

    print(
        f"AI failed → fallback to source: "
        f"{source_name} = {source_bias:.2f}"
    )
    return (source_bias, 0.5)


def analyze_unscored_articles():
    """
    Analyze articles using hybrid transformer-based approach.

    Sentiment Analysis:
      - Model: cardiffnlp/twitter-roberta-base-sentiment-latest
      - Output: sentiment_score (-1 to +1)

    Political Bias Detection:
      - Model: matous-volf/political-leaning-politics
      - Output: bias_score (-1 to +1), bias_intensity (0 to 1)

    General/Lexical Bias Detection:
      - Model: valurank/distilroberta-bias
      - Output: general_bias (BIASED/UNBIASED), general_bias_score (0-1)

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
        "(hybrid methodology + general bias)..."
    )

    for article in articles:
        content = article.get("content") or ""
        title = article.get("title") or ""
        source = article.get("source") or ""

        text = f"{title}. {content}".strip()

        if len(text) < 10:
            print(f"Skipping article {article['id']} - no content")
            continue

        # 1. Sentiment analysis
        sentiment = _sentiment_score(text)
        time.sleep(1.0)

        # 2. Hybrid political bias detection
        bias_score, bias_intensity = _hybrid_bias_analysis(text, source)
        time.sleep(1.0)

        # 3. General/lexical bias detection
        general_bias_label, general_bias_score = _detect_general_bias(text)

        sent_str = (
            f"{sentiment:.3f}" if sentiment is not None else "N/A"
        )
        bias_str = (
            f"{bias_score:.2f}" if bias_score is not None else "N/A"
        )
        intensity_str = (
            f"{bias_intensity:.2f}" if bias_intensity is not None
            else "N/A"
        )

        print(
            f"Article {article['id']}: "
            f"Sentiment={sent_str}, "
            f"Bias={bias_str}, "
            f"Intensity={intensity_str}, "
            f"GeneralBias={general_bias_label} "
            f"({source})"
        )

        update_data = {}
        if sentiment is not None:
            update_data["sentiment_score"] = sentiment
        if bias_score is not None:
            update_data["bias_score"] = bias_score
        if bias_intensity is not None:
            update_data["bias_intensity"] = bias_intensity
        if general_bias_label is not None:
            update_data["general_bias"] = general_bias_label
        if general_bias_score is not None:
            update_data["general_bias_score"] = general_bias_score

        if update_data:
            supabase.table("articles").update(
                update_data,
            ).eq("id", article["id"]).execute()

    print("Analysis job complete.")
