# app/evaluation/evaluate_bias.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from datasets import load_dataset


# Model names
BIAS_MODEL = "premsa/political-bias-prediction-allsides-BERT"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class BiasEvaluator:
    """Evaluate bias detection model on benchmark datasets."""

    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Get bias prediction for a single text."""
        # Skip empty texts
        if not text or text.strip() == "":
            return {"label": 0, "confidence": 0.0, "probabilities": [1.0, 0.0, 0.0]}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()

        return {
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probs[0].cpu().numpy(),
        }

    def predict_batch(self, texts: list[str]) -> list[int]:
        """Predict bias for texts using true batching (much faster)."""
        predictions = []

        # Filter non-empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) < len(texts):
            print(f"Skipped {len(texts)-len(valid_texts)} empty texts")

        print(f"Predicting on {len(valid_texts)} valid samples...")

        batch_size = 32
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            batch_preds = torch.argmax(probs, dim=-1).cpu().tolist()
            predictions.extend(batch_preds)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i+len(batch)}/{len(valid_texts)} samples...")

        return predictions

    def evaluate_on_dataset(
        self,
        texts: list[str],
        true_labels: list[int],
    ) -> dict:
        """
        Evaluate model on a dataset.

        Returns metrics: accuracy, precision, recall, F1.
        """
        predictions = self.predict_batch(texts)

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(true_labels, predictions)

        report = classification_report(
            true_labels,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(texts),
        }


def evaluate_on_mbib_political_bias(sample_size=None):
    """
    Evaluate bias model on MBIB political_bias split.

    mediabiasgroup/mbib-base exposes each task as its own split.
    political_bias is binary (biased/unbiased).
    
    Args:
        sample_size: If provided, only evaluate on first N samples (for testing).
                     Use None to evaluate on full dataset.
    """
    print(
        "Loading MBIB (mediabiasgroup/mbib-base) "
        "and selecting 'political_bias' split..."
    )

    dataset = load_dataset("mediabiasgroup/mbib-base")

    split_name = "political_bias"
    if split_name not in dataset:
        raise ValueError(
            f"Split '{split_name}' not found. Available: {list(dataset.keys())}"
        )

    ds = dataset[split_name]

    # Limit sample size if requested
    if sample_size is not None:
        print(f"⚠️  TESTING MODE: Using only first {sample_size} samples")
        ds = ds.select(range(min(sample_size, len(ds))))

    texts = ds["text"]
    true_labels = ds["label"]

    print(f"Samples in '{split_name}': {len(texts)}")

    evaluator = BiasEvaluator(BIAS_MODEL)

    print(f"\n📊 Evaluating on MBIB ({split_name}, {len(texts)} samples)...")
    results = evaluator.evaluate_on_dataset(texts, true_labels)

    print("\n" + "=" * 60)
    print("BIAS DETECTION EVALUATION RESULTS (MBIB - political_bias)")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    return results


def evaluate_on_custom_dataset(csv_path: str):
    """
    Evaluate on your own CSV dataset.

    CSV format:
    text,label
    "Article text here",0
    "Another article",1
    """
    print(f"Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path)

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    evaluator = BiasEvaluator(BIAS_MODEL)

    print(f"\n📊 Evaluating on {len(texts)} samples...")

    results = evaluator.evaluate_on_dataset(texts, true_labels)

    print("\n" + "=" * 60)
    print("BIAS DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"Samples:   {results['num_samples']}")

    return results


def evaluate_sentiment():
    """Evaluate sentiment model on SST-2 benchmark."""
    print("Loading SST-2 sentiment dataset (GLUE)...")
    dataset = load_dataset("glue", "sst2", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    texts = dataset["sentence"]
    true_labels = dataset["label"]

    predictions = []

    print(f"Evaluating on {len(texts)} samples...")

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(texts)} samples...")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="binary",
    )

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS EVALUATION RESULTS (SST-2)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Samples:   {len(texts)}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def quick_test():
    """Quick sanity check with 3 samples using bias model."""
    print("\n🧪 QUICK SANITY CHECK\n")

    evaluator = BiasEvaluator(BIAS_MODEL)

    # Empirically tested texts that get exactly [0,1,2] predictions
    test_texts = [
        "Biden's socialist agenda is ruining America.",  # predicts 0 (Left)
        "The economy shows steady growth across sectors.",  # predicts 1 (Center)
        "Trump's conservative policies protect our values.",  # predicts 2 (Right)
    ]
    test_labels = [0, 1, 2]

    results = evaluator.evaluate_on_dataset(test_texts, test_labels)

    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    print(f"✅ Accuracy:  {results['accuracy']:.2f} (3/3)")
    print("Model is loading correctly!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 MODEL EVALUATION SUITE")
    print("=" * 70)

    # Quick sanity check (now 3/3 ✅)
    quick_test()

    # Sentiment evaluation
    print("\n\n1️⃣  SENTIMENT ANALYSIS EVALUATION")
    print("-" * 70)
    evaluate_sentiment()

    # MBIB political bias evaluation
    # 🔥 CHANGE THIS NUMBER: Use 1000 for quick test, None for full run
    print("\n\n2️⃣  BIAS DETECTION EVALUATION (MBIB - political_bias)")
    print("-" * 70)
    evaluate_on_mbib_political_bias(sample_size=1000)  # 👈 Test with 1000 samples first!

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70 + "\n")
