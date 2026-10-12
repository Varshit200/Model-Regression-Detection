
from app.classifier import classify_email


def main():
    print("Starting Model Regression Detection System...")

    email = (
        "I was charged twice for my monthly subscription. "
        "Please refund the extra payment."
    )

    print("Classifying customer email...")

    result = classify_email(email)

    print("\n--- Classification Result ---")
    print("Category:", result.category)
    print("Summary:", result.summary)
    print("\nJSON output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
