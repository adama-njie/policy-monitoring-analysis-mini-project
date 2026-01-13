"""
NLTK Data Setup Script
======================
Run this script BEFORE running the preprocessing pipeline to ensure all NLTK data is downloaded.
"""

import nltk
import ssl

# Fix SSL certificate issues if they occur
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading required NLTK data...")
print("=" * 60)

# List of required NLTK data packages
nltk_packages = [
    'punkt',
    'punkt_tab',
    'stopwords',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'wordnet',
    'omw-1.4'
]

for package in nltk_packages:
    print(f"Downloading '{package}'...", end=" ")
    try:
        nltk.download(package, quiet=True)
        print("✓ Done")
    except Exception as e:
        print(f"✗ Error: {e}")

print("=" * 60)
print("NLTK data download complete!")
print("\nVerifying installation...")

# Test tokenization
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    test_text = "This is a test sentence. Let's see if tokenization works!"
    words = word_tokenize(test_text)
    sentences = sent_tokenize(test_text)
    
    print(f"✓ Word tokenization working: {len(words)} words detected")
    print(f"✓ Sentence tokenization working: {len(sentences)} sentences detected")
    print("\n✅ All NLTK dependencies are properly installed!")
    
except Exception as e:
    print(f"\n❌ Error during verification: {e}")
    print("You may need to manually download NLTK data.")
    print("Run: python -m nltk.downloader all")