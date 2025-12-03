
import json
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from data_utils import load_corpus

def test_load_corpus():
    # Create a dummy corpus file
    dummy_corpus = [
        {"text": "This is the first document. It has some text."},
        {"text": "This is the second document. It also has text."},
        {"other_field": "This should be ignored."}
    ]
    
    corpus_file = "dummy_corpus.json"
    with open(corpus_file, "w") as f:
        json.dump(dummy_corpus, f)
        
    try:
        print("Testing load_corpus with dummy file...")
        corpus = load_corpus(corpus_file)
        print(f"Loaded corpus with {len(corpus)} items.")
        print(f"First item: {corpus[0]}")
        
        assert len(corpus) > 0
        assert 'contents' in corpus[0]
        assert corpus[0]['contents'] == "This is the first document. It has some text. This is the second document. It also has text."
        # Wait, the splitter joins them with space.
        
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(corpus_file):
            os.remove(corpus_file)

if __name__ == "__main__":
    test_load_corpus()
