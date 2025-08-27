import pytest

from src.ngram_engine.ngram import NgramEngine


def test_generate_sentence():
    # This is a basic test to check if sentence generation runs without errors.
    # It's not a test for the quality of the generated text.
    try:
        engine = NgramEngine(n=3, k=500, advanced=True)
        start_context = ["<s>", "the"]
        sentence = engine.generate_sentence(start_context)
        print("Generated sentence:", " ".join(sentence))
        assert isinstance(sentence, list)
        assert len(sentence) > 0
        assert "<s>" in sentence
    except FileNotFoundError:
        pytest.skip(
            "Skipping test, required n-gram/vocab files not found. "
            "Run the n-gram generation script first."
        )
    except Exception as e:
        pytest.fail(f"Sentence generation failed with an exception: {e}")


def test_get_word_from_context():
    # This test checks if the word generation function returns a string.
    try:
        engine = NgramEngine(n=3, k=500, advanced=True)
        context = ("the", "king")
        word = engine.get_word_from_context(context, n=3, k=500, suffix="_adv")
        print(f"Generated word for context {context}: {word}")

        # The word can be None if no word is found, which is a valid outcome.
        assert isinstance(word, str) or word is None
    except FileNotFoundError:
        pytest.skip(
            "Skipping test, required n-gram/vocab files not found. "
            "Run the n-gram generation script first."
        )
        print("Skipping test, required n-gram/vocab files not found.")
    except Exception as e:
        pytest.fail(f"get_word_from_context failed with an exception: {e}")
        print(e)
