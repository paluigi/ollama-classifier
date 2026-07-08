"""Unit tests for OllamaBackend helpers (no server required).

These cover the structural-token extraction logic used by the empirical
forced-generation ``tokenize()``/``score()`` path. The synthetic logprobs
mirror the tokenization Ollama actually emits for a ``{"label": "..."}``
JSON-enum constrained response.
"""

from ollama_classifier.backends.base import TokenLogprob
from ollama_classifier.backends.ollama import OllamaBackend


class TestLabelTokenLogprobs:
    """``OllamaBackend._label_token_logprobs`` extracts the value tokens."""

    def test_single_token_label(self):
        """``{"label": "sports"}`` -> exactly the ``sports`` token."""
        # Mirrors a real qwen2.5 emit: '{', ' "', 'label', '":', ' "',
        # 'sports', '"', ' }'
        logprobs = [
            TokenLogprob(token="{", logprob=-17.726),
            TokenLogprob(token=' "', logprob=-13.196),
            TokenLogprob(token="label", logprob=-0.000),
            TokenLogprob(token='":', logprob=-0.000),
            TokenLogprob(token=' "', logprob=-0.000),
            TokenLogprob(token="sports", logprob=-1.288),
            TokenLogprob(token='"', logprob=-0.001),
            TokenLogprob(token=" }", logprob=-0.000),
        ]
        out = OllamaBackend._label_token_logprobs(logprobs, "sports")
        assert [lp.token for lp in out] == ["sports"]
        assert out[0].logprob == -1.288

    def test_multi_token_label(self):
        """A label that spans two value tokens is returned in full, in order."""
        # '{"label": "' + 'tech' + ' support' + '" }'
        logprobs = [
            TokenLogprob(token='{"label": "', logprob=-10.0),
            TokenLogprob(token="tech", logprob=-0.5),
            TokenLogprob(token=" support", logprob=-0.7),
            TokenLogprob(token='" }', logprob=-0.0),
        ]
        out = OllamaBackend._label_token_logprobs(logprobs, "tech support")
        assert [lp.token for lp in out] == ["tech", " support"]

    def test_minimal_json_no_spaces(self):
        """Compact JSON ``{"label":"sports"}`` is handled too."""
        logprobs = [
            TokenLogprob(token='{"label":"', logprob=-10.0),
            TokenLogprob(token="sports", logprob=-1.288),
            TokenLogprob(token='"}', logprob=-0.0),
        ]
        out = OllamaBackend._label_token_logprobs(logprobs, "sports")
        assert [lp.token for lp in out] == ["sports"]

    def test_fallback_skeleton_filter(self):
        """When the value span cannot be located, skeleton filtering applies."""
        # Label text never appears verbatim -> primary mapping raises ValueError
        # and the fallback drops structure/key tokens, keeping the rest.
        logprobs = [
            TokenLogprob(token="{", logprob=-10.0),
            TokenLogprob(token=' "', logprob=-10.0),
            TokenLogprob(token="label", logprob=-0.0),
            TokenLogprob(token='":', logprob=-0.0),
            TokenLogprob(token=' "', logprob=-0.0),
            TokenLogprob(token="sports", logprob=-1.288),
            TokenLogprob(token='"', logprob=-0.0),
            TokenLogprob(token="}", logprob=-0.0),
        ]
        out = OllamaBackend._label_token_logprobs(logprobs, "missing")
        # Only the non-structure token survives
        assert [lp.token for lp in out] == ["sports"]

    def test_empty_logprobs(self):
        assert OllamaBackend._label_token_logprobs([], "sports") == []
