from types import SimpleNamespace

import pytest

from yuformer_eval.heldout_prefix import bounded_utf8_prefix, verify_encoded_document


class ByteTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(text.encode('utf-8')))


def test_multibyte_prefix_counts_real_raw_bytes_and_all_body_targets():
    result = bounded_utf8_prefix('中文字符测试'*10, ByteTokenizer(), target_tokens=32, eod_token_id=0)
    body = result['text'].encode('utf-8')
    assert result['token_ids'] == [0, *body]
    assert result['uncompressed_utf8_bytes'] == len(body)
    assert len(result['token_ids']) == result['target_tokens']+1
    assert 24 <= result['target_tokens'] <= 32
    assert result['text'] in '中文字符测试'*10


def test_source_mismatch_cannot_be_hidden_by_an_identical_document_length():
    with pytest.raises(ValueError, match='actual admitted validation bytes'):
        verify_encoded_document([4, 5, 6], [4, 7, 6, 0], eod_token_id=0)
    with pytest.raises(ValueError, match='length/EOD'):
        verify_encoded_document([4, 5, 6], [4, 5, 6, 9], eod_token_id=0)
    verify_encoded_document([4, 5, 6], [4, 5, 6, 0], eod_token_id=0)


def test_short_document_cannot_be_labeled_a_long_context_case():
    with pytest.raises(ValueError, match='unchanged75pct'):
        bounded_utf8_prefix('short', ByteTokenizer(), target_tokens=64, eod_token_id=0)
