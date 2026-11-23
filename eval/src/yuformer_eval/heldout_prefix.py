"""Raw UTF-8 accounting and exact source-token checks for paired long BPB."""


def verify_encoded_document(actual_ids, audited_ids, *, eod_token_id):
    if len(audited_ids) != len(actual_ids)+1 or int(audited_ids[-1]) != eod_token_id:
        raise ValueError('raw heldout document length/EOD differs from its audited token ledger')
    if any(int(left) != int(right) for left, right in zip(actual_ids, audited_ids[:-1], strict=True)):
        raise ValueError('raw heldout document does not reencode to the actual admitted validation bytes')


def bounded_utf8_prefix(text, tokenizer, *, target_tokens, eod_token_id):
    if not isinstance(text, str) or not text or type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError('long BPB needs actual nonempty source text and an explicit token budget')
    # Select on character boundaries, then reencode that exact prefix. Slicing
    # token IDs and guessing a byte denominator loses BPE boundary/UTF-8 context.
    left, right = 1, len(text)
    best = None
    while left <= right:
        middle = (left+right)//2
        prefix = text[:middle]
        ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        if len(ids) <= target_tokens:
            best = (prefix, ids)
            left = middle+1
        else:
            right = middle-1
    if best is None or not target_tokens*.75 <= len(best[1]) <= target_tokens:
        raise ValueError('actual raw prefix cannot meet the unchanged75pct long-context floor')
    prefix, ids = best
    encoded = prefix.encode('utf-8')
    if not encoded:
        raise ValueError('BPB denominator must contain actual original UTF-8 bytes')
    return {'text': prefix, 'token_ids': [eod_token_id, *ids],
            'target_tokens': len(ids), 'uncompressed_utf8_bytes': len(encoded)}
