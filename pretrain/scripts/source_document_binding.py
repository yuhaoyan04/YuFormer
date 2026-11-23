"""Contract for source-row verification in newly materialized training streams."""

SOURCE_DOCUMENT_BINDING_VERSION = 4


def verified_source_document_binding(meta: dict) -> bool:
    binding = meta.get("source_document_binding") if isinstance(meta, dict) else None
    if not isinstance(binding, dict):
        return False
    fingerprint = binding.get("roster_fingerprint")
    return (type(binding.get("schema_version")) is int
            and binding["schema_version"] == SOURCE_DOCUMENT_BINDING_VERSION
            and binding.get("expected_doc_ids_verified") is True
            and binding.get("tokenizer_bytes_verified") is True
            and binding.get("order_bytes_verified") is True
            and binding.get("source_handles_verified") is True
            and isinstance(fingerprint, str) and len(fingerprint) == 64
            and all(char in "0123456789abcdef" for char in fingerprint))
