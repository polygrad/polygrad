"""Compose a token-embedding backbone and projection using the public Model API.

The caller loads/tokenizes with the checkpoint's processor and supplies its
skip-token IDs. Prompts and text processing are not model graph operations.
Both query and document calls share weights; exclude_tokens selects filtering.
Keep logical graphs when loading projection tensors (create(logical='always')).
"""


def compose_token_embeddings(runtime, backbone, projection, inputs, skip_token_ids=()):
    Tensor = runtime.Tensor
    skip = tuple(dict.fromkeys(int(token) for token in skip_token_ids))

    def forward(input_ids, attention_mask, position_ids, exclude_tokens):
        hidden = backbone.forward(input_ids=input_ids, attention_mask=attention_mask,
                                  position_ids=position_ids)['last_hidden_state']
        embeddings = hidden.linear(projection.T).normalize(dim=-1, eps=1e-12)
        keep = attention_mask != 0
        for token in skip:
            keep = keep & ((input_ids != token) | (exclude_tokens == 0))
        # A mask avoids data-dependent output shapes and GPU-to-host compaction.
        return {'token_embeddings': embeddings, 'token_mask': keep}

    exclude = Tensor.empty(1, dtype='int32')
    try:
        return runtime.Model(forward, inputs={**inputs, 'exclude_tokens': exclude},
                             params={'projection': projection})
    finally:
        exclude.dispose()
