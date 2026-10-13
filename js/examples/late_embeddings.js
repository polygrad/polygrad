'use strict'

// Public Model composition; the caller owns tokenization, prompts and checkpoint
// skip-token IDs. Load projection tensors with logical: 'always' on this runtime.
async function composeTokenEmbeddings(pg, backbone, projection, inputs, skipTokenIds = []) {
  const skip = [...new Set(skipTokenIds)]
  const exclude = pg.Tensor.empty([1], {dtype: 'int32'})
  try { return await pg.Model.fromCallableAsync(({input_ids, attention_mask, position_ids, exclude_tokens}) => {
    const hidden = backbone.forward({input_ids, attention_mask, position_ids}).last_hidden_state
    const token_embeddings = hidden.linear(projection.T).normalize({dim: -1, eps: 1e-12})
    let token_mask = attention_mask.ne(0)
    for (const token of skip)
      token_mask = token_mask.bitwiseAnd(input_ids.ne(token).bitwiseOr(exclude_tokens.eq(0)))
    return {token_embeddings, token_mask}
  }, {inputs: {...inputs, exclude_tokens: exclude}, params: {projection}})
  } finally { exclude.dispose() }
}

module.exports = {composeTokenEmbeddings}
