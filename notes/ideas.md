# Ideas (not to be pursued without a human OK)
- Output `top_logprobs` are available: the answer's first-token distribution could replace or cross-check the 16-sample estimates in Phase 3.
- Check whether the filler-induced flips are the same items across filler types (counting 25 vs dots 100) and across re-runs.
- Classify what the wrong no-filler answers are (which chain step fails) for items that flip vs items that don't; the chain is already stored per item.
- To make the error-mix test discriminating, build a task variant where binding errors are common (more distractors, similar names) so binding and arithmetic each have enough mass.
- Repeat the disruption test on a 25-unit filler (block of ~6), where there is no spare room for a chain to run around the block.
