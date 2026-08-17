# Run Configuration

::: center
  **Field**         **Value**
  ----------------- -------------------------------------------------------------------------
  Question          *"Why is community detection hard and how can quantum computing help?"*
  Retrieval arm     [`typed_graph`]{style="color: accent"}
  Invocation        `scripts/ask.py --system typed_graph`
  Embedding model   `sentence-transformers/allenai-specter`
  Graph index       25,410 node names embedded
  Compute device    `mps` (Apple Silicon)
  Synthesis model   `gpt-5.4-mini`
  Papers grounded   4
  Wall clock        $\approx$ 64 s
:::

[Hub requests were unauthenticated (no `HF_TOKEN` set); this affects
download rate limits only, not retrieval or results. Model weights
resolved from cache --- 199 shards, sub-second
load.]{style="color: rulegray"}

# Timeline

:::: center
::: tabular
\@l l r l@ **From** & **To** & **Elapsed** & **Stage**\
:30:09 & 19:30:12 & $\approx$ 3 s & Device selection, encoder load
(cached)\
19:30:12 & 19:31:09 & $\approx$ 57 s & Typed-graph node-name embedding
(25,410 names)\
19:31:09 & 19:31:13 & $\approx$ 4 s & Synthesis call\
& **$\approx$ 64 s** &\
:::
::::

Node-name embedding accounts for roughly **89% of wall clock**. It is a
whole-graph cost that does not vary with the query, so it belongs in a
persisted index rather than in the request path --- the single largest
available latency win for this arm.

# Synthesized Answer

## Why community detection is hard {#why-community-detection-is-hard .unnumbered}

In multipartite networks the problem is "not trivial" and "usually
requires ad hoc methodologies".[^1] Modularity maximization has "been
largely studied" with many competing methods proposed --- itself
evidence that this is a difficult optimization problem rather than a
straightforward task.[^2] Performance is characteristically evaluated on
benchmark graphs as mixing increases, indicating sensitivity to graph
structure.[^3]

## How quantum computing can help {#how-quantum-computing-can-help .unnumbered}

- Community detection can be recast as an optimization problem that a
  **quantum annealer** can solve.[^4]

- One paper presents a **modularity-based formulation for multiple
  communities** and runs it on the **D-Wave** annealer. [^5]

- The reported result is a "highly optimized community structure", with
  the annealer achieving the best value based on community structure
  among the methods compared.[^6]

- More generally, quantum computers are described as a means to solve
  problems intractable in reasonable time classically, with quantum
  algorithms potentially outperforming classical ones on specific
  problems.[^7]

## Stated limitation {#stated-limitation .unnumbered}

The system flagged its own evidence gap: the retrieved excerpts *do not*
supply a single unified explanation of *why* community detection is hard
in general. They establish three narrower points only --- nontrivial
structure, ad hoc handling in multipartite settings, and modularity
maximization as a challenging optimization task.[^8]

# Grounding

Four distinct papers were cited. Full Semantic Scholar identifiers:

::: center
  **Short**    **Full identifier**                          **Role in answer**
  ------------ -------------------------------------------- ----------------------------
  `15eded04`   `15eded04386a8982ccd5627bd1efe70bbf624c02`   Quantum advantage, general
  `19e9eaa0`   `19e9eaa00a36ed3334ea0c18b337a18d18f95c3e`   Modularity QUBO, D-Wave
  `9be428c9`   `9be428c9383d47b86570b1b9fc20faf006346c5d`   Multipartite hardness
  `f9db7ae0`   `f9db7ae0a333ef8a21317d1a3126d75da9d43ff4`   Quantum advantage, general
:::

Every claim carries a paper-level citation, and the two halves of the
question draw on different sources --- hardness from `9be428c9`, the
quantum result from `19e9eaa0` --- rather than resolving onto one
dominant paper.

# Cost

::: center
  **Model**          **Calls**      **In**   **Out**   **Cached**     **Cost**
  ---------------- ----------- ----------- --------- ------------ ------------
  `gpt-5.4-mini`             1       1,194       301            0       \$0.00
  **Total**              **1**   **1,194**   **301**        **0**   **\$0.00**
:::

# Cross-Arm Context

[Not from this run --- carried over from the `fulltext` vector arm on
the same question, for comparison.]{style="color: rulegray"}

::: center
  **Measure**         [`fulltext`]{style="color: accent"}   [`typed_graph`]{style="color: accent"}
  ----------------- ------------------------------------- ----------------------------------------
  Input tokens                                     11,316                                    1,194
  Output tokens                                       421                                      301
  Papers grounded                                       1                                        4
  Wall clock                                $\approx$ 9 s                           $\approx$ 64 s
  Cost                                             \$0.01                                   \$0.00
:::

The typed graph reached **four papers on one-tenth the input tokens**,
and spread its citations across the two halves of the question --- where
the vector arm retrieved 20 chunks from 13 papers and grounded on one.
That is the behaviour the typed graph is built to produce.

Two caveats keep this from being a result. First, one question is an
anecdote, not evidence; it needs the full 34-question set with
per-query-type breakdown before it means anything. Second, the
typed-graph answer explicitly hedged that its excerpts gave no unified
account of the hardness --- so the thinner evidence budget may be buying
precision at the cost of coverage. Both arms should be scored against
the gold set on citation recall before either is preferred.

[^1]: `9be428c9`

[^2]: `19e9eaa0`

[^3]: `9be428c9`

[^4]: `19e9eaa0`

[^5]: `19e9eaa0`

[^6]: `19e9eaa0`

[^7]: `f9db7ae0`, `15eded04`

[^8]: `9be428c9`, `19e9eaa0`
