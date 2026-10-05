# CUSTOM.md — what this fork adds over upstream goose

`custom` is the fork's own branch. Everything below is the difference between it
and `https://github.com/aaif-goose/goose.git` at `main`, taken from the commit
history, not from a changelog.

## TL;DR

The fork keeps upstream's agent and adds four things it does not have:

1. **A retrievable history.** Compaction no longer discards. Compacted and
   cleared messages stay on record and can be read back (`/archive`), the model
   can ask for a compaction, and each compaction is recorded with its token
   numbers.
2. **A request that survives real content.** Image size limits come from the
   provider instead of a guess, the agent measures a request before sending it
   and evicts the largest block when it is still too big, and a rejection for
   size is reported as "too large" rather than "context length".
3. **Streaming that fails predictably.** Time-to-first-line and inter-chunk idle
   are budgeted separately, a dropped stream is retried with a notice, 4xx
   retries that the policy can prove are permanent stop, and the
   10-minute-hang class of bug is fixed.
4. **A decision-model tool.** `jev` (ask / steer) exposes a separate model that
   returns a typed answer — a probability, an option, or a rubric level — for
   questions the caller does not want to decide itself.

Plus a long tail of session-store, CLI-UX, and portability fixes listed below.

| | |
|---|---|
| fork | `https://github.com/m696e/goose.git`, branch `custom` |
| upstream | `https://github.com/aaif-goose/goose.git` |
| version | `1.52.0` |
| as of | 2026-10-02, 49 commits ahead of upstream `main` |

Two of those commits (`5fbba1ef6`, `09d663d84`) are also open upstream pull
requests, so they will disappear from this list once merged.

---

## 1. Context management and compaction

Upstream compaction summarises and drops. This fork keeps the dropped material.

- **Compacted and cleared history is kept on record** (`86af2d471`). Rather than
  deleting, the messages are marked agent-invisible and retained, so a person
  can go back and read what was removed. This is the change every other one in
  the section depends on.
- **The model can ask for a compaction** (`0af87599f`), and the request may
  carry **a note** explaining why (`046ee11ad`).
- **A compaction that would repeat the previous one is refused**
  (`5cd4d56d1`), which stops the loop where a compaction produces a context
  that immediately compacts again.
- **Notices are recorded, token numbers are labelled, and evictions are
  logged** (`f8ce9be0c`) — the compaction record is auditable rather than a
  single "context was rewritten" line.
- **Compaction never keeps images** (`6e5a8b77d`): a compaction strips image
  payloads from the context it builds.
- **Compaction stages are shown while the context is rewritten** (`ed8fc38ce`),
  so a long compaction does not look like a hang.
- **The context size survives a request that reports no usage** (`09d663d84`,
  also an open upstream PR) — the size was being reset to zero whenever the
  provider returned usage-less chunks.
- **Messages the agent can no longer see have their images and reasoning
  elided** (`d1b033ae5`). This is storage hygiene, not a behaviour change: the
  agent was never going to see them again, and `/archive` renders text only, so
  the elision is invisible to a reader while removing the byte cost. Long
  reasoning blocks are replaced with a marker; images become a one-line note
  recording their MIME type and size.

## 2. Images and request size

The upstream failure mode is a request that the provider rejects with an opaque
error. The fork measures and classifies before sending.

- **The image dimension limit comes from the provider, not a guess**
  (`12fd1314a`). Each provider declares the limit it actually enforces.
- **An image the provider refuses for its shape is reported as such**
  (`5b6f94b04`), instead of surfacing as a generic request failure.
- **`read_image` stays inside the provider's request budget** (`cde32b893`),
  downscaling rather than producing a request that cannot be sent.
- **The agent tells the model what to shrink when a request is too large**
  (`69637cb1c`), and **evicts the largest content block when the request is
  still too large** (`d8d225f26`). The eviction is largest-first, so the
  smallest amount of context is lost.
- **A size refusal is classified as "too large", not "context length
  exceeded"** (`9f2c72769`) — the two were being conflated, so the wrong
  recovery was attempted.
- **A turn's tool results stay contiguous when one of them carries an image**
  (`d4387758b`). Some providers reject a turn whose tool results are not
  adjacent; an image block used to split them.
- **Declarative providers can declare model vision support** (`96f3acdd0`).

## 3. Streaming, timeouts, and retries

- **An inter-chunk timeout on SSE streaming** (`7f775aad2`) fixes the class of
  bug where a stalled stream hangs for ten minutes. The timeout is measured
  **on raw lines, not assembled messages** (`56cde6537`).
- **Time-to-first-line is budgeted separately from the inter-chunk idle
  window** (`644f6ea8f`). A slow first token is not the same failure as a
  stalled stream, and the two need different limits.
- **A transient provider error auto-retries the stream** (`bb0419499`), and the
  **user is shown a retry notice when the stream recovers** (`311ed8b09`).
- **A retryable provider error resends the turn** (`e60ef08c1`).
- **4xx responses that the policy can prove are permanent are not retried**
  (`57dae1803`) — upstream retried them, wasting the turn.
- **OpenAI Responses stream events tolerate missing fields** (`68a2df564`) and
  **events without `created_at` or `sequence_number`** (`4add07bcc`).

## 4. The `jev` decision tool and permissions

- **`jev` — ask a decision model for a typed answer** (`695045cf3`). The model
  returns a number, not prose: a yes/no probability, one of a set of options, or
  a level from an ordered rubric.
- **A `steer` tool** (`4b4ba9f7c`) that orders candidate research directions and
  returns a direction to consider or an explicit no-steer — deliberately not a
  number, so it cannot be mistaken for evidence.
- **A shadow-mode read-only classifier backed by a decision provider**
  (`2065dad7c`), enabled by **`GOOSE_JEV_SHADOW=1`** (`346d24928`).
- **The user can grant a capability to a session** (`d904271df`), and **the
  model gets a read-only view of its own session** (`65a179888`).

## 5. Session store

- **Messages are indexed by session and message id** (`5fbba1ef6`, also an open
  upstream PR) — the missing index made message lookups and deletions slow.
- **Deleting a session removes its compaction events** (`6f43d7547`); they were
  being orphaned.
- **The message index is re-ensured for databases stamped past schema v17**
  (`965fcd864`), repairing databases that had the version but not the index.
- **Token counts are 64-bit** (`fb0a1ab07`). The `accumulated_*` columns are
  lifetime sums held in SQLite `INTEGER` (64-bit), but the Rust type was
  `Option<i32>`, so a long session that passed `i32::MAX` could not be decoded
  at all and the whole session became unloadable. The fix widens token counts to
  `TokenCount = i64` across the usage types, the SDK wire mirror, the CLI
  metadata, and the compaction/eviction paths, and removes two latent
  `i32::MAX` clamps.
- **The session name is reported in chat-history search** (`7a9b74698`).

## 6. CLI UX

- **A thinking notice while the model reasons** (`ff3886498`), **a prefilling
  notice while waiting for the first token** (`48ddf838d`), and **a tool-call
  reception notice while its arguments stream** (`eb82ad3ec`). A slow model
  looks slow rather than dead.
- **The command the user just typed is no longer echoed back** (`e6ef5ccba`).
- **A fuzzy session picker and keyword session search** (`61da2f49a`).
- **Session cost is estimated with the session's own provider**
  (`b2828ed27`) — the pricing came from the wrong provider.

## 7. Providers and build

- **DeepSeek reasoning model support** (`8f9202f60`).
- **rustls and `aws-lc-rs` are kept out of `native-tls` builds**
  (`a4eced719`). A `native-tls` build then links OpenSSL and contains no
  `aws_lc` or `hyper-rustls` symbols, which is checkable with `grep -ac` and
  `ldd` on the binary.
- **The build works on Android/Termux (aarch64-linux-android)** (`fff0a330d`).
- **Release build performance and iteration recipes are documented**
  (`66fe116bb`).

---

## Seeing the diff yourself

```bash
git clone https://github.com/m696e/goose.git
cd goose
git remote add upstream https://github.com/aaif-goose/goose.git
git fetch upstream
git log --oneline upstream/main..origin/custom
```

## What is *not* different

- The agent loop, provider implementations, and extension protocol are
  upstream's. Nothing here changes the wire format seen by an extension.
- The workspace and its crates are upstream's. A build may choose a narrower
  feature set (for instance `--no-default-features --features code-mode,native-tls`),
  but that is a build decision, not a fork change.
