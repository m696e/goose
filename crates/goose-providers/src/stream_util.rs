//! Helpers shared by the streaming provider implementations.

use std::pin::Pin;
use std::time::Duration;

use anyhow::Error as AnyhowError;
use async_stream::try_stream;
use futures::Stream;
use tokio_stream::StreamExt;

/// Inter-chunk idle window: how long a stream may go without a line before the
/// connection is considered dead.
pub const DEFAULT_CHUNK_TIMEOUT_SECS: u64 = 15;

/// Budget for the **first** line of a stream: request queueing plus prompt
/// processing (prefill) on the provider side.
///
/// Deliberately much larger than [`DEFAULT_CHUNK_TIMEOUT_SECS`]. An idle gap
/// mid-response measures the network, but time-to-first-line measures the
/// provider's prefill, which grows with context size: a few-hundred-thousand
/// token prompt can take a minute or more before the model emits anything, and
/// the request is perfectly healthy while that happens. It is still bounded, so
/// a wedged connection fails in two minutes rather than hanging until the
/// socket-level read timeout.
pub const DEFAULT_FIRST_LINE_TIMEOUT_SECS: u64 = 120;

/// Which window elapsed when [`with_line_timeout`] gave up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutPhase {
    /// The provider had not sent a single line yet.
    FirstLine,
    /// The stream went idle after it had already produced output.
    Idle,
}

/// Inter-chunk idle window; `GOOSE_INFERENCE_CHUNK_TIMEOUT_SECS` overrides it.
pub fn chunk_timeout() -> Duration {
    Duration::from_secs(env_secs(
        "GOOSE_INFERENCE_CHUNK_TIMEOUT_SECS",
        DEFAULT_CHUNK_TIMEOUT_SECS,
    ))
}

/// Time-to-first-line budget; `GOOSE_INFERENCE_FIRST_LINE_TIMEOUT_SECS` overrides
/// it.
pub fn first_line_timeout() -> Duration {
    Duration::from_secs(env_secs(
        "GOOSE_INFERENCE_FIRST_LINE_TIMEOUT_SECS",
        DEFAULT_FIRST_LINE_TIMEOUT_SECS,
    ))
}

fn env_secs(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

/// The two idle budgets applied to a provider's raw SSE line stream.
///
/// Providers expose these through the declarative `stream_chunk_timeout_secs` /
/// `stream_first_line_timeout_secs` config keys; [`Default`] reads the
/// `GOOSE_INFERENCE_*` environment overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamTimeouts {
    /// Maximum gap between lines once the stream has started.
    pub chunk: Duration,
    /// Budget for the first line. `None` exempts it, leaving time-to-first-line
    /// to the socket-level timeout (used by providers that load a model on
    /// demand, where the wait is expected and unbounded).
    pub first_line: Option<Duration>,
}

impl Default for StreamTimeouts {
    fn default() -> Self {
        Self {
            chunk: chunk_timeout(),
            first_line: Some(first_line_timeout()),
        }
    }
}

/// Wraps a raw Server-Sent-Events **line** stream with an idle ("no data
/// received") timeout.
///
/// The timeout must be measured against line arrival, not against the stream of
/// assembled provider messages. The OpenAI/Anthropic decoders buffer tool-call
/// arguments and emit nothing until the tool call is complete, so applying the
/// timeout downstream of the decoder turns a slow-but-healthy tool call into a
/// bogus `NetworkError` (the connection is fine — the model is just still
/// generating). Timing the raw lines restores the intended meaning: "the
/// provider sent no data for N seconds". It also means keepalive frames
/// (`event: ping`, SSE comments, blank lines) that the decoders discard still
/// count as liveness.
///
/// `first_line_budget` separates the two budgets: `Some(_)` bounds
/// time-to-first-line (queueing + prefill) on its own, more generous window,
/// while `None` defers it entirely to the request-level timeout.
///
/// `on_timeout` builds the error yielded when a window elapses; callers use it to
/// produce a `ProviderError` variant that downstream code can downcast (rather
/// than a bare `anyhow` error, which would surface as a generic stream-decode
/// failure), and to describe which window expired.
pub fn with_line_timeout<F>(
    stream: impl Stream<Item = anyhow::Result<String>> + Unpin + Send + 'static,
    idle: Duration,
    first_line_budget: Option<Duration>,
    on_timeout: F,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>
where
    F: Fn(TimeoutPhase) -> AnyhowError + Send + 'static,
{
    Box::pin(try_stream! {
        let mut stream = stream;

        match first_line_budget {
            None => match stream.next().await {
                Some(first_line) => yield first_line?,
                None => return,
            },
            Some(budget) => match tokio::time::timeout(budget, stream.next()).await {
                Ok(Some(first_line)) => yield first_line?,
                Ok(None) => return,
                Err(_) => Err(on_timeout(TimeoutPhase::FirstLine))?,
            },
        }

        loop {
            match tokio::time::timeout(idle, stream.next()).await {
                Ok(Some(line)) => yield line?,
                Ok(None) => break,
                Err(_) => Err(on_timeout(TimeoutPhase::Idle))?,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ProviderError::{self, NetworkError};
    use futures::stream;

    fn line(s: &str) -> anyhow::Result<String> {
        Ok(s.to_string())
    }

    fn network_error(_phase: TimeoutPhase) -> anyhow::Error {
        NetworkError("Stream timed out waiting for next chunk".to_string()).into()
    }

    /// `stream::unfold` is not `Unpin`; the real callers (a `FramedRead`) are.
    fn pinned<S: Stream + Send + 'static>(s: S) -> Pin<Box<S>> {
        Box::pin(s)
    }

    /// Yields `count` lines `gap` apart. A `gap` of one hour simulates silence.
    fn spaced(count: u32, gap: Duration) -> impl Stream<Item = anyhow::Result<String>> {
        pinned(stream::unfold(0u32, move |i| async move {
            if i >= count {
                return None;
            }
            tokio::time::sleep(gap).await;
            Some((line("data: chunk"), i + 1))
        }))
    }

    #[tokio::test]
    async fn yields_lines_that_arrive_within_the_window() {
        let src = stream::iter(vec![line("data: a"), line("data: b"), line("[DONE]")]);
        let out: Vec<String> = with_line_timeout(src, Duration::from_secs(60), None, network_error)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(out, vec!["data: a", "data: b", "[DONE]"]);
    }

    #[tokio::test]
    async fn steady_lines_do_not_trip_the_idle_timeout() {
        // Lines arrive every 20ms for a total well beyond the 50ms window: this is
        // the slow-tool-call case that used to be misreported as a network failure
        // because nothing was yielded until the whole call was assembled.
        let src = spaced(10, Duration::from_millis(20));
        let out: Vec<_> = with_line_timeout(src, Duration::from_millis(50), None, network_error)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(out.len(), 10);
        assert!(out.into_iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn silence_longer_than_the_window_yields_the_timeout_error() {
        let src = stream::unfold(0u32, |i| async move {
            match i {
                0 => Some((line("data: first"), 1)),
                // then go silent forever
                _ => {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    Some((line("never"), i + 1))
                }
            }
        });
        let mut s = with_line_timeout(pinned(src), Duration::from_millis(50), None, network_error);
        assert_eq!(s.next().await.unwrap().unwrap(), "data: first");
        let err = s.next().await.unwrap().unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ProviderError>(),
            Some(NetworkError(_))
        ));
    }

    #[tokio::test]
    async fn exempt_first_line_defers_the_timeout() {
        // First line after 60ms (> 50ms window) still succeeds when exempt.
        let src = spaced(1, Duration::from_millis(60));
        let out: Vec<_> = with_line_timeout(src, Duration::from_millis(50), None, network_error)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(out.len(), 1);
        assert!(out[0].is_ok());
    }

    #[tokio::test]
    async fn slow_first_line_is_allowed_by_its_own_budget() {
        // A 60ms first line blows the 20ms inter-chunk window but is within the
        // 5s first-line budget — the slow-prefill case.
        let src = spaced(1, Duration::from_millis(60));
        let out: Vec<_> = with_line_timeout(
            src,
            Duration::from_millis(20),
            Some(Duration::from_secs(5)),
            network_error,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(out.len(), 1, "first line should not be reported as stalled");
        assert!(out[0].is_ok());
    }

    #[tokio::test]
    async fn first_line_past_its_budget_reports_the_first_line_phase() {
        // One hour of silence: the first-line budget expires before any line, so
        // the reported phase must be FirstLine (not Idle), letting callers name
        // slow prefill instead of blaming the network.
        let src = spaced(1, Duration::from_secs(3600));
        let phases = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = phases.clone();
        let mut s = with_line_timeout(
            src,
            Duration::from_secs(3600),
            Some(Duration::from_millis(50)),
            move |phase| {
                seen.lock().unwrap().push(phase);
                network_error(phase)
            },
        );

        let err = s.next().await.unwrap().unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ProviderError>(),
            Some(NetworkError(_))
        ));
        assert_eq!(*phases.lock().unwrap(), vec![TimeoutPhase::FirstLine]);
    }

    #[tokio::test]
    async fn idle_gap_after_output_reports_the_idle_phase() {
        let phases = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = phases.clone();
        let src = stream::unfold(0u32, |i| async move {
            match i {
                0 => Some((line("data: first"), 1)),
                _ => {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    Some((line("never"), i + 1))
                }
            }
        });
        let mut s = with_line_timeout(
            pinned(src),
            Duration::from_millis(50),
            Some(Duration::from_millis(50)),
            move |phase| {
                seen.lock().unwrap().push(phase);
                network_error(phase)
            },
        );

        assert_eq!(s.next().await.unwrap().unwrap(), "data: first");
        s.next().await.unwrap().unwrap_err();
        assert_eq!(*phases.lock().unwrap(), vec![TimeoutPhase::Idle]);
    }

    #[test]
    fn default_timeouts_bound_the_first_line() {
        // Assert against the accessors rather than literal seconds so that a
        // GOOSE_INFERENCE_*_TIMEOUT_SECS override in the environment cannot make
        // this fail spuriously.
        let timeouts = StreamTimeouts::default();
        assert_eq!(timeouts.chunk, chunk_timeout());
        assert_eq!(timeouts.first_line, Some(first_line_timeout()));

        assert_eq!(DEFAULT_CHUNK_TIMEOUT_SECS, 15);
        assert_eq!(DEFAULT_FIRST_LINE_TIMEOUT_SECS, 120);
        // The first line gets its own, more generous budget: it covers prefill,
        // not a network gap.
        assert!(first_line_timeout() > chunk_timeout());
    }
}
