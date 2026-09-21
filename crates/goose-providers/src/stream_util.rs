//! Helpers shared by the streaming provider implementations.

use std::pin::Pin;
use std::time::Duration;

use anyhow::Error as AnyhowError;
use async_stream::try_stream;
use futures::Stream;
use tokio_stream::StreamExt;

/// Wraps a raw Server-Sent-Events **line** stream with an idle ("no data
/// received") timeout.
///
/// The timeout must be measured against line arrival, not against the stream of
/// assembled provider messages. The OpenAI/Anthropic decoders buffer tool-call
/// arguments and emit nothing until the tool call is complete, so applying the
/// timeout downstream of the decoder turns a slow-but-healthy tool call into a
/// bogus `NetworkError` (the connection is fine — the model is just still
/// generating). Timing the raw lines restores the intended meaning: "the
/// provider sent no data for N seconds".
///
/// `exempt_first_line` leaves time-to-first-token to the request-level timeout
/// instead of the (much shorter) idle timeout. This is required for providers
/// (e.g. Ollama) that can spend minutes loading a model before the first token,
/// but not for hosted APIs where the token is the thing worth failing fast on.
///
/// `on_timeout` builds the error yielded when the idle window elapses; callers
/// use it to produce a `ProviderError` variant that downstream code can
/// downcast (rather than a bare `anyhow` error, which would surface as a
/// generic stream-decode failure).
pub fn with_line_timeout<F>(
    stream: impl Stream<Item = anyhow::Result<String>> + Unpin + Send + 'static,
    idle: Duration,
    exempt_first_line: bool,
    on_timeout: F,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>
where
    F: Fn() -> AnyhowError + Send + 'static,
{
    Box::pin(try_stream! {
        let mut stream = stream;

        if exempt_first_line {
            match stream.next().await {
                Some(first_line) => yield first_line?,
                None => return,
            }
        }

        loop {
            match tokio::time::timeout(idle, stream.next()).await {
                Ok(Some(line)) => yield line?,
                Ok(None) => break,
                Err(_) => Err(on_timeout())?,
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

    fn network_error() -> anyhow::Error {
        NetworkError("Stream timed out waiting for next chunk".to_string()).into()
    }

    /// `stream::unfold` is not `Unpin`; the real callers (a `FramedRead`) are.
    fn pinned<S: Stream + Send + 'static>(s: S) -> Pin<Box<S>> {
        Box::pin(s)
    }

    #[tokio::test]
    async fn yields_lines_that_arrive_within_the_window() {
        let src = stream::iter(vec![line("data: a"), line("data: b"), line("[DONE]")]);
        let out: Vec<String> = with_line_timeout(src, Duration::from_secs(60), true, network_error)
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
        let src = pinned(stream::unfold(0u32, |i| async move {
            if i >= 10 {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            Some((line("data: chunk"), i + 1))
        }));
        let out: Vec<_> = with_line_timeout(src, Duration::from_millis(50), true, network_error)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(out.len(), 10);
        assert!(out.into_iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn silence_longer_than_the_window_yields_the_timeout_error() {
        let src = pinned(stream::unfold(0u32, |i| async move {
            match i {
                0 => Some((line("data: first"), 1)),
                // then go silent forever
                _ => {
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    Some((line("never"), i + 1))
                }
            }
        }));
        let mut s = with_line_timeout(src, Duration::from_millis(50), true, network_error);
        assert_eq!(s.next().await.unwrap().unwrap(), "data: first");
        let err = s.next().await.unwrap().unwrap_err();
        assert!(matches!(
            err.downcast_ref::<ProviderError>(),
            Some(NetworkError(_))
        ));
    }

    #[tokio::test]
    async fn first_line_exemption_defers_the_timeout() {
        // First line after 60ms (> 50ms window) still succeeds when exempt.
        let src = pinned(stream::unfold(0u32, |i| async move {
            match i {
                0 => {
                    tokio::time::sleep(Duration::from_millis(60)).await;
                    Some((line("data: slow-first"), 1))
                }
                _ => None,
            }
        }));
        let out: Vec<_> = with_line_timeout(src, Duration::from_millis(50), true, network_error)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(out.len(), 1);
        assert!(out[0].is_ok());
    }
}
