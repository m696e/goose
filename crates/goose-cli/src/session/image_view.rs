//! Inline image display in the terminal, for the person watching rather than
//! for the model.
//!
//! A `read_image` result arrives as an image content block, and a terminal
//! cannot print one: `render_tool_response` has no text to write for it, so the
//! user saw the "Loaded image from ..." line and never the picture the model
//! was handed. When the terminal speaks a graphics protocol, chafa is asked to
//! emit that protocol and the bytes are written through untouched, so the image
//! appears inline where the tool result is printed.
//!
//! Nothing here reaches the model — the tool result is not modified, and this
//! runs entirely inside the CLI's output path.
//!
//! - `GOOSE_CLI_IMAGE_PROTOCOL`: `kitty` (default where it is supported),
//!   `sixel`, `iterm`, or `off`
//! - `GOOSE_CLI_IMAGE_SIZE`: `chafa -s` value, the largest cell footprint the
//!   image may occupy, default `80x24`
//! - `GOOSE_CLI_IMAGE_BIN`: renderer binary, default `chafa`

use base64::Engine;
use goose::config::Config;
use rmcp::model::ImageContent;
use std::io::{IsTerminal, Write};
use std::process::{Command, Stdio};

pub const DEFAULT_SIZE: &str = "80x24";
pub const DEFAULT_BINARY: &str = "chafa";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Kitty,
    Sixels,
    Iterm,
}

impl Protocol {
    fn chafa_format(self) -> &'static str {
        match self {
            Self::Kitty => "kitty",
            Self::Sixels => "sixels",
            Self::Iterm => "iterm",
        }
    }
}

/// The protocol named by configuration, if it names a usable one.
fn configured_protocol(value: &str) -> Option<Protocol> {
    match value.trim().to_ascii_lowercase().as_str() {
        "kitty" => Some(Protocol::Kitty),
        "sixel" | "sixels" => Some(Protocol::Sixels),
        "iterm" | "iterm2" => Some(Protocol::Iterm),
        _ => None,
    }
}

/// The protocol a terminal is known to speak from what it advertises.
///
/// Only terminals that actually implement the kitty protocol are guessed at,
/// because emitting graphics escapes at anything else prints them as text: an
/// unrecognised terminal gets nothing at all.
fn detected_protocol(
    kitty_window_id: Option<&str>,
    term_program: Option<&str>,
) -> Option<Protocol> {
    if kitty_window_id.is_some() {
        return Some(Protocol::Kitty);
    }
    match term_program.map(str::to_ascii_lowercase).as_deref() {
        Some("ghostty") | Some("wezterm") | Some("kitty") => Some(Protocol::Kitty),
        _ => None,
    }
}

/// Which protocol to emit, from configuration or from the terminal's own
/// advertisement. `GOOSE_CLI_IMAGE_PROTOCOL=off` is the documented way to stop
/// the rendering; an unrecognised value is treated the same way rather than
/// guessed at.
pub fn protocol() -> Option<Protocol> {
    if let Ok(value) = Config::global().get_param::<String>("GOOSE_CLI_IMAGE_PROTOCOL") {
        return configured_protocol(&value);
    }
    detected_protocol(
        std::env::var("KITTY_WINDOW_ID").ok().as_deref(),
        std::env::var("TERM_PROGRAM").ok().as_deref(),
    )
}

fn setting(key: &str, default: &str) -> String {
    Config::global()
        .get_param::<String>(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The escape sequence that draws `image`, or `None` when nothing can be drawn.
fn art_for(image: &ImageContent, protocol: Protocol, size: &str, binary: &str) -> Option<String> {
    let bytes = base64::prelude::BASE64_STANDARD.decode(&image.data).ok()?;
    match chafa(&bytes, protocol, size, binary) {
        Ok(art) => Some(art),
        Err(error) => {
            tracing::debug!("not drawing the image inline: {error}");
            None
        }
    }
}

/// `--probe=off` is deliberate: probing puts a query on the terminal and waits
/// for its answer, which is not something a tool result should do.
fn chafa(bytes: &[u8], protocol: Protocol, size: &str, binary: &str) -> Result<String, String> {
    let mut child = Command::new(binary)
        .args([
            "--probe=off",
            "-f",
            protocol.chafa_format(),
            "-s",
            size,
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not run {binary}: {error}"))?;

    let mut stdin = child.stdin.take().ok_or("renderer stdin unavailable")?;
    // The renderer reads its input to completion and writes a small escape
    // stream, so writing before reading cannot deadlock on a full pipe.
    if stdin.write_all(bytes).is_err() {
        // An image the renderer cannot parse closes the pipe early; the exit
        // status below is the error worth reporting, not the broken pipe.
    }
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|error| format!("{binary}: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{binary} failed: {}",
            stderr.trim().lines().next().unwrap_or("no output")
        ));
    }
    let art = String::from_utf8_lossy(&output.stdout).to_string();
    if art.trim().is_empty() {
        return Err(format!("{binary} produced no image"));
    }
    Ok(art)
}

/// Draw `image` where output is currently being written.
///
/// The renderer is run only for an interactive terminal: a pipe or a file gets
/// the escape bytes as garbage, and `--output-format json` must stay parseable.
/// Returns whether anything was drawn, so a caller can say something else when
/// the terminal cannot show it.
pub fn draw(image: &ImageContent) -> bool {
    if !std::io::stdout().is_terminal() {
        return false;
    }
    let Some(protocol) = protocol() else {
        return false;
    };
    let size = setting("GOOSE_CLI_IMAGE_SIZE", DEFAULT_SIZE);
    let binary = setting("GOOSE_CLI_IMAGE_BIN", DEFAULT_BINARY);
    match art_for(image, protocol, &size, &binary) {
        Some(art) => {
            // The escape sequence is written verbatim: the tool-output printer
            // strips control characters, which would leave the image as noise.
            println!();
            print!("{art}");
            let _ = std::io::stdout().flush();
            true
        }
        None => false,
    }
}

/// Draw the image in a tool result.
///
/// The result's own summary line already says what the image is, so nothing is
/// printed when the terminal cannot show it.
pub fn print_inline(image: &ImageContent) {
    draw(image);
}

/// Draw an image that is a message in its own right, with no summary line to
/// fall back on.
///
/// A terminal that cannot show it gets one line naming the image. The base64
/// was never meant to be read, so it is not printed either way.
pub fn print_message_image(image: &ImageContent) {
    if !draw(image) {
        println!("{}", message_image_summary(image));
    }
}

/// The line shown in place of an image this terminal cannot draw. The base64
/// is never printed: it is not something a person can read.
fn message_image_summary(image: &ImageContent) -> String {
    // base64 carries three bytes in every four characters, and a trailing
    // group of two characters carries one more.
    let unpadded = image.data.trim_end_matches('=').len();
    let bytes = unpadded / 4 * 3 + (unpadded % 4).saturating_sub(1);
    format!("Image: [{}, {bytes} bytes]", image.mime_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_names_a_protocol_or_nothing() {
        assert_eq!(configured_protocol("kitty"), Some(Protocol::Kitty));
        assert_eq!(configured_protocol(" SIXEL "), Some(Protocol::Sixels));
        assert_eq!(configured_protocol("iterm2"), Some(Protocol::Iterm));
        assert_eq!(configured_protocol("off"), None);
        assert_eq!(configured_protocol("teletext"), None);
    }

    #[test]
    fn only_terminals_that_speak_the_protocol_are_guessed_at() {
        assert_eq!(detected_protocol(Some("1"), None), Some(Protocol::Kitty));
        assert_eq!(
            detected_protocol(None, Some("ghostty")),
            Some(Protocol::Kitty)
        );
        assert_eq!(
            detected_protocol(None, Some("WezTerm")),
            Some(Protocol::Kitty)
        );
        assert_eq!(
            detected_protocol(None, Some("kitty")),
            Some(Protocol::Kitty)
        );
        assert_eq!(detected_protocol(None, Some("iTerm.app")), None);
        assert_eq!(detected_protocol(None, Some("xterm")), None);
        assert_eq!(detected_protocol(None, None), None);
    }

    /// An 8x8 gradient, so the rendering tests need no image crate and no
    /// file on disk.
    fn tiny_png() -> ImageContent {
        const GRADIENT_8X8: &str = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAIElEQVQI12NkYGhgwAZYVCTeYZdQFX9Lso53JOsg0Q4Ah3QIwtyKM/MAAAAASUVORK5CYII=";
        ImageContent::new(GRADIENT_8X8, "image/png")
    }

    #[test]
    fn renders_an_escape_sequence_rather_than_text() {
        let art = match art_for(&tiny_png(), Protocol::Kitty, "10x5", DEFAULT_BINARY) {
            Some(art) => art,
            None => {
                eprintln!("skipping: renderer unavailable");
                return;
            }
        };
        assert!(
            art.starts_with('\u{1b}'),
            "the output must be a terminal escape sequence, got {:?}",
            art.chars().take(20).collect::<String>()
        );
        assert!(
            art.contains("\u{1b}_G"),
            "expected the kitty graphics introducer"
        );
    }

    #[test]
    fn an_undecodable_image_draws_nothing() {
        let image = ImageContent::new("not base64 !!!", "image/png");
        assert!(art_for(&image, Protocol::Kitty, "10x5", DEFAULT_BINARY).is_none());
    }

    #[test]
    fn the_fallback_line_describes_the_image_without_dumping_it() {
        let summary = message_image_summary(&tiny_png());
        assert_eq!(summary, "Image: [image/png, 89 bytes]");
        assert!(!summary.contains("iVBOR"));
    }

    #[test]
    fn a_missing_renderer_draws_nothing() {
        let image = tiny_png();
        assert!(art_for(&image, Protocol::Kitty, "10x5", "chafa-does-not-exist-xyz").is_none());
    }
}
