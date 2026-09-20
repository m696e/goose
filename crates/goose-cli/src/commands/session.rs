use anyhow::{Context, Result};

use cliclack::{confirm, multiselect, select};
use console::style;
use etcetera::home_dir;
use goose::session::{
    export_session_to_html, export_session_to_markdown, generate_diagnostics, DiagnosticsLevel,
    Session, SessionManager, SessionType,
};
use goose::utils::safe_truncate;
use regex::Regex;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::path::PathBuf;

const TRUNCATED_DESC_LENGTH: usize = 60;
const SESSION_PICKER_ROWS: usize = 10;
const SEARCH_SNIPPET_LENGTH: usize = 120;

fn display_path_with_tilde(path: &Path) -> String {
    #[cfg(not(target_os = "windows"))]
    if let Ok(home) = home_dir() {
        if let Ok(stripped) = path.strip_prefix(&home) {
            return format!("~/{}", stripped.display());
        }
    }
    path.display().to_string()
}

async fn remove_sessions(session_manager: &SessionManager, sessions: Vec<Session>) -> Result<()> {
    println!("The following sessions will be removed:");
    for session in &sessions {
        println!(
            "- {} {} ({})",
            session.id,
            session.name,
            display_path_with_tilde(&session.working_dir)
        );
    }

    let should_delete = confirm("Are you sure you want to delete these sessions?")
        .initial_value(false)
        .interact()?;

    if should_delete {
        for session in sessions {
            session_manager.delete_session(&session.id).await?;
            println!("Session `{}` removed.", session.id);
        }
    } else {
        println!("Skipping deletion of the sessions.");
    }

    Ok(())
}

fn prompt_interactive_session_removal(sessions: &[Session]) -> Result<Vec<Session>> {
    if sessions.is_empty() {
        println!("No sessions to delete.");
        return Ok(vec![]);
    }

    let mut selector = multiselect(
        "Select sessions to delete (use spacebar, Enter to confirm, Ctrl+C to cancel):",
    );

    let display_map: std::collections::HashMap<String, Session> = sessions
        .iter()
        .map(|s| {
            let desc = if s.name.is_empty() {
                "(no name)"
            } else {
                &s.name
            };
            let truncated_desc = safe_truncate(desc, TRUNCATED_DESC_LENGTH);
            let display_text = format!(
                "{} - {} ({}) - {}",
                session_activity_at(s),
                truncated_desc,
                s.id,
                display_path_with_tilde(&s.working_dir)
            );
            (display_text, s.clone())
        })
        .collect();

    for display_text in display_map.keys() {
        selector = selector.item(display_text.clone(), display_text.clone(), "");
    }

    let selected_display_texts: Vec<String> = selector.interact()?;

    let selected_sessions: Vec<Session> = selected_display_texts
        .into_iter()
        .filter_map(|text| display_map.get(&text).cloned())
        .collect();

    Ok(selected_sessions)
}

pub async fn handle_session_remove(
    session_id: Option<String>,
    name: Option<String>,
    regex_string: Option<String>,
) -> Result<()> {
    let session_manager = SessionManager::instance();

    let matched_sessions: Vec<Session>;

    if let Some(id_val) = session_id {
        match session_manager.get_session(&id_val, false).await {
            Ok(session) => matched_sessions = vec![session],
            Err(_) => return Err(anyhow::anyhow!("Session ID '{}' not found.", id_val)),
        }
    } else if let Some(name_val) = name {
        let all_sessions = session_manager.list_all_sessions().await?;
        if let Some(session) = all_sessions.into_iter().find(|s| s.name == name_val) {
            matched_sessions = vec![session];
        } else {
            return Err(anyhow::anyhow!(
                "Session with name '{}' not found.",
                name_val
            ));
        }
    } else if let Some(regex_val) = regex_string {
        let session_regex = Regex::new(&regex_val)
            .with_context(|| format!("Invalid regex pattern '{}'", regex_val))?;

        let visible_sessions = session_manager.list_sessions().await?;
        matched_sessions = visible_sessions
            .into_iter()
            .filter(|session| session_regex.is_match(&session.id))
            .collect();

        if matched_sessions.is_empty() {
            println!("Regex string '{}' does not match any sessions", regex_val);
            return Ok(());
        }
    } else {
        let visible_sessions = session_manager.list_sessions().await?;
        if visible_sessions.is_empty() {
            return Err(anyhow::anyhow!("No sessions found."));
        }
        matched_sessions = prompt_interactive_session_removal(&visible_sessions)?;
    }

    if matched_sessions.is_empty() {
        return Ok(());
    }

    remove_sessions(&session_manager, matched_sessions).await
}

pub async fn handle_session_rename(session_id: String, new_name: String) -> Result<()> {
    let new_name = new_name.trim();
    if new_name.is_empty() {
        anyhow::bail!("Session name cannot be empty");
    }

    let session_manager = SessionManager::instance();

    session_manager
        .update(&session_id)
        .user_provided_name(new_name)
        .apply()
        .await?;

    println!("Session `{}` renamed to '{}'.", session_id, new_name);
    Ok(())
}

fn write_line_or_broken_pipe_ok<W: Write>(out: &mut W, line: &str) -> Result<bool> {
    match writeln!(out, "{line}") {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn session_activity_at(session: &Session) -> chrono::DateTime<chrono::Utc> {
    session.last_message_at.unwrap_or(session.updated_at)
}

fn session_list_limit_if_safe_to_push_down(
    ascending: bool,
    working_dir: Option<&Path>,
    limit: Option<usize>,
) -> Option<usize> {
    if !ascending && working_dir.is_none() {
        limit
    } else {
        None
    }
}

pub async fn handle_session_list(
    format: String,
    ascending: bool,
    working_dir: Option<PathBuf>,
    limit: Option<usize>,
) -> Result<()> {
    let session_manager = SessionManager::instance();
    let mut sessions = if let Some(limit) =
        session_list_limit_if_safe_to_push_down(ascending, working_dir.as_deref(), limit)
    {
        session_manager.list_sessions_with_limit(limit).await?
    } else {
        session_manager.list_sessions().await?
    };

    if let Some(ref pat) = working_dir {
        let pat_lower = pat.to_string_lossy().to_lowercase();
        sessions.retain(|s| {
            s.working_dir
                .to_string_lossy()
                .to_lowercase()
                .contains(&pat_lower)
        });
    }

    if ascending {
        sessions.sort_by_key(session_activity_at);
    } else {
        sessions.sort_by_key(|b| std::cmp::Reverse(session_activity_at(b)));
    }

    if let Some(n) = limit {
        sessions.truncate(n);
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();

    match format.as_str() {
        "json" => {
            let payload = serde_json::to_string(&sessions)?;
            if !write_line_or_broken_pipe_ok(&mut out, &payload)? {
                return Ok(());
            }
        }
        _ => {
            if sessions.is_empty() {
                if !write_line_or_broken_pipe_ok(&mut out, "No sessions found")? {
                    return Ok(());
                }
                return Ok(());
            }

            if !write_line_or_broken_pipe_ok(&mut out, "Available sessions:")? {
                return Ok(());
            }

            for session in sessions {
                let output = format!(
                    "{} - {} - {} - {}",
                    session.id,
                    session.name,
                    session_activity_at(&session),
                    display_path_with_tilde(&session.working_dir)
                );
                if !write_line_or_broken_pipe_ok(&mut out, &output)? {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// Window of `content` around the first match of any query word.
///
/// The search returns whole messages, so truncating from the front can hide the
/// very word that matched. This centres the window on the first hit instead.
fn matching_snippet(content: &str, query: &str, max_chars: usize) -> String {
    let flat = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = flat.chars().collect();
    if chars.len() <= max_chars {
        return flat;
    }

    let lowered: Vec<char> = flat.to_lowercase().chars().collect();
    if lowered.len() != chars.len() {
        return safe_truncate(&flat, max_chars);
    }

    let hit = query
        .split_whitespace()
        .filter_map(|word| {
            let needle: Vec<char> = word.to_lowercase().chars().collect();
            if needle.is_empty() || needle.len() > lowered.len() {
                return None;
            }
            lowered
                .windows(needle.len())
                .position(|window| window == needle.as_slice())
        })
        .min();

    let Some(hit) = hit else {
        return safe_truncate(&flat, max_chars);
    };

    let start = hit.saturating_sub(max_chars / 3);
    let end = (start + max_chars).min(chars.len());

    format!(
        "{}{}{}",
        if start > 0 { "..." } else { "" },
        chars[start..end].iter().collect::<String>(),
        if end < chars.len() { "..." } else { "" }
    )
}

/// Wrap every occurrence of a query word in `text` with `style_match`.
///
/// Matching is case-insensitive, as in the search itself. `text` comes from
/// [`matching_snippet`], so it already contains at least one hit.
fn highlight_matches(text: &str, query: &str, style_match: impl Fn(&str) -> String) -> String {
    let needles: Vec<Vec<char>> = query
        .split_whitespace()
        .map(|word| word.to_lowercase().chars().collect())
        .filter(|needle: &Vec<char>| !needle.is_empty())
        .collect();

    let chars: Vec<char> = text.chars().collect();
    let lowered: Vec<char> = text.to_lowercase().chars().collect();
    if needles.is_empty() || lowered.len() != chars.len() {
        return text.to_string();
    }

    let mut out = String::new();
    let mut index = 0;
    while index < chars.len() {
        let matched = needles.iter().find_map(|needle| {
            lowered[index..]
                .starts_with(needle.as_slice())
                .then_some(needle.len())
        });

        match matched {
            Some(len) => {
                let hit: String = chars[index..index + len].iter().collect();
                out.push_str(&style_match(&hit));
                index += len;
            }
            None => {
                out.push(chars[index]);
                index += 1;
            }
        }
    }

    out
}

pub async fn handle_session_search(query: String, format: String, limit: usize) -> Result<()> {
    let session_manager = SessionManager::instance();
    let results = session_manager
        .search_chat_history(
            &query,
            Some(limit),
            None,
            None,
            None,
            vec![SessionType::User, SessionType::Scheduled],
        )
        .await?;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    if format == "json" {
        let payload = serde_json::to_string(&results)?;
        write_line_or_broken_pipe_ok(&mut out, &payload)?;
        return Ok(());
    }

    if results.total_matches == 0 {
        write_line_or_broken_pipe_ok(&mut out, &format!("No sessions match \"{query}\""))?;
        return Ok(());
    }

    let summary = format!(
        "{} matching message(s) across {} session(s) for \"{}\":",
        results.total_matches,
        results.results.len(),
        query
    );
    if !write_line_or_broken_pipe_ok(&mut out, &style(summary).bold().to_string())? {
        return Ok(());
    }

    for result in &results.results {
        let description = if result.session_description.is_empty() {
            "(unnamed)".to_string()
        } else {
            result.session_description.clone()
        };

        let header = format!(
            "{} - {} - {} - {}",
            style(&result.session_id).cyan(),
            style(description).bold(),
            style(result.last_activity).dim(),
            style(display_path_with_tilde(Path::new(
                &result.session_working_dir
            )))
            .dim(),
        );
        if !write_line_or_broken_pipe_ok(&mut out, &header)? {
            return Ok(());
        }

        for message in &result.messages {
            let role = style(format!("[{}]", message.role)).dim();
            let snippet = matching_snippet(&message.content, &query, SEARCH_SNIPPET_LENGTH);
            let snippet = highlight_matches(&snippet, &query, |hit| {
                style(hit).yellow().bold().to_string()
            });

            let line = format!("    {role} {snippet}");
            if !write_line_or_broken_pipe_ok(&mut out, &line)? {
                return Ok(());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod session_search_tests {
    use super::*;

    #[test]
    fn short_content_is_flattened_not_elided() {
        let snippet = matching_snippet("line one\n\n   line two", "two", 120);

        assert_eq!(snippet, "line one line two");
    }

    #[test]
    fn long_content_keeps_the_match_visible() {
        let content = format!("{} NEEDLE {}", "filler ".repeat(40), "tail ".repeat(40));

        let snippet = matching_snippet(&content, "needle", 60);

        assert!(snippet.contains("NEEDLE"), "{snippet}");
        assert!(snippet.starts_with("..."), "{snippet}");
        assert!(snippet.ends_with("..."), "{snippet}");
    }

    #[test]
    fn unmatched_long_content_falls_back_to_the_front() {
        let content = "word ".repeat(200);

        let snippet = matching_snippet(&content, "absent", 40);

        assert!(snippet.starts_with("word"), "{snippet}");
        assert!(snippet.ends_with("..."), "{snippet}");
    }

    #[test]
    fn highlight_wraps_every_match_case_insensitively() {
        let styled = highlight_matches("Tor over SSH, tor again", "tor", |m| format!("<{m}>"));

        assert_eq!(styled, "<Tor> over SSH, <tor> again");
    }

    #[test]
    fn highlight_covers_every_query_word() {
        let styled = highlight_matches("alpha then beta", "beta alpha", |m| format!("[{m}]"));

        assert_eq!(styled, "[alpha] then [beta]");
    }

    #[test]
    fn highlight_is_a_noop_without_query_words() {
        let styled = highlight_matches("nothing here", "   ", |m| format!("<{m}>"));

        assert_eq!(styled, "nothing here");
    }
}

#[cfg(test)]
mod session_list_tests {
    use super::*;

    #[test]
    fn pushes_down_only_safe_session_list_limits() {
        assert_eq!(
            session_list_limit_if_safe_to_push_down(false, None, Some(50)),
            Some(50)
        );
        assert_eq!(
            session_list_limit_if_safe_to_push_down(true, None, Some(50)),
            None
        );
        assert_eq!(
            session_list_limit_if_safe_to_push_down(false, Some(Path::new("/tmp")), Some(50)),
            None
        );
        assert_eq!(
            session_list_limit_if_safe_to_push_down(false, None, None),
            None
        );
    }
}

pub async fn handle_session_export(
    session_id: String,
    output_path: Option<PathBuf>,
    format: String,
) -> Result<()> {
    let session_manager = SessionManager::instance();
    let session = match session_manager.get_session(&session_id, true).await {
        Ok(session) => session,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Session '{}' not found or failed to read: {}",
                session_id,
                e
            ));
        }
    };

    let output = match format.as_str() {
        "json" => serde_json::to_string_pretty(&session)?,
        "yaml" => serde_yaml::to_string(&session)?,
        "markdown" => {
            let conversation = session
                .conversation
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Session has no messages"))?;
            export_session_to_markdown(conversation.user_visible_messages(), &session.name)
        }
        "html" => export_session_to_html(&session)?,
        _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
    };

    if let Some(output_path) = output_path {
        fs::write(&output_path, output).with_context(|| {
            format!("Failed to write to output file: {}", output_path.display())
        })?;
        println!("Session exported to {}", output_path.display());
    } else {
        println!("{}", output);
    }

    Ok(())
}

pub async fn handle_session_import(input: String) -> Result<()> {
    let json = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read session import file: {input}"))?;

    let format = goose::session::import_formats::detect_format(&json);
    let label = match format {
        goose::session::import_formats::ImportFormat::Goose => "goose",
        goose::session::import_formats::ImportFormat::ClaudeCode => "Claude Code",
        goose::session::import_formats::ImportFormat::Codex => "Codex",
        goose::session::import_formats::ImportFormat::Pi => "Pi",
    };
    println!("Detected format: {}", label);

    let session_manager = SessionManager::instance();
    let session = session_manager
        .import_session(&json, Some(SessionType::User))
        .await?;

    println!("Session imported:");
    println!("{} - {}", session.id, session.name);

    Ok(())
}

#[cfg(unix)]
fn open_diagnostics_output(path: &Path) -> io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(windows)]
fn open_diagnostics_output(path: &Path) -> io::Result<fs::File> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use winapi::um::winbase::FILE_FLAG_OPEN_REPARSE_POINT;
    use winapi::um::winnt::FILE_ATTRIBUTE_REPARSE_POINT;

    let file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "diagnostics output must be a regular file",
        ));
    }
    file.set_len(0)?;
    Ok(file)
}

#[cfg(not(any(unix, windows)))]
fn open_diagnostics_output(path: &Path) -> io::Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

pub async fn handle_diagnostics(session_id: &str, output_path: Option<PathBuf>) -> Result<()> {
    println!(
        "Generating diagnostics report for session '{}'...",
        session_id
    );

    let session_manager = SessionManager::instance();
    let diagnostics_report =
        generate_diagnostics(&session_manager, session_id, DiagnosticsLevel::Full)
            .await
            .with_context(|| {
                format!(
                    "Failed to generate diagnostics report for session '{}'",
                    session_id
                )
            })?;
    let diagnostics_data = serde_json::to_vec_pretty(&diagnostics_report)
        .context("Failed to serialize diagnostics report")?;

    let output_file = if let Some(path) = output_path {
        path.clone()
    } else {
        PathBuf::from(format!("diagnostics_{}.json", session_id))
    };

    let mut file = open_diagnostics_output(&output_file).context(format!(
        "Failed to create output file: {}",
        output_file.display()
    ))?;

    file.write_all(&diagnostics_data)
        .context("Failed to write diagnostics data")?;

    println!("Diagnostics report saved to: {}", output_file.display());

    Ok(())
}

/// Prompt the user to interactively select a session
///
/// Shows a list of available sessions and lets the user select one
pub async fn prompt_interactive_session_selection(
    session_manager: &SessionManager,
) -> Result<String> {
    let sessions = session_manager.list_sessions().await?;

    if sessions.is_empty() {
        return Err(anyhow::anyhow!("No sessions found"));
    }

    // Build the selection prompt
    let mut selector = select("Select a session");

    // Map to display text
    let display_map: std::collections::HashMap<String, Session> = sessions
        .iter()
        .map(|s| {
            let desc = if s.name.is_empty() {
                "(no name)"
            } else {
                &s.name
            };
            let truncated_desc = safe_truncate(desc, TRUNCATED_DESC_LENGTH);

            let display_text = format!("{} - {} ({})", s.updated_at, truncated_desc, s.id);
            (display_text, s.clone())
        })
        .collect();

    // Add each session as an option
    for display_text in display_map.keys() {
        selector = selector.item(display_text.clone(), display_text.clone(), "");
    }

    // Add a cancel option
    let cancel_value = String::from("cancel");
    selector = selector.item(cancel_value, "Cancel", "");

    // Get user selection
    let selected_display_text: String = selector.interact()?;

    if selected_display_text == "cancel" {
        return Err(anyhow::anyhow!("No session selected"));
    }

    // Retrieve the selected session
    if let Some(session) = display_map.get(&selected_display_text) {
        Ok(session.id.clone())
    } else {
        Err(anyhow::anyhow!("Invalid selection"))
    }
}

/// Display text for a session in the resume picker.
///
/// cliclack's filter mode scores this string (Jaro-Winkler over the lowercased
/// label, plus a bonus when every typed word appears in it), so the label carries
/// the fields a user is likely to type. The name leads to earn the prefix bonus.
///
/// The last-message snippet is deliberately absent: it is only hydrated for the
/// ACP list path, so it is always `None` on this one.
fn session_picker_label(session: &Session) -> String {
    let name = if session.name.is_empty() {
        "(unnamed)"
    } else {
        session.name.as_str()
    };

    format!(
        "{} - {} - {} - {}",
        safe_truncate(name, TRUNCATED_DESC_LENGTH),
        session.id,
        session_activity_at(session),
        safe_truncate(
            &display_path_with_tilde(&session.working_dir),
            TRUNCATED_DESC_LENGTH
        ),
    )
}

/// Pick a session to resume from a list that narrows as the user types.
///
/// The most recently used session is pre-selected, so pressing Enter reproduces
/// the non-interactive "resume the last session" behaviour. Returns `Ok(None)`
/// when there is no user session to offer; what that means is the caller's call.
pub async fn prompt_interactive_session_resume(
    session_manager: &SessionManager,
) -> Result<Option<String>> {
    let sessions = session_manager
        .list_sessions_by_types(&[SessionType::User])
        .await?;

    let Some(most_recent) = sessions.first() else {
        return Ok(None);
    };

    let mut selector = select("Select a session to resume (type to filter):")
        .filter_mode()
        .max_rows(SESSION_PICKER_ROWS)
        .initial_value(most_recent.id.clone());

    for session in &sessions {
        selector = selector.item(session.id.clone(), session_picker_label(session), "");
    }

    Ok(Some(selector.interact()?))
}

#[cfg(test)]
mod session_picker_tests {
    use super::*;
    use chrono::Utc;

    fn session(name: &str, dir: &str) -> Session {
        Session {
            id: "20260101_120000".to_string(),
            working_dir: PathBuf::from(dir),
            name: name.to_string(),
            updated_at: Utc::now(),
            ..Default::default()
        }
    }

    #[test]
    fn label_carries_every_field_a_user_might_type() {
        let label = session_picker_label(&session("project-x", "/home/min/work/api"));

        assert!(label.contains("project-x"), "{label}");
        assert!(label.contains("20260101_120000"), "{label}");
        assert!(label.contains("work/api"), "{label}");
    }

    #[test]
    fn label_marks_unnamed_sessions() {
        let label = session_picker_label(&session("", "/tmp"));

        assert!(label.contains("(unnamed)"), "{label}");
    }
}

#[cfg(test)]
mod diagnostics_output_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_new_output_file() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("diagnostics.json");

        let mut file = open_diagnostics_output(&output).unwrap();
        file.write_all(b"diagnostics").unwrap();
        drop(file);

        assert_eq!(fs::read(&output).unwrap(), b"diagnostics");
    }

    #[test]
    fn truncates_existing_regular_output_file() {
        let temp_dir = TempDir::new().unwrap();
        let output = temp_dir.path().join("diagnostics.json");
        fs::write(&output, "old diagnostics").unwrap();

        let mut file = open_diagnostics_output(&output).unwrap();
        file.write_all(b"new").unwrap();
        drop(file);

        assert_eq!(fs::read(&output).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn refuses_existing_symlink_output() {
        use std::os::unix::fs::symlink;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("target.json");
        let output = temp_dir.path().join("diagnostics.json");
        fs::write(&target, "preserve").unwrap();
        symlink(&target, &output).unwrap();

        assert!(open_diagnostics_output(&output).is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "preserve");
    }

    #[cfg(unix)]
    #[test]
    fn refuses_dangling_symlink_output() {
        use std::os::unix::fs::symlink;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("missing.json");
        let output = temp_dir.path().join("diagnostics.json");
        symlink(&target, &output).unwrap();

        assert!(open_diagnostics_output(&output).is_err());
        assert!(!target.exists());
    }

    #[cfg(windows)]
    #[test]
    fn refuses_existing_symlink_output() {
        use std::os::windows::fs::symlink_file;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("target.json");
        let output = temp_dir.path().join("diagnostics.json");
        fs::write(&target, "preserve").unwrap();
        if symlink_file(&target, &output).is_err() {
            return;
        }

        assert!(open_diagnostics_output(&output).is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "preserve");
    }
}
