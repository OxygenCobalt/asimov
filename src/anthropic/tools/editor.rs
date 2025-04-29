use crate::{
    anthropic::Claude,
    core::{llm::Content, tool::ProviderTool},
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::{fs, io};

pub struct Editor {
    model: Claude,
}

#[derive(Deserialize, JsonSchema, Debug)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum EditorInput {
    /// View the contents of the file at the given path.
    View {
        path: PathBuf,
        /// 1-based start and end lines (inclusive)
        view_range: Option<[u64; 2]>,
    },
    /// Replace a specific instance of a given string with a new string in the file at the given path.
    /// There should be only one instance of the old string in the file.
    StrReplace {
        path: PathBuf,
        old_str: String,
        new_str: String,
    },
    /// Create a new file at the given path with the provided text. Overwrites if exists.
    Create { path: PathBuf, file_text: String },
    /// Insert a new line of text at the given 1-based line number.
    Insert {
        path: PathBuf,
        /// 1-based line number to insert at
        insert_line: u64,
        new_str: String,
    },
    /// Revert the last edit to the file.
    UndoEdit { path: PathBuf },
}

// Helper to map std::io::Error to Content
fn io_error_to_content(err: io::Error, path: &PathBuf) -> Content {
    Content::Text(format!("I/O error for file {:?}: {}", path, err))
}

impl Editor {
    pub fn new(model: Claude) -> Self {
        Self { model }
    }
}

impl ProviderTool for Editor {
    type Input = EditorInput;

    fn id(&self) -> String {
        match self.model {
            Claude::ThreeDotFiveSonnet => "text_editor_20241022".to_string(),
            Claude::ThreeDotSevenSonnet => "text_editor_20250124".to_string(),
        }
    }

    fn name(&self) -> String {
        "str_replace_editor".to_string()
    }

    fn call(&self, input: Self::Input) -> Result<Vec<Content>, Content> {
        match input {
            EditorInput::View { path, view_range } => {
                // Check if the path is a directory first
                match fs::metadata(&path) {
                    Ok(metadata) => {
                        if metadata.is_dir() {
                            // List directory contents
                            let mut entries = String::new();
                            match fs::read_dir(&path) {
                                Ok(read_dir) => {
                                    for entry in read_dir {
                                        match entry {
                                            Ok(dir_entry) => {
                                                entries.push_str(&format!(
                                                    "{}\n",
                                                    dir_entry.path().display()
                                                ));
                                            }
                                            Err(e) => return Err(io_error_to_content(e, &path)), // Error reading specific entry
                                        }
                                    }
                                    Ok(vec![Content::Text(format!(
                                        "Directory listing for {:?}:\n{}",
                                        path, entries
                                    ))])
                                }
                                Err(e) => Err(io_error_to_content(e, &path)), // Error reading directory itself
                            }
                        } else {
                            // It's a file, proceed with reading content
                            let content = fs::read_to_string(&path)
                                .map_err(|e| io_error_to_content(e, &path))?;

                            match view_range {
                                Some(range) => {
                                    // Handle specific range view
                                    let lines: Vec<&str> = content.lines().collect();
                                    let start_line = (range[0].saturating_sub(1)) as usize; // Convert 1-based to 0-based
                                    let end_line = (range[1]).min(lines.len() as u64) as usize; // Convert 1-based end to 0-based exclusive index, capped

                                    if start_line >= end_line || start_line >= lines.len() {
                                        return Err(Content::Text(format!(
                                            "Invalid view range [{}-{}] for file with {} lines.",
                                            range[0],
                                            range[1],
                                            lines.len()
                                        )));
                                    }

                                    let selected_lines = lines[start_line..end_line].join("\n");
                                    Ok(vec![Content::Text(selected_lines)])
                                }
                                None => {
                                    // No range specified, return entire file content
                                    Ok(vec![Content::Text(content)])
                                }
                            }
                        }
                    }
                    Err(e) => Err(io_error_to_content(e, &path)), // Error getting metadata
                }
            }
            EditorInput::StrReplace {
                path,
                old_str,
                new_str,
            } => {
                let content =
                    fs::read_to_string(&path).map_err(|e| io_error_to_content(e, &path))?;

                let matches: Vec<_> = content.match_indices(&old_str).collect();
                if matches.len() != 1 {
                    return Err(Content::Text(format!(
                        "Expected exactly one occurrence of '{}' in {:?}, but found {}.",
                        old_str,
                        path,
                        matches.len()
                    )));
                }

                let new_content = content.replacen(&old_str, &new_str, 1);
                fs::write(&path, new_content).map_err(|e| io_error_to_content(e, &path))?;

                Ok(vec![Content::Text(format!(
                    "Successfully replaced string in {:?}",
                    path
                ))])
            }
            EditorInput::Create { path, file_text } => {
                // Ensure parent directory exists
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|e| io_error_to_content(e, &path))?;
                }
                fs::write(&path, file_text).map_err(|e| io_error_to_content(e, &path))?;
                Ok(vec![Content::Text(format!(
                    "Successfully created/updated file {:?}",
                    path
                ))])
            }
            EditorInput::Insert {
                path,
                insert_line,
                new_str,
            } => {
                // Ensure insert_line is 1 or greater
                if insert_line == 0 {
                    return Err(Content::Text(
                        "Insert line number must be 1 or greater.".to_string(),
                    ));
                }

                let content =
                    fs::read_to_string(&path).map_err(|e| io_error_to_content(e, &path))?;
                let mut lines: Vec<String> = content.lines().map(String::from).collect();

                let insert_index = (insert_line.saturating_sub(1)) as usize; // Convert 1-based to 0-based index

                if insert_index > lines.len() {
                    return Err(Content::Text(format!(
                        "Insert line {} is out of bounds for file with {} lines.",
                        insert_line,
                        lines.len()
                    )));
                }

                lines.insert(insert_index, new_str);

                let new_content = lines.join("\n");
                fs::write(&path, new_content).map_err(|e| io_error_to_content(e, &path))?;

                Ok(vec![Content::Text(format!(
                    "Successfully inserted line at {} in {:?}",
                    insert_line, path
                ))])
            }
            EditorInput::UndoEdit { path } => {
                // Proper undo requires history tracking, which is complex.
                Err(Content::Text(format!(
                    "Undo functionality is not implemented for file {:?}",
                    path
                )))
            }
        }
    }
}
