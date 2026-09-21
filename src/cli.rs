//! Command-line interface for mai-timing.
//!
//! Provides commands for training timing models and predicting phoneme durations.

use clap::{Parser, Subcommand};

use crate::error::TimingError;
#[cfg(feature = "train")]
use crate::model::TimingMetadata;
use crate::model::UtteranceInput;
#[cfg(feature = "train")]
use crate::resources::{load_language_info_from_path, save_timing_model};
use crate::resources::{load_timing_lookup, load_timing_model};
#[cfg(feature = "train")]
use crate::train::load_label_map;

/// mai-timing: Open-source phoneme timing model for voice synthesis
#[derive(Parser)]
#[command(name = "mai-timing")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a timing model from TextGrid files
    Train {
        /// Path to directory containing TextGrid files
        textgrid_dir: String,

        /// Optional path to a global phoneme YAML file declaring phoneme types,
        /// Omit to use the inventory bundled with flow.
        #[arg(short = 'g', long)]
        language_info_path: Option<String>,

        /// Optional path to a label map YAML file that translates TextGrid phoneme labels
        /// to the labels used in the language file. Omit if your TextGrid files already
        /// use the same labels.
        #[arg(short = 'm', long)]
        label_map_path: Option<String>,

        /// Output path for the timing model YAML
        #[arg(short = 'o', long, default_value = "timing_model.yaml")]
        output: String,

        /// Library name for metadata
        #[arg(long, default_value = "Unknown")]
        library: String,

        /// Language name for metadata
        #[arg(long, default_value = "Unknown")]
        language_name: String,

        /// Voice color name for metadata
        #[arg(long, default_value = "Default")]
        voice_color: String,

        /// Minimum phoneme duration in ms to include
        #[arg(long, default_value = "10")]
        min_duration: u16,

        /// Maximum phoneme duration in ms to include
        #[arg(long, default_value = "2000")]
        max_duration: u16,
    },

    /// Predict timings for phoneme sequences
    Predict {
        /// Path to timing model YAML file
        timing_model: String,

        /// Optional path to a global phoneme YAML file declaring phoneme types,
        /// used for fallback classification of unseen clusters. Omit to use the
        /// inventory bundled with flow.
        #[arg(short = 'g', long)]
        language_info: Option<String>,

        /// Phoneme sequence as JSON array of X-SAMPA strings (e.g., '["k", "a", "t"]')
        #[arg(short = 'i', long)]
        input: Option<String>,

        /// Path to utterance YAML file (alternative to -i)
        #[arg(short = 'f', long)]
        file: Option<String>,

        /// Output format: yaml, json, or simple
        #[arg(short = 'o', long, default_value = "yaml")]
        output_format: String,
    },

    /// Show information about a timing model
    Info {
        /// Path to timing model YAML file
        timing_model: String,
    },
}

/// Run the CLI application.
pub fn run() -> Result<(), TimingError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            textgrid_dir,
            language_info_path,
            label_map_path,
            output,
            library,
            language_name,
            voice_color,
            min_duration,
            max_duration,
        } => run_train(
            &textgrid_dir,
            language_info_path.as_deref(),
            label_map_path.as_deref(),
            &output,
            &library,
            &language_name,
            &voice_color,
            min_duration,
            max_duration,
        ),
        Commands::Predict {
            timing_model,
            language_info,
            input,
            file,
            output_format,
        } => run_predict(
            &timing_model,
            language_info.as_deref(),
            input,
            file,
            &output_format,
        ),
        Commands::Info { timing_model } => run_info(&timing_model),
    }
}

#[cfg(feature = "train")]
fn run_train(
    textgrid_dir: &str,
    language_info_path: Option<&str>,
    label_map_path: Option<&str>,
    output: &str,
    library: &str,
    language_name: &str,
    voice_color: &str,
    min_duration: u16,
    max_duration: u16,
) -> Result<(), TimingError> {
    match language_info_path {
        Some(p) => eprintln!("Loading global phoneme inventory from {}...", p),
        None => eprintln!("Using bundled global phoneme inventory..."),
    }
    let language_info = load_language_info_from_path(language_info_path)?;

    let label_map = label_map_path
        .map(|p| {
            eprintln!("Loading label map from {}...", p);
            load_label_map(p)
        })
        .transpose()?;

    let metadata = TimingMetadata::new(library, language_name, voice_color);
    let config = crate::train::TrainingConfig {
        min_duration_ms: min_duration,
        max_duration_ms: max_duration,
    };

    eprintln!("Training from TextGrid files in {}...", textgrid_dir);

    let model = crate::train::train_from_textgrids(
        textgrid_dir,
        &language_info,
        label_map.as_ref(),
        metadata,
        Some(config),
    )?;

    eprintln!(
        "Built model with {} cluster patterns and {} generic patterns",
        model.cluster_timings.len(),
        model.generic_timings.len()
    );

    save_timing_model(&model, output)?;
    eprintln!("Saved timing model to {}", output);

    Ok(())
}

#[cfg(not(feature = "train"))]
fn run_train(
    _textgrid_dir: &str,
    _language_info_path: Option<&str>,
    _label_map_path: Option<&str>,
    _output: &str,
    _library: &str,
    _language_name: &str,
    _voice_color: &str,
    _min_duration: u16,
    _max_duration: u16,
) -> Result<(), TimingError> {
    Err(TimingError::Other(
        "Training feature not enabled. Rebuild with --features train".to_string(),
    ))
}

fn run_predict(
    timing_model_path: &str,
    global_path: Option<&str>,
    input: Option<String>,
    file: Option<String>,
    output_format: &str,
) -> Result<(), TimingError> {
    // Get phonemes from input or file
    let phonemes: Vec<String> = if let Some(input_str) = input {
        serde_json::from_str(&input_str)
            .map_err(|e| TimingError::Other(format!("Failed to parse input as JSON: {}", e)))?
    } else if let Some(file_path) = file {
        let content =
            std::fs::read_to_string(&file_path).map_err(|e| TimingError::io(&file_path, e))?;
        let utterance: UtteranceInput =
            serde_yaml::from_str(&content).map_err(|e| TimingError::yaml(&file_path, e))?;
        utterance.phonemes
    } else {
        return Err(TimingError::Other(
            "Either --input or --file must be provided".to_string(),
        ));
    };

    if phonemes.is_empty() {
        return Err(TimingError::EmptyInput);
    }

    // Load timing model and the phoneme type classifier (custom or bundled)
    let lookup = load_timing_lookup(timing_model_path, global_path)?;

    // Get predictions
    let result = lookup.predict(&phonemes);

    // Output in requested format
    match output_format {
        "json" => {
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| TimingError::Other(format!("Failed to serialize: {}", e)))?;
            println!("{}", json);
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&result)
                .map_err(|e| TimingError::Other(format!("Failed to serialize: {}", e)))?;
            print!("{}", yaml);
        }
        "simple" => {
            for timing in &result.timings {
                println!("{}: {}ms", timing.phoneme, timing.duration_ms);
            }
            println!("---");
            println!("Total: {}ms", result.total_duration_ms);
        }
        _ => {
            return Err(TimingError::Other(format!(
                "Unknown output format: {}",
                output_format
            )));
        }
    }

    Ok(())
}

fn run_info(timing_model_path: &str) -> Result<(), TimingError> {
    let model = load_timing_model(timing_model_path)?;

    println!("Timing Model: {}", timing_model_path);
    println!("  Library: {}", model.metadata.library);
    println!("  Language: {}", model.metadata.language);
    println!("  Voice Color: {}", model.metadata.voice_color);

    if let Some(created_at) = &model.metadata.created_at {
        println!("  Created: {}", created_at);
    }

    println!("  Cluster patterns: {}", model.cluster_timings.len());
    println!("  Generic patterns: {}", model.generic_timings.len());

    // Count total samples
    let cluster_samples: usize = model.cluster_timings.iter().map(|c| c.samples.len()).sum();
    let generic_samples: usize = model.generic_timings.iter().map(|g| g.samples.len()).sum();
    println!("  Total cluster samples: {}", cluster_samples);
    println!("  Total generic samples: {}", generic_samples);

    if let Some(files) = &model.metadata.source_files {
        println!("  Source files: {}", files.len());
    }

    Ok(())
}
