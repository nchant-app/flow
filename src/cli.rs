//! Command-line interface for mai-timing.
//!
//! Provides commands for training timing models and predicting phoneme durations.

#[cfg(feature = "cli")]
use clap::{Parser, Subcommand};

#[cfg(feature = "cli")]
use crate::error::TimingError;
#[cfg(feature = "cli")]
use crate::model::{TimingMetadata, UtteranceInput};
#[cfg(feature = "cli")]
use crate::predict::load_timing_lookup;
#[cfg(feature = "cli")]
use crate::train::{load_language_info, load_phoneme_map, load_timing_model, save_timing_model};

/// mai-timing: Open-source phoneme timing model for voice synthesis
#[cfg(feature = "cli")]
#[derive(Parser)]
#[command(name = "mai-timing")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[cfg(feature = "cli")]
#[derive(Subcommand)]
pub enum Commands {
    /// Train a timing model from TextGrid files
    Train {
        /// Path to directory containing TextGrid files
        textgrid_dir: String,

        /// Path to the mapping yaml file (TextGrid phoneme mapping -> phoneme type mappings)
        #[arg(short = 'm', long)]
        phoneme_map: String,

        /// Path to language YAML file (vowels, diphthongs, syllabic consonants)
        #[arg(short = 'l', long)]
        language_info: String,

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

        /// Path to mapping yaml file (internal phonemes -> desired phonemes)
        #[arg(short = 'g', long)]
        phoneme_map: String,

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
#[cfg(feature = "cli")]
pub fn run() -> Result<(), TimingError> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            textgrid_dir,
            phoneme_map,
            language_info,
            output,
            library,
            language_name,
            voice_color,
            min_duration,
            max_duration,
        } => run_train(
            &textgrid_dir,
            &phoneme_map,
            &language_info,
            &output,
            &library,
            &language_name,
            &voice_color,
            min_duration,
            max_duration,
        ),
        Commands::Predict {
            timing_model,
            phoneme_map,
            input,
            file,
            output_format,
        } => run_predict(&timing_model, &phoneme_map, input, file, &output_format),
        Commands::Info { timing_model } => run_info(&timing_model),
    }
}

#[cfg(feature = "cli")]
fn run_train(
    textgrid_dir: &str,
    phoneme_map: &str,
    language_path: &str,
    output: &str,
    library: &str,
    language_name: &str,
    voice_color: &str,
    min_duration: u16,
    max_duration: u16,
) -> Result<(), TimingError> {
    eprintln!("Loading phoneme map from {}...", phoneme_map);
    let phoneme_map = load_phoneme_map(phoneme_map)?;

    eprintln!("Loading language info from {}...", language_path);
    let language_info = load_language_info(language_path)?;

    let metadata = TimingMetadata::new(library, language_name, voice_color);
    let config = crate::train::TrainingConfig {
        min_duration_ms: min_duration,
        max_duration_ms: max_duration,
    };

    eprintln!("Training from TextGrid files in {}...", textgrid_dir);

    #[cfg(feature = "train")]
    {
        let model = crate::train::train_from_textgrids(
            textgrid_dir,
            &phoneme_map,
            &language_info,
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
    }

    #[cfg(not(feature = "train"))]
    {
        return Err(TimingError::Other(
            "Training feature not enabled. Rebuild with --features train".to_string(),
        ));
    }

    Ok(())
}

#[cfg(feature = "cli")]
fn run_predict(
    timing_model_path: &str,
    phoneme_map: &str,
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

    // Load timing model and phoneme map
    let lookup = load_timing_lookup(timing_model_path, phoneme_map)?;

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

#[cfg(feature = "cli")]
fn run_info(timing_model_path: &str) -> Result<(), TimingError> {
    let model = load_timing_model(timing_model_path)?;

    println!("Timing Model: {}", timing_model_path);
    println!("  Version: {}", model.version);
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
