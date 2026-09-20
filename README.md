# nChant Flow

This is the non-AI timing model for nChant, handling the allotting of consonants between and around vowels proportionally with respect to a minimum consonant to vowel ratio. In simpler terms, it decides how much of a note each consonant should take up while leaving room for the vowel so the singing doesn't get muddied. While we are planning an eventual AI timing model, this model has potential benefits, especially for fast singing, and will be left as an option.

## Usage

We provide a CLI for using the model, but the functions that CLI calls are available as a library. Apart from the timing data itself, the only input you may need is a global phoneme file — and that's optional, since flow bundles a default one.

### Training Input

#### Global Phoneme File

A single global file describes everything the model needs about a language's sound inventory:

- the articulatory **type** of every phoneme (plosive, fricative, vowel, …), used by the generic timing fallback;
- which phonemes are **vowels** which become the boundaries used to split consonant clusters during training; and
- the **sonorants** that can stand in for a vowel nucleus in the absence of one.

flow bundles a default global inventory, so you usually don't need to provide one. To model a specific language, supply your own file with the same structure — listing only that language's phonemes — and pass it as the global file.

Each top-level key is a phoneme type; its list holds the phonemes of that type:

```yaml
plosives:
  - "p_h"
  - "p"
  - "b"
  - "t"
  - "d"
  - "k"
  - "g"
  - "?"
affricates:
  - "ts"
  - "dz"
  - "tS"
  - "dZ"
fricatives:
  - "f"
  - "v"
  - "s"
  - "z"
  - "S"
  - "h"
sonorants:
  - "m"
  - "n"
  - "N"
  - "l"
  - "r"
  - "j"
  - "w"
taps:
  - "4"
vowels:
  - "{"
  - "E"
  - "I"
  - "U"
  - "u"
  - "i"
  - "@"
  - "aI"
  - "eI"
  - "OI"
  - "@U"
  - "aU"
```

The phoneme labels are the canonical labels — whatever notation system you choose. If your TextGrid files already use the same labels, no label map is needed. If they use different labels (e.g. a different phoneme notation, IPA, or a custom set), provide a label map to translate them.

#### Label Map (optional)

If your TextGrid phoneme labels differ from the labels in the global file, provide a label map. This is a simple flat YAML mapping from TextGrid label to global file label:

```yaml
ph: "p_h"
"p>": "p_}"
py: "p'"
bh: "b'"
```

Omit this file entirely if your TextGrid files already use the same labels as the global file.

### Training Output

The trainer reads TextGrid files from a directory and produces a `timing_model.yaml` file containing the learned cluster and generic timing data, ready for use with `predict`.

### Prediction Input

Pass a list of phoneme labels (in the global file's notation) for each note you want to time. Via the CLI you can supply them as a JSON array or a YAML file.

## Installation

Install [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) and then:

```sh
cargo install flow
```

### CLI

```sh
# Train a model — global file and label map are both optional
flow train ./textgrids \
  --output timing_model.yaml \
  --library "MyVoice" --language-name "English" --voice-color "Default"

# Train with a custom global phoneme file and a label map
flow train ./textgrids \
  --language-info english.yaml \
  --label-map arpabet_to_english.yaml \
  --output timing_model.yaml \
  --library "MyVoice" --language-name "English" --voice-color "Default"

# Predict timings — global file is optional (the bundled one is used by default)
flow predict timing_model.yaml \
  --input '["h","E","l","@U"]'

# Show model info
flow info timing_model.yaml
```

### Library

```rust
use nchant_flow::TimingEngine;

// The bundled global inventory supplies phoneme types plus vowels, diphthongs,
// and syllabic consonants. Use `from_paths_with_global` to pass a custom one.
let engine = TimingEngine::from_paths("timing_model.yaml")?;

let result = engine.predict(&["h".to_string(), "E".to_string(), "l".to_string(), "@U".to_string()]);
println!("{}", result);
```

To train programmatically:

```rust
use nchant_flow::{
    load_language_info_from_global, load_phoneme_map_from_global, load_label_map,
    train::train_from_textgrids, TimingMetadata,
};

// Pass None to use the bundled global inventory, or Some(path) for a custom one.
let phoneme_map = load_phoneme_map_from_global(None)?;
let language_info = load_language_info_from_global(None)?;

// Label map is optional — pass None if TextGrid labels already match
let label_map = Some(load_label_map("label_map.yaml")?);

let model = train_from_textgrids(
    "./textgrids",
    &phoneme_map,
    &language_info,
    label_map.as_ref(),
    TimingMetadata::new("MyVoice", "English", "Default"),
    None,
)?;
```
