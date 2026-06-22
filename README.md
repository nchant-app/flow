# Maghni Timing

This is the non-AI timing model for Maghni, handling the allotting of consonants between and around vowels proportionally with respect to a minimum consonant to vowel ratio. In simpler terms, it decides how much of a note each consonant should take up while leaving room for the vowel so the singing doesn't get muddied. While we are planning an eventual AI timing model, this model has potential benefits, especially for fast singing, and will be left as an option.

## Usage

We provide a CLI for using the model, but the functions that CLI calls are available as a library. Regardless, you will need two files/objects to utilize the model, apart from the timing data itself.

### Training Input

#### Phoneme Map

The phoneme map provides a mapping from the exact phonemes used within your TextGrids to our X-SAMPA system (M-SAMPA, when the need to differentiate arrives). This is to avoid making you convert your TextGrids. This should be similar to the following:

```yaml
base_map:
	- "ph": "p_h"
	- "p>": "p}"
	- "py": "p'"
	...etc
```

If any consonants are left off this list, or if none is passed at all, we assume your TextGrids are already in our format.

#### Language Information

The language information file/object defines the consonants that the are expected to be present within the TextGrids. If using the CLI, this should be a YAML file similar to the following:

```yaml
consonants:
	- "p_h"
	- "p"
	- "b"
	...etc
vowels:
	- "{"
	- "E"
	- "I"
	...etc
syllabic_consonants:
	- "m"
	- "n"
	- "l"
	- "N"
```

The `consonants` and `vowels` arrays tell the trainer which phonemes should be considered. We toss any non-consonants, since this timing model could be better described as a consonant ratio generator. The `syllabic_consonants` field, required only in the prediction flow, is used to determine if a note without any vowels should have a consonant designed as a stand-in vowel or if the note simply has no vowel. If any consonant in that vowel-less note is within the syllabic consonants field, it's treated as a vowel.

### Training Output

badum TODO

### Prediction Input

TODO: summary

#### Notes

For prediction, the notes you require to predict should be passed in as a simple TextGrid file with each section containing an array of the phonemes to be predicted. For example:

`[ h E ] [ l oU ] [w @r 5 d]`

The tier named "phonemes" will be used; if that doesn't exist, the first tier will be used. The arrays of phonemes must be split by spaces.

TODO: update functionality to fit this^^

## Usage and Installation

We have provided some default language information files for you to use, as well as some simple mappings from common systems, but all will need to be edited to fit your needs.

### CLI

The CLI program is simple. Install cargo, the Rust CLI tool, and install  `maghni-timing` with the following command:

```sh
cargo install maghni-timing
```

Then, you can run it like this:

```sh
maghni-timing train LANGUAGE_FILE.yaml OUTPUT_FILE.yaml [--phoneme_map arpabet_to_msampa.yaml]
```

TODO: update functionality to actually work like this

The model will be output as a YAML file, which can then be read by `maghni-timing predict`:

```sh
maghni-timing predict TIMING_MODEL.yaml NOTES.TextGrid OUTPUT.yaml
```

The output is of the `Library` format described above.

### Library

TODO: summary
TODO: after making lib use an object, add info about it here
