# Subtitle Workflow

Local subtitle production workflow for extracting, translating, rendering, and polishing bilingual TV subtitles.

## Main Entry

Run the full auto workflow from the subtitle/video directory:

```powershell
python .\run_all.py .
```

The auto mode groups inputs by current subtitle state:

- videos with no embedded or external subtitles: Step 1 to Step 8, then Step 9
- videos with embedded subtitles but no external subtitles: Step 0, then Step 6 to Step 8, then Step 9
- videos with external monolingual subtitles: Step 6 to Step 8, then Step 9
- existing bilingual ASS subtitles: Step 9 only

Intermediate files are preserved by default. Step 10 cleanup is opt-in.

## Model Flow

- Step 7 uses the local Qwen model for dialogue, lyrics, OCR translation, and cultural notes.
- Step 9 uses the backend model for final polish, OCR cleanup, and reusable hint extraction.
- Step 9 hint files are fed back into Step 7 by matching only guidance relevant to the current segment.

## Reusable Hint Files

- `subtitle_glossary.json`: names, terms, and cultural-note glossary
- `common_mistranslation_hints.json`: recurring proper-noun, phrase, and mistranslation traps
- `common_phrase_correction_hints.json`: reusable colloquial or industry phrase guidance
- `ocr_low_value_short_texts.json`: OCR fragments that should usually be dropped when shown alone

## Notes

Do not commit generated subtitles, logs, media, or `temp/` cache files. They are ignored by `.gitignore`.
