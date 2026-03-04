# Information and References

Use this directory to store project reference material that should guide implementation decisions.

## How to Use This Folder
- Put downloaded notes, PDFs, snippets, or design docs directly in `information/`.
- Prefer descriptive file names such as `attention-notes.md` or `optimizer-paper-summary.md`.
- Keep each file focused on one topic so it is easy to reference later.

## External References
- Sebastian Raschka article:
  https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up
- LLMs-from-scratch repository:
  https://github.com/rasbt/LLMs-from-scratch
- Local submodule checkout:
  `external/LLMs-from-scratch`

## Local Reference Files
- Book PDF:
  `Build a Large Language Model (From Scratch) - Sebastian Raschka.pdf`
- Working notes derived from PDF + external sources:
  `raschka-reference-notes.md`
- Implementation tracker derived from those notes:
  `raschka-implementation-checklist.md`

## Suggested Subfolders (Optional)
- `information/papers/` for papers and summaries
- `information/datasets/` for dataset notes and schema docs
- `information/experiments/` for run logs and observations

## Submodule Maintenance
- Initialize after clone:
  `git submodule update --init --recursive`
- Pull latest upstream reference code:
  `git submodule update --remote information/external/LLMs-from-scratch`
