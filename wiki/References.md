# References

## Primary External References
- Sebastian Raschka article:  
  https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up
- Sebastian Raschka code repository:  
  https://github.com/rasbt/LLMs-from-scratch

## In-Repo Reference Notes
- `information/README.md`
- `information/raschka-reference-notes.md`
- `information/raschka-implementation-checklist.md`
- `information/external/LLMs-from-scratch` (submodule checkout)

## How We Use References
- Use Raschka sources as design and implementation guidance.
- Translate concepts into this repository's modular code layout.
- Track adaptation status in checklist and roadmap docs.

## Sync Reference Repo
```bash
git submodule update --init --recursive
git submodule update --remote information/external/LLMs-from-scratch
```
