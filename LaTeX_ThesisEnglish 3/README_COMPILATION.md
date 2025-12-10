# Compilation Instructions

To compile this thesis, you need a LaTeX distribution installed (like TeX Live or MacTeX).

## Prerequisites
1.  **LaTeX Distribution**: Ensure you have `pdflatex` and `bibtex` installed.
2.  **Missing Package**: The file `BachelorThesis.tex` uses `\usepackage{rutitlepage}`, which seems to be missing from this directory. You may need to download `rutitlepage.sty` (likely from the Radboud University website or your supervisor) and place it in this directory.

## How to Compile
I have created a script `compile_thesis.sh` that runs the necessary commands.

1.  Open a terminal in this directory.
2.  Run the script:
    ```bash
    ./compile_thesis.sh
    ```

## Manual Compilation
If you prefer to run the commands manually:

```bash
pdflatex BachelorThesis
bibtex BachelorThesis
pdflatex BachelorThesis
pdflatex BachelorThesis
```

## Fixes Applied
I have corrected an incorrect file path in `BachelorThesis.tex`:
- Changed `\input{LaTeX_ThesisEnglish/discussion_&_limintations}` to `\input{discussion_&_limintations}`.
