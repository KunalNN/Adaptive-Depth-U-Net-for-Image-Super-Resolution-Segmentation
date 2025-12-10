#!/bin/bash
# Compile the thesis
pdflatex -interaction=nonstopmode BachelorThesis
bibtex BachelorThesis
pdflatex -interaction=nonstopmode BachelorThesis
pdflatex -interaction=nonstopmode BachelorThesis
