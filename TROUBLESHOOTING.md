Troubleshooting: Extraction & Summaries

If you see "No text could be extracted", empty summaries, or the app hangs while generating a summary, try the following:

- Verify the uploaded files contain selectable (text) content; scanned images require OCR.
- For PDF extraction, ensure `pypdf` or `unstructured` is installed and available in your environment.
- If a DOCX or PPTX fails to extract, try opening it in Word/PowerPoint and saving as a new file.
- Paste a representative text snippet into the app's text box as a quick test.
- After clicking "Generate Summary", expand the diagnostics (snippets and sources) to see what was extracted.
- If vector store building is slow or fails, ensure `sentence-transformers` and `faiss-cpu` are installed.

If problems persist, attach a sample file that reproduces the issue and run `python scripts/test_loaders.py` to inspect loader outputs.
