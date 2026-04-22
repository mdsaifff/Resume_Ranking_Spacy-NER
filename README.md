# Resume Ranking using SpaCy NER

This project explores a simple approach to ranking resumes against a given job description using Named Entity Recognition (NER) with SpaCy.

The core idea is to move beyond basic keyword matching and instead compare structured information extracted from both resumes and job descriptions. Using SpaCy, the system identifies relevant entities such as skills, qualifications, experience, and education, and uses these to estimate how well a candidate matches a role.

## How it works

The pipeline is straightforward:

- Parse resumes and job descriptions as raw text
- Use SpaCy’s NER to extract relevant entities
- Normalize and group extracted information (e.g., skills, degrees, roles)
- Compare the extracted entities between resume and job description
- Compute a similarity score to rank candidates

Instead of treating text as a flat bag of words, this approach focuses on *what* information is present and how it aligns with the job requirements.

## Why this approach

Traditional keyword-based systems often fail when:
- Different terms are used for the same skill
- Context matters (e.g., “worked with Python” vs “expert in Python”)
- Important information is embedded in unstructured text

By using NER, we attempt to capture more meaningful signals from the data. While this is still a simplified model and not production-ready, it demonstrates how structured information extraction can improve resume screening.

## Limitations

This is an experimental project and has a few limitations:
- SpaCy’s default NER model is not fully optimized for resume-specific entities
- Entity extraction may miss domain-specific skills or misclassify terms
- Similarity scoring is relatively simple and can be improved with more advanced methods (e.g., embeddings or semantic similarity)
