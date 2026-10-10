import pymupdf
import pytest

from resume_ranker.parse import read_all, read_text


def test_read_text_txt_handles_utf8_and_bom():
    assert read_text("﻿héllo · world".encode("utf-8"), "a.txt") == "héllo · world"


def test_read_text_pdf():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Python developer")
    assert "Python developer" in read_text(pdf.tobytes(), "cv.PDF")


def test_read_text_pdf_without_text_layer():
    pdf = pymupdf.open()
    pdf.new_page()
    with pytest.raises(ValueError, match="no extractable text"):
        read_text(pdf.tobytes(), "scan.pdf")


@pytest.mark.parametrize("data,name", [(b"   \n", "a.txt"), (b"x", "a.docx"), (b"x", "noext")])
def test_read_text_rejects(data, name):
    with pytest.raises(ValueError):
        read_text(data, name)


def test_read_all_skips_bad_files():  # Review Focus 5
    docs, warnings = read_all([
        ("good.txt", b"python"), ("empty.txt", b""), ("broken.pdf", b"not a pdf"), ("cv.docx", b"x"),
    ])
    assert docs == [("good.txt", "python")]
    assert [w.split(":")[0] for w in warnings] == ["skipped empty.txt", "skipped broken.pdf", "skipped cv.docx"]
