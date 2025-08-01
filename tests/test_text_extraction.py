"""
Testing the text-extraction submodule and ensuring the quality of text extraction.

The tested code might be in _page.py.
"""

import re
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from pypdf import PdfReader, mult
from pypdf._text_extraction import set_custom_rtl
from pypdf._text_extraction._layout_mode._fixed_width_page import text_show_operations
from pypdf.errors import ParseError, PdfReadError
from pypdf.generic import ContentStream

from . import get_data_from_url

TESTS_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_ROOT.parent
RESOURCE_ROOT = PROJECT_ROOT / "resources"
SAMPLE_ROOT = PROJECT_ROOT / "sample-files"


@pytest.mark.samples
@pytest.mark.parametrize(("visitor_text"), [None, lambda a, b, c, d, e: None])  # noqa: ARG005
def test_multi_language(visitor_text):
    reader = PdfReader(RESOURCE_ROOT / "multilang.pdf")
    txt = reader.pages[0].extract_text(visitor_text=visitor_text)
    assert "Hello World" in txt, "English not correctly extracted"
    # iss #1296
    assert "مرحبا بالعالم" in txt, "Arabic not correctly extracted"
    assert "Привет, мир" in txt, "Russian not correctly extracted"
    assert "你好世界" in txt, "Chinese not correctly extracted"
    assert "สวัสดีชาวโลก" in txt, "Thai not correctly extracted"
    assert "こんにちは世界" in txt, "Japanese not correctly extracted"
    # check customizations
    set_custom_rtl(None, None, "Russian:")
    assert ":naissuR" in reader.pages[0].extract_text(
        visitor_text=visitor_text
    ), "(1) CUSTOM_RTL_SPECIAL_CHARS failed"
    set_custom_rtl(None, None, [ord(x) for x in "Russian:"])
    assert ":naissuR" in reader.pages[0].extract_text(
        visitor_text=visitor_text
    ), "(2) CUSTOM_RTL_SPECIAL_CHARS failed"
    set_custom_rtl(0, 255, None)
    assert ":hsilgnE" in reader.pages[0].extract_text(
        visitor_text=visitor_text
    ), "CUSTOM_RTL_MIN/MAX failed"
    set_custom_rtl("A", "z", [])
    assert ":hsilgnE" in reader.pages[0].extract_text(
        visitor_text=visitor_text
    ), "CUSTOM_RTL_MIN/MAX failed"
    set_custom_rtl(-1, -1, [])  # to prevent further errors

    reader = PdfReader(SAMPLE_ROOT / "015-arabic/habibi-rotated.pdf")
    assert "habibi" in reader.pages[0].extract_text(visitor_text=visitor_text)
    assert "حَبيبي" in reader.pages[0].extract_text(visitor_text=visitor_text)
    assert "habibi" in reader.pages[1].extract_text(visitor_text=visitor_text)
    assert "حَبيبي" in reader.pages[1].extract_text(visitor_text=visitor_text)
    assert "habibi" in reader.pages[2].extract_text(visitor_text=visitor_text)
    assert "حَبيبي" in reader.pages[2].extract_text(visitor_text=visitor_text)
    assert "habibi" in reader.pages[3].extract_text(visitor_text=visitor_text)
    assert "حَبيبي" in reader.pages[3].extract_text(visitor_text=visitor_text)


@pytest.mark.parametrize(
    ("file_name", "constraints"),
    [
        (
            "inkscape-abc.pdf",
            {
                "A": lambda x, y: 0 < x < 94 and 189 < y < 283,  # In upper left
                "B": lambda x, y: 94 < x < 189 and 94 < y < 189,  # In the center
                "C": lambda x, y: 189 < x < 283 and 0 < y < 94,  # In lower right
            },
        )
    ],
)
def test_visitor_text_matrices(file_name, constraints):
    """
    Checks if the matrices given to the visitor_text function when calling
    `extract_text` on the first page of `file_name` match some given constraints.
    `constraints` is a dictionary mapping a line of text to a constraint that should
    evaluate to `True` on its expected x,y-coordinates.
    """
    reader = PdfReader(RESOURCE_ROOT / file_name)

    lines = []

    def visitor_text(text, cm, tm, font_dict, font_size) -> None:
        ctm = mult(tm, cm)
        x = ctm[4]  # mult(tm, cm)[4]
        y = ctm[5]  # mult(tm, cm)[5]
        lines.append({"text": text, "x": x, "y": y})

    reader.pages[0].extract_text(visitor_text=visitor_text)

    for text, constraint in constraints.items():
        matches = [li for li in lines if li["text"].strip() == text]
        assert len(matches) <= 1, f"Multiple lines match {text}"
        assert len(matches) >= 1, f"No lines match {text}"

        x = matches[0]["x"]
        y = matches[0]["y"]
        assert constraint(x, y), f'Line "{text}" is wrong at x:{x}, y:{y}'


@pytest.mark.xfail(reason="known whitespace issue #2336")
@pytest.mark.enable_socket
def test_issue_2336():
    name = "Pesquisa-de-Precos-Combustiveis-novembro-2023.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(name=name)))
    page = reader.pages[0]
    actual_text = page.extract_text()
    assert "Beira Rio" in actual_text


def test_layout_mode_font_class_to_dict():
    from pypdf._text_extraction._layout_mode._font import Font  # noqa: PLC0415

    font = Font("foo", space_width=8, encoding="utf-8", char_map={}, font_dictionary={})
    assert Font.to_dict(font) == {
        "char_map": {},
        "encoding": "utf-8",
        "font_dictionary": {},
        "space_width": 8,
        "subtype": "foo",
        "width_map": {},
        "interpretable": True,
    }


@pytest.mark.enable_socket
@patch("pypdf._text_extraction._layout_mode._fixed_width_page.logger_warning")
def test_uninterpretable_type3_font(mock_logger_warning):
    url = "https://github.com/user-attachments/files/18551904/UninterpretableType3Font.pdf"
    name = "UninterpretableType3Font.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    page = reader.pages[0]
    assert page.extract_text(extraction_mode="layout") == ""
    mock_logger_warning.assert_called_with(
        "PDF contains an uninterpretable font. Output will be incomplete.",
        "pypdf._text_extraction._layout_mode._fixed_width_page"
    )


@pytest.mark.enable_socket
def test_layout_mode_epic_page_fonts():
    url = "https://github.com/py-pdf/pypdf/files/13836944/Epic.Page.PDF"
    name = "Epic Page.PDF"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    expected = (RESOURCE_ROOT / "Epic.Page.layout.txt").read_text(encoding="utf-8")
    assert expected == reader.pages[0].extract_text(extraction_mode="layout")


def test_layout_mode_uncommon_operators():
    # Coverage for layout mode Tc, Tz, Ts, ', ", TD, TL, and Tw
    reader = PdfReader(RESOURCE_ROOT / "toy.pdf")
    expected = (RESOURCE_ROOT / "toy.layout.txt").read_text(encoding="utf-8")
    assert expected == reader.pages[0].extract_text(extraction_mode="layout")


@pytest.mark.enable_socket
def test_layout_mode_type0_font_widths():
    # Cover both the 'int int int' and 'int [int int ...]' formats for Type0
    # /DescendantFonts /W array entries.
    url = "https://github.com/py-pdf/pypdf/files/13533204/Claim.Maker.Alerts.Guide_pg2.PDF"
    name = "Claim Maker Alerts Guide_pg2.PDF"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    expected = (RESOURCE_ROOT / "Claim Maker Alerts Guide_pg2.layout.txt").read_text(
        encoding="utf-8"
    )
    assert expected == reader.pages[0].extract_text(extraction_mode="layout")


@pytest.mark.enable_socket
def test_layout_mode_indirect_sequence_font_widths():
    # Cover the situation where the sequence for font widths is an IndirectObject
    # https://github.com/py-pdf/pypdf/pull/2788
    url = "https://github.com/user-attachments/files/16491621/2788_example.pdf"
    name = "2788_example.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    assert reader.pages[0].extract_text(extraction_mode="layout") == ""
    url = "https://github.com/user-attachments/files/16491619/2788_example_malformed.pdf"
    name = "2788_example_malformed.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    with pytest.raises(ParseError) as exc:
        reader.pages[0].extract_text(extraction_mode="layout")
        assert str(exc.value).startswith("Invalid font width definition")


def dummy_visitor_text(text, ctm, tm, fd, fs):
    pass


@patch("pypdf._page.logger_warning")
def test_layout_mode_warnings(mock_logger_warning):
    # Check that a warning is issued when an argument is ignored
    reader = PdfReader(RESOURCE_ROOT / "hello-world.pdf")
    page = reader.pages[0]
    page.extract_text(extraction_mode="plain", visitor_text=dummy_visitor_text)
    mock_logger_warning.assert_not_called()
    page.extract_text(extraction_mode="layout", visitor_text=dummy_visitor_text)
    mock_logger_warning.assert_called_with(
        "Argument visitor_text is ignored in layout mode", "pypdf._page"
    )


@pytest.mark.enable_socket
def test_space_with_one_unit_smaller_than_font_width():
    """Tests for #1328"""
    url = "https://github.com/py-pdf/pypdf/files/9498481/0004.pdf"
    name = "iss1328.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    page = reader.pages[0]
    extracted = page.extract_text()
    assert "Reporting crude oil leak.\n" in extracted


@pytest.mark.enable_socket
def test_space_position_calculation():
    """Tests for #1153"""
    url = "https://github.com/py-pdf/pypdf/files/9164743/file-0.pdf"
    name = "iss1153.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    page = reader.pages[3]
    extracted = page.extract_text()
    assert "Shortly after the Geneva BOF session, the" in extracted


def test_text_leading_height_unit():
    """Tests for #2262"""
    reader = PdfReader(RESOURCE_ROOT / "toy.pdf")
    page = reader.pages[0]
    extracted = page.extract_text()
    assert "Something[cited]\n" in extracted


def test_layout_mode_space_vertically_font_height_weight():
    """Tests layout mode with vertical space and font height weight (issue #2915)"""
    with open(RESOURCE_ROOT / "crazyones.pdf", "rb") as inputfile:
        # Load PDF file from file
        reader = PdfReader(inputfile)
        page = reader.pages[0]

        # Normal behaviour
        with open(RESOURCE_ROOT / "crazyones_layout_vertical_space.txt", "rb") as pdftext_file:
            pdftext = pdftext_file.read()

        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=True).encode("utf-8")

        # Compare the text of the PDF to a known source
        for expected_line, actual_line in zip(text.splitlines(), pdftext.splitlines()):
            assert expected_line == actual_line

        pdftext = pdftext.replace(b"\r\n", b"\n")  # fix for windows
        assert text == pdftext

        # Blank lines are added to truly separate paragraphs
        with open(RESOURCE_ROOT / "crazyones_layout_vertical_space_font_height_weight.txt", "rb") as pdftext_file:
            pdftext = pdftext_file.read()

        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=True,
                                 layout_mode_font_height_weight=0.85).encode("utf-8")

        # Compare the text of the PDF to a known source
        for expected_line, actual_line in zip(text.splitlines(), pdftext.splitlines()):
            assert expected_line == actual_line

        pdftext = pdftext.replace(b"\r\n", b"\n")  # fix for windows
        assert text == pdftext


@pytest.mark.enable_socket
def test_infinite_loop_arrays():
    """Tests for #2928"""
    url = "https://github.com/user-attachments/files/17576546/arrayabruptending.pdf"
    name = "arrayabruptending.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))

    page = reader.pages[0]
    extracted = page.extract_text()
    assert "RNA structure comparison" in extracted


@pytest.mark.enable_socket
def test_content_stream_is_dictionary_object(caplog):
    """Tests for #2995"""
    url = "https://github.com/user-attachments/files/18049322/6fa5fd46-5f98-4a67-800d-5e2362b0164f.pdf"
    name = "iss2995.pdf"
    data = get_data_from_url(url, name=name)

    reader = PdfReader(BytesIO(data))
    page = reader.pages[0]
    assert "\nYours faithfully   \n" in page.extract_text()
    assert "Expected StreamObject, got DictionaryObject instead. Data might be wrong." in caplog.text
    caplog.clear()

    reader = PdfReader(BytesIO(data), strict=True)
    page = reader.pages[0]
    with pytest.raises(PdfReadError) as exception:
        page.extract_text()
    assert (
        "Invalid Elementary Object starting with b\\'\\\\x18\\' @3557: b\\'ateDecode/Length 629\\\\x18ck["
        in exception.value.args[0]
    )


@pytest.mark.enable_socket
def test_tz_with_no_operands():
    """Tests for #2975"""
    url = "https://github.com/user-attachments/files/17974120/9E5E080E-C8DB-4A6B-822B-9A67DC04E526-120438.pdf"
    name = "iss2975.pdf"
    data = get_data_from_url(url, name=name)

    reader = PdfReader(BytesIO(data))
    page = reader.pages[1]
    assert "\nThankyouforyourattentiontothismatter.\n" in page.extract_text()


@pytest.mark.enable_socket
def test_iss3060():
    """Test for not throwing 'font not set: is PDF missing a Tf operator'"""
    url = "https://github.com/user-attachments/files/18482531/test-anon.pdf"
    name = "iss3060.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    # pypdf.errors.PdfReadError: font not set: is PDF missing a Tf operator?
    txt = reader.pages[0].extract_text(extraction_mode="layout")
    assert txt.startswith(" *******")


@pytest.mark.enable_socket
def test_iss3074():
    """Test for not throwing 'ZeroDivisionError: float division by zero'"""
    url = "https://github.com/user-attachments/files/18533211/test-anon.pdf"
    name = "iss3074.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    # pypdf.errors.PdfReadError: ZeroDivisionError: float division by zero
    txt = reader.pages[0].extract_text(extraction_mode="layout")
    assert txt.strip().startswith("AAAAAA")


@pytest.mark.enable_socket
def test_layout_mode_text_state():
    """Ensure the text state is stored and reset with q/Q operators."""
    # Get the PDF from issue #3212
    url = "https://github.com/user-attachments/files/19396790/garbled.pdf"
    name = "garbled-font.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    # Get the txt from issue #3212 and normalize line endings
    txt_url = "https://github.com/user-attachments/files/19510731/garbled-font.layout.txt"
    txt_name = "garbled-font.layout.txt"
    expected = get_data_from_url(txt_url, name=txt_name).decode("utf-8").replace("\r\n", "\n")

    assert expected == reader.pages[0].extract_text(extraction_mode="layout")


@pytest.mark.enable_socket
def test_rotated_line_wrap():
    """Ensure correct 2D translation of rotated text after a line wrap."""
    # Get the PDF from issue #3247
    url = "https://github.com/user-attachments/files/19696918/link16-line-wrap.sanitized.pdf"
    name = "link16-line-wrap.sanitized.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    # Get the txt from issue #3247 and normalize line endings
    txt_url = "https://github.com/user-attachments/files/19696917/link16-line-wrap.sanitized.expected.txt"
    txt_name = "link16-line-wrap.sanitized.expected.txt"
    expected = get_data_from_url(txt_url, name=txt_name).decode("utf-8").replace("\r\n", "\n")

    assert expected == reader.pages[0].extract_text()


@pytest.mark.parametrize(
        ("op", "msg"),
        [
            (b"BT", "Unbalanced target operations, expected b'ET'."),
            (b"q", "Unbalanced target operations, expected b'Q'."),
        ],
)
def test_layout_mode_warns_on_malformed_content_stream(op, msg, caplog):
    """Ensures that imbalanced q/Q or EB/ET is handled gracefully."""
    text_show_operations(ops=iter([([], op)]), fonts={})
    assert caplog.records
    assert caplog.records[-1].msg == msg


def test_process_operation__cm_multiplication_issue():
    """Test for #3262."""
    reader = PdfReader(RESOURCE_ROOT / "crazyones.pdf")
    page = reader.pages[0]
    content = page.get_contents().get_data()
    content = content.replace(b" 1 0 0 1 72 720 cm ", b" 0.70278 65.3 163.36 cm ")
    stream = ContentStream(stream=None, pdf=reader)
    stream.set_data(content)
    page.replace_contents(stream)
    assert page.extract_text().startswith("The Crazy Ones\nOctober 14, 1998\n")


@pytest.mark.enable_socket
def test_rotated_layout_mode(caplog):
    """Ensures text extraction of rotated pages, as in issue #3270."""
    url = "https://github.com/user-attachments/files/19981120/rotated-page.pdf"
    name = "rotated-page.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))
    page = reader.pages[0]

    page.transfer_rotation_to_content()
    text = page.extract_text(extraction_mode="layout")

    assert not caplog.records, "No warnings should be issued"
    assert text, "Text matching the page rotation should be extracted"
    assert re.search(r"\r?\n +69\r?\n +UNCLASSIFIED$", text), "Contents should be in expected layout"


@pytest.mark.enable_socket
@pytest.mark.filterwarnings("ignore::pypdf.errors.PdfReadWarning")
def test_extract_text__none_objects():
    url = "https://github.com/user-attachments/files/18381726/tika-957721.pdf"
    name = "tika-957721.pdf"
    reader = PdfReader(BytesIO(get_data_from_url(url, name=name)))

    reader.pages[0].extract_text()
    reader.pages[8].extract_text()


@pytest.mark.enable_socket
def test_extract_text__with_visitor_text():
    def visitor_text(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        pass

    url = "https://github.com/user-attachments/files/18381718/tika-952016.pdf"
    name = "tika-952016.pdf"
    stream = BytesIO(get_data_from_url(url, name=name))
    reader = PdfReader(stream)
    page = reader.pages[0]
    page.extract_text(visitor_text=visitor_text)

    reader = PdfReader(BytesIO(get_data_from_url(name="TextAttack_paper.pdf")))
    page = reader.pages[0]
    page.extract_text(visitor_text=visitor_text)


@pytest.mark.enable_socket
def test_extract_text__restore_cm_stack_pop_error():
    url = "https://github.com/user-attachments/files/18381737/tika-966635.pdf"
    name = "tika-966635.pdf"
    stream = BytesIO(get_data_from_url(url, name=name))
    reader = PdfReader(stream)
    page = reader.pages[10]

    # There is a previous error we already omit ("pop from empty list"), thus
    # check for the message explicitly here.
    with pytest.raises(IndexError, match="list index out of range"):
        page.extract_text()
