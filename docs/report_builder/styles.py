"""
scripts/report_builder/styles.py
--------------------------------
Typography, layout, color palette, and XML styling helpers for the BTP Report.
Designed to meet formal IIT Bombay academic submission standards.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Academic Color Palette
COLOR_NAVY_DARK = RGBColor(0, 32, 64)       # #002040 - Main Titles
COLOR_NAVY_PRIMARY = RGBColor(0, 51, 102)   # #003366 - Heading 1 & Table Headers
COLOR_STEEL_BLUE = RGBColor(43, 76, 126)    # #2B4C7E - Heading 2
COLOR_SLATE_BLUE = RGBColor(74, 107, 148)   # #4A6B94 - Heading 3
COLOR_CHARCOAL = RGBColor(51, 51, 51)       # #333333 - Body Text
COLOR_MUTED_GRAY = RGBColor(102, 102, 102)  # #666666 - Captions & Subtitles
COLOR_CALLOUT_BORDER = "003366"             # Hex for XML
COLOR_CALLOUT_BG = "F0F4F8"                 # Light Ice Blue
COLOR_ALT_ROW_BG = "F8F9FA"                 # Soft Gray for alternating rows
COLOR_BORDER_GRAY = "D3D3D3"                # Subtle table borders

def setup_document_formatting(doc):
    """Sets standard A4 size, 1-inch margins, and default typography."""
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        
        # Section 1 (Cover Page) has clean header & footer
        section.header.is_linked_to_previous = False
        section.footer.is_linked_to_previous = False
        for p in section.header.paragraphs:
            p.text = ""
        for p in section.footer.paragraphs:
            p.text = ""

    # Configure Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = COLOR_CHARCOAL
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

def add_page_number_field(run):
    """Injects a dynamic w:fldSimple PAGE element into a run."""
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    run._r.append(fldSimple)

def add_toc_entry(doc, title, page_str, level=1, is_bold=False):
    """Adds a Table of Contents / List entry with native dot leaders flush to right margin."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2.5 if level == 1 else 1)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.1
    if level == 2:
        p.paragraph_format.left_indent = Inches(0.25)
    elif level == 3:
        p.paragraph_format.left_indent = Inches(0.45)
        
    pPr = p._p.get_or_add_pPr()
    tabs = OxmlElement('w:tabs')
    tab = OxmlElement('w:tab')
    tab.set(qn('w:val'), 'right')
    tab.set(qn('w:leader'), 'dot')
    tab.set(qn('w:pos'), '9028')  # 6.27 inches in dxa (A4: 8.27 in - 2.0 in = 6.27 in)
    tabs.append(tab)
    pPr.append(tabs)
    
    r1 = p.add_run(title)
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(10 if level > 1 else 10.5)
    r1.font.bold = is_bold
    r1.font.color.rgb = COLOR_NAVY_PRIMARY if is_bold else COLOR_CHARCOAL
    
    r2 = p.add_run('\t' + str(page_str))
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10 if level > 1 else 10.5)
    r2.font.bold = is_bold
    r2.font.color.rgb = COLOR_NAVY_PRIMARY if is_bold else COLOR_CHARCOAL
    return p

def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = COLOR_NAVY_DARK
    return p

def add_subtitle(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = COLOR_MUTED_GRAY
    return p

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_NAVY_PRIMARY
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_STEEL_BLUE
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_SLATE_BLUE
    return p

def add_paragraph(doc, text, bold_prefix=None, italic=False, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_CHARCOAL
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.italic = italic
    run.font.color.rgb = COLOR_CHARCOAL
    return p

def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_CHARCOAL
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.color.rgb = COLOR_CHARCOAL
    return p

def add_callout(doc, title, text):
    """Creates a styled single-cell callout box with a colored left accent border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    
    # XML Shading & Left Border
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{COLOR_CALLOUT_BG}"/>')
    tcPr.append(shd)
    
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="36" w:space="0" w:color="{COLOR_CALLOUT_BORDER}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    # Padding
    mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'  <w:top w:w="120" w:type="dxa"/>'
        f'  <w:left w:w="180" w:type="dxa"/>'
        f'  <w:bottom w:w="120" w:type="dxa"/>'
        f'  <w:right w:w="180" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(mar)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)
    r_t = p.add_run(f"KEY FINDING / RESEARCH INSIGHT: {title}\n")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(10.5)
    r_t.font.bold = True
    r_t.font.color.rgb = COLOR_NAVY_PRIMARY
    
    r_b = p.add_run(text)
    r_b.font.name = 'Times New Roman'
    r_b.font.size = Pt(10)
    r_b.font.italic = True
    r_b.font.color.rgb = COLOR_CHARCOAL
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_styled_table(doc, headers, data, col_widths=None, alignment=None):
    """Builds an elegant academic table with shaded header, alternating rows, and clean borders."""
    tbl = doc.add_table(rows=len(data) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header Row
    hdr_row = tbl.rows[0]
    trPr = hdr_row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    
    for i, h_text in enumerate(headers):
        cell = hdr_row.cells[i]
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="003366"/>'))
        
        # Border
        b = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="12" w:space="0" w:color="002040"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:bottom w:val="single" w:sz="18" w:space="0" w:color="002040"/>'
            f'  <w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(b)
        
        # Padding
        mar = parse_xml(
            f'<w:tcMar {nsdecls("w")}>'
            f'  <w:top w:w="120" w:type="dxa"/>'
            f'  <w:bottom w:w="120" w:type="dxa"/>'
            f'  <w:left w:w="140" w:type="dxa"/>'
            f'  <w:right w:w="140" w:type="dxa"/>'
            f'</w:tcMar>'
        )
        tcPr.append(mar)
        
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (alignment and alignment[i] == 'C') else (WD_ALIGN_PARAGRAPH.RIGHT if (alignment and alignment[i] == 'R') else WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(h_text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for r_idx, row_data in enumerate(data):
        row = tbl.rows[r_idx + 1]
        bg_col = COLOR_ALT_ROW_BG if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            tcPr = cell._tc.get_or_add_tcPr()
            if bg_col != "FFFFFF":
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_col}"/>'))
                
            b = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{COLOR_BORDER_GRAY}"/>'
                f'  <w:left w:val="none"/>'
                f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{COLOR_BORDER_GRAY}"/>'
                f'  <w:right w:val="none"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(b)
            
            mar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'  <w:top w:w="90" w:type="dxa"/>'
                f'  <w:bottom w:w="90" w:type="dxa"/>'
                f'  <w:left w:w="140" w:type="dxa"/>'
                f'  <w:right w:w="140" w:type="dxa"/>'
                f'</w:tcMar>'
            )
            tcPr.append(mar)
            
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (alignment and alignment[c_idx] == 'C') else (WD_ALIGN_PARAGRAPH.RIGHT if (alignment and alignment[c_idx] == 'R') else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
            run.font.color.rgb = COLOR_CHARCOAL

    # Ensure rows do not split awkwardly across pages
    for r in tbl.rows:
        trPr = r._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

    if col_widths:
        for r in tbl.rows:
            for c_idx, w in enumerate(col_widths):
                r.cells[c_idx].width = Inches(w)
                
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return tbl

def add_figure_with_caption(doc, img_path, caption, fig_num, width_in=6.2):
    """Inserts a centered image with a formatted academic caption."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    
    if os.path.exists(img_path):
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_in))
    else:
        run_err = p_img.add_run(f"[FIGURE PLACEHOLDER: Image not found at {img_path}]")
        run_err.font.italic = True
        run_err.font.color.rgb = RGBColor(180, 50, 50)
        
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    
    r_num = p_cap.add_run(f"Figure {fig_num}: ")
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(9.5)
    r_num.font.bold = True
    r_num.font.color.rgb = COLOR_NAVY_PRIMARY
    
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = COLOR_CHARCOAL
    return p_cap

def add_equation(doc, eq_text, eq_num):
    """Formats a displayed equation with right-aligned numbering."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    
    r_eq = p.add_run(f"{eq_text}")
    r_eq.font.name = 'Times New Roman'
    r_eq.font.size = Pt(10.5)
    r_eq.font.italic = True
    r_eq.font.color.rgb = COLOR_CHARCOAL
    
    r_tab = p.add_run(f"\t\t\t\t\t\t({eq_num})")
    r_tab.font.name = 'Times New Roman'
    r_tab.font.size = Pt(10)
    r_tab.font.bold = True
    r_tab.font.color.rgb = COLOR_MUTED_GRAY
    return p

