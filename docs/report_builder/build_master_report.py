"""
scripts/report_builder/build_master_report.py
---------------------------------------------
Master builder script that synthesizes all modules and generates the complete,
professionally styled BTP Progress Report in DOCX format matching BTP_report_Part2.pdf.
"""

import sys
from pathlib import Path

# Ensure local report builder modules are importable
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import os
import shutil
import time
import docx
from docx.shared import Inches, Pt
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from styles import (
    setup_document_formatting, add_page_number_field,
    COLOR_MUTED_GRAY, COLOR_CHARCOAL
)
from front_matter import add_front_matter
from chapter1 import add_chapter_1
from chapter2 import add_chapter_2
from chapter3 import add_chapter_3
from chapter4 import add_chapter_4
from chapter5 import add_chapter_5
from chapter6 import add_chapter_6
from chapter7 import add_chapter_7
from chapter8 import add_chapter_8
from back_matter import add_back_matter

def main():
    start_time = time.time()
    print("=" * 80)
    print("STARTING BTP REPORT GENERATION: PRABHAKAR YADAV (23B0653)")
    print("=" * 80)

    workspace_root = Path.cwd()
    downloads_dir = Path(r"C:\Users\PRABHAKAR\Downloads")
    submission_dir = workspace_root / "FINAL_ORGANIZED_SUBMISSION" / "4_Reports"
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Document
    doc = docx.Document()

    print("\n[1/12] Configuring Section 1 (Title Page Layout & A4 Dimensions)...")
    setup_document_formatting(doc)

    print("[2/12] Assembling Front Matter (Cover with IITB Logo, Abstract, Contents, Figures, Tables)...")
    add_front_matter(doc)

    print("[3/12] Configuring Section 3 for Main Body (Decimal Page Numbering from 1, Running Header)...")
    sec3 = doc.add_section(WD_SECTION.NEW_PAGE)
    sec3.top_margin = Inches(0.85)
    sec3.bottom_margin = Inches(0.85)
    sec3.left_margin = Inches(1.0)
    sec3.right_margin = Inches(1.0)
    sec3.page_width = Inches(8.27)
    sec3.page_height = Inches(11.69)
    sec3.header.is_linked_to_previous = False
    sec3.footer.is_linked_to_previous = False

    # Ensure headers are completely clean and blank across all body pages (no top line)
    for p in sec3.header.paragraphs:
        p.text = ""

    # Decimal page numbering starting at 1
    sectPr3 = sec3._sectPr
    pgNumType3 = OxmlElement('w:pgNumType')
    pgNumType3.set(qn('w:fmt'), 'decimal')
    pgNumType3.set(qn('w:start'), '1')
    sectPr3.append(pgNumType3)

    p_f3 = sec3.footer.paragraphs[0]
    p_f3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f3 = p_f3.add_run()
    r_f3.font.name = "Times New Roman"
    r_f3.font.size = Pt(10)
    r_f3.font.color.rgb = COLOR_CHARCOAL
    add_page_number_field(r_f3)

    print("[4/12] Generating Chapter 1: Introduction...")
    add_chapter_1(doc)

    print("[5/12] Generating Chapter 2: Background and Literature Review...")
    add_chapter_2(doc)

    print("[6/12] Generating Chapter 3: Datasets and Ground-Truth Preparation...")
    add_chapter_3(doc)

    print("[7/12] Generating Chapter 4: Methodology and Model Architecture...")
    add_chapter_4(doc)

    print("[8/12] Generating Chapter 5: Experiments and Model Development...")
    add_chapter_5(doc)

    print("[9/12] Generating Chapter 6: Results, Benchmarks, and Visual Evaluation...")
    add_chapter_6(doc)

    print("[10/12] Generating Chapter 7: Implementation and System Workflow...")
    add_chapter_7(doc)

    print("[11/12] Generating Chapter 8: Conclusions and Future Work...")
    add_chapter_8(doc)

    print("[12/12] Generating References & Appendices A, B, C, D...")
    add_back_matter(doc)

    # Save to a temporary master file first to prevent corruption from file locks
    temp_out = workspace_root / "temp_generated_master_report.docx"
    print("\n" + "=" * 80)
    print(f"SAVING MASTER DOCX DOCUMENT...")
    doc.save(str(temp_out))
    print(f"Temporary Master: {temp_out} ({temp_out.stat().st_size / (1024*1024):.2f} MB)")

    # Target destinations
    targets = [
        workspace_root / "23b0653.docx",
        downloads_dir / "23b0653.docx",
        submission_dir / "23b0653.docx",
        workspace_root / "BTP_Report_Prabhakar_Yadav_23B0653.docx",
        downloads_dir / "BTP_Report_Prabhakar_Yadav_23B0653.docx",
        submission_dir / "BTP_Report_Prabhakar_Yadav_23B0653.docx"
    ]

    for target in targets:
        try:
            shutil.copy2(temp_out, target)
            print(f"  [OK] Saved to: {target}")
        except PermissionError:
            print(f"  [LOCKED - OPEN IN WORD] {target}")
        except Exception as e:
            print(f"  [ERROR] {target}: {e}")

    # Remove temporary master file
    if temp_out.exists():
        temp_out.unlink()

    elapsed = time.time() - start_time
    file_size_mb = (downloads_dir / "23b0653.docx").stat().st_size / (1024 * 1024)
    num_paras = len(doc.paragraphs)
    num_tables = len(doc.tables)

    print("\n" + "=" * 80)
    print("BTP PROJECT REPORT SUCCESSFULLY GENERATED & DISTRIBUTED!")
    print(f"• Target File Name:       23b0653.docx")
    print(f"• Total Paragraphs:       {num_paras:,}")
    print(f"• Total Formatted Tables: {num_tables}")
    print(f"• Total Embedded Figures: 8 high-resolution figures + 1 IITB crest")
    print(f"• Generated File Size:    {file_size_mb:.2f} MB")
    print(f"• Generation Time:        {elapsed:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
