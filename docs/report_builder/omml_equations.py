"""
scripts/report_builder/omml_equations.py
----------------------------------------
Generates native Word OMML (Office Open XML Math) equations.
Equations are rendered as true Word mathematical objects, centered,
with right-aligned formal equation numbers (e.g. (2.1), (3.1), (4.1)).
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

COLOR_NAVY_PRIMARY = RGBColor(0, 51, 102)
COLOR_MUTED_GRAY = RGBColor(102, 102, 102)

def set_cell_border_none(cell):
    """Removes borders from a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        '<w:top w:val="none"/>'
        '<w:left w:val="none"/>'
        '<w:bottom w:val="none"/>'
        '<w:right w:val="none"/>'
        '</w:tcBorders>'
    )
    tcPr.append(tcBorders)

def add_display_equation(doc, omml_math_xml, eq_num_str):
    """
    Inserts a centered native Word OMML display equation
    with a right-aligned equation number using a clean borderless 2-column layout.
    """
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # A4 text width = 6.27 inches
    table.columns[0].width = Inches(5.45)
    table.columns[1].width = Inches(0.82)

    row = table.rows[0]
    cell_math = row.cells[0]
    cell_num = row.cells[1]

    cell_math.width = Inches(5.45)
    cell_num.width = Inches(0.82)
    cell_math.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell_num.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    set_cell_border_none(cell_math)
    set_cell_border_none(cell_num)

    # Math Paragraph (Centered)
    p_math = cell_math.paragraphs[0]
    p_math.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_math.paragraph_format.space_before = Pt(4)
    p_math.paragraph_format.space_after = Pt(4)
    
    # Append OMML element
    omml_elem = parse_xml(omml_math_xml)
    p_math._p.append(omml_elem)

    # Number Paragraph (Right Aligned)
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(4)
    p_num.paragraph_format.space_after = Pt(4)
    r_num = p_num.add_run(f"({eq_num_str})")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(10.5)
    r_num.font.bold = True
    r_num.font.color.rgb = COLOR_NAVY_PRIMARY

    # Add subtle spacing after equation table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(4)
    return table

# =============================================================================
# PRE-BUILT OMML EQUATION DEFINITIONS
# =============================================================================

# Equation 2.1: Normalized Difference Vegetation Index (NDVI)
OMML_NDVI = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NDVI</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NIR</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> − </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Red</m:t></m:r></m:sub></m:sSub>
    </m:num>
    <m:den>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NIR</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> + </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Red</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> + ε</m:t></m:r>
    </m:den>
  </m:f>
</m:oMath>
</m:oMathPara>"""

# Equation 2.2: Normalized Difference Water Index (NDWI)
OMML_NDWI = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NDWI</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Green</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> − </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NIR</m:t></m:r></m:sub></m:sSub>
    </m:num>
    <m:den>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Green</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> + </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>ρ</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NIR</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t> + ε</m:t></m:r>
    </m:den>
  </m:f>
</m:oMath>
</m:oMathPara>"""

# Equation 2.3: Excess Green Index (ExG)
OMML_EXG = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>ExG</m:t></m:r>
  <m:r><m:t> = 2 × G − R − B</m:t></m:r>
</m:oMath>
</m:oMathPara>"""

# Equation 3.1: Radiometric Percentile Clipping & Mapping
OMML_INORM = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>I</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>norm</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d>
  <m:r><m:t> = round</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
    <m:r><m:t>255 × </m:t></m:r>
    <m:f>
      <m:fPr><m:type m:val="bar"/></m:fPr>
      <m:num>
        <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>clip</m:t></m:r>
        <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
          <m:sSub><m:e><m:r><m:t>I</m:t></m:r></m:e><m:sub><m:r><m:t>raw</m:t></m:r></m:sub></m:sSub>
          <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d>
          <m:r><m:t>, </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>02</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t>, </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>98</m:t></m:r></m:sub></m:sSub>
        </m:e></m:d>
        <m:r><m:t> − </m:t></m:r>
        <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>02</m:t></m:r></m:sub></m:sSub>
      </m:num>
      <m:den>
        <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>98</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t> − </m:t></m:r>
        <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:t>02</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t> + ε</m:t></m:r>
      </m:den>
    </m:f>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 4.1: Dynamic Logit Fusion
OMML_FUSION = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = σ</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:t>α</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>macro</m:t></m:r></m:sub></m:sSub>
  </m:e></m:d>
  <m:r><m:t> · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>SegFormer</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> + </m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:r><m:t>1 − σ</m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
      <m:sSub><m:e><m:r><m:t>α</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>macro</m:t></m:r></m:sub></m:sSub>
    </m:e></m:d>
  </m:e></m:d>
  <m:r><m:t> · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>UNet</m:t></m:r></m:sub></m:sSub>
</m:oMath>
</m:oMathPara>"""

# Equation 4.2: Spatial Gating Mask
OMML_GATING = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>M</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>builtup</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val=""/><m:sepChr m:val=""/></m:dPr><m:e>
    <m:eqArr>
      <m:e><m:r><m:t>1,  if  argmax</m:t></m:r><m:sSub><m:e><m:r><m:t></m:t></m:r></m:e><m:sub><m:r><m:t>c ∈ {0..4}</m:t></m:r></m:sub></m:sSub><m:r><m:t> </m:t></m:r><m:sSub><m:e><m:r><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:t>5c</m:t></m:r></m:sub></m:sSub><m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d><m:r><m:t> = 3</m:t></m:r></m:e>
      <m:e><m:r><m:t>0,  otherwise</m:t></m:r></m:e>
    </m:eqArr>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 4.3: Asphalt-Safe Guard
OMML_ASPHALT_GUARD = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>IsWater</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
    <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NDWI</m:t></m:r><m:r><m:t> > 0.18  ∧  </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>I</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NIR</m:t></m:r></m:sub></m:sSub><m:r><m:t> &lt; 35.0  ∧  BlobArea ≥ 800 px</m:t></m:r>
  </m:e></m:d>
  <m:r><m:t> ∧ </m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>road</m:t></m:r></m:sub></m:sSub><m:r><m:t> &lt; 0.35  ∧  </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>P</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>builtup</m:t></m:r></m:sub></m:sSub><m:r><m:t> &lt; 0.35</m:t></m:r>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 4.4: Dataset 2 Macro Loss (Focal + Lovász)
OMML_LMACRO = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = 0.50 · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Focal</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, </m:t></m:r>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Y</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
  </m:e></m:d>
  <m:r><m:t> + 0.50 · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Lovasz</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, </m:t></m:r>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Y</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>5c</m:t></m:r></m:sub></m:sSub>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 4.5: Dataset 3 Calibrated Focal-Dice Loss (Vashi)
OMML_LVASHI = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Vashi</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = 0.50 · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Focal</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Vashi</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, </m:t></m:r>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Y</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Vashi</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, γ=1.5, </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>w</m:t></m:r>
  </m:e></m:d>
  <m:r><m:t> + 0.50 · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>SoftDice</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Z</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Vashi</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, </m:t></m:r>
    <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>Y</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Vashi</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>, </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="b"/></m:rPr><m:t>w</m:t></m:r>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 6.1: SpaceNet Buffer IoU
OMML_IOU_BUFFER = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>IoU</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>buffer</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>P</m:t></m:r>
    <m:r><m:t>, </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>G</m:t></m:r>
    <m:r><m:t>, </m:t></m:r>
    <m:r><m:t>d</m:t></m:r>
  </m:e></m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num>
      <m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr><m:e>
        <m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>P</m:t></m:r>
        <m:r><m:t> ∩ </m:t></m:r>
        <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>B</m:t></m:r></m:e><m:sub><m:r><m:t>d</m:t></m:r></m:sub></m:sSub>
        <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>G</m:t></m:r></m:e></m:d>
      </m:e></m:d>
    </m:num>
    <m:den>
      <m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr><m:e>
        <m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>P</m:t></m:r>
        <m:r><m:t> ∪ </m:t></m:r>
        <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>B</m:t></m:r></m:e><m:sub><m:r><m:t>d</m:t></m:r></m:sub></m:sSub>
        <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>G</m:t></m:r></m:e></m:d>
      </m:e></m:d>
    </m:den>
  </m:f>
</m:oMath>
</m:oMathPara>"""

# Cross-Entropy and Soft Dice for General Reference
OMML_LCE = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>CE</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = − </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num><m:r><m:t>1</m:t></m:r></m:num>
    <m:den><m:r><m:t>N</m:t></m:r></m:den>
  </m:f>
  <m:nary>
    <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
    <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
    <m:e>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>c=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>C</m:t></m:r></m:sup>
        <m:e>
          <m:sSub><m:e><m:r><m:t>m</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t> · </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t> · log</m:t></m:r>
          <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
            <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          </m:e></m:d>
        </m:e>
      </m:nary>
    </m:e>
  </m:nary>
</m:oMath>
</m:oMathPara>"""

OMML_LDICE = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Dice</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = 1 − </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num>
      <m:r><m:t>2 </m:t></m:r>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
        <m:e>
          <m:sSub><m:e><m:r><m:t>m</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
        </m:e>
      </m:nary>
      <m:r><m:t> + ε</m:t></m:r>
    </m:num>
    <m:den>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
        <m:e>
          <m:sSub><m:e><m:r><m:t>m</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:sSup><m:e><m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
        </m:e>
      </m:nary>
      <m:r><m:t> + </m:t></m:r>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
        <m:e>
          <m:sSub><m:e><m:r><m:t>m</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:sSup><m:e><m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
        </m:e>
      </m:nary>
      <m:r><m:t> + ε</m:t></m:r>
    </m:den>
  </m:f>
</m:oMath>
</m:oMathPara>"""

OMML_LTOTAL = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>total</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = </m:t></m:r>
  <m:sSub><m:e><m:r><m:t>λ</m:t></m:r></m:e><m:sub><m:r><m:t>1</m:t></m:r></m:sub></m:sSub>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>CE</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> + </m:t></m:r>
  <m:sSub><m:e><m:r><m:t>λ</m:t></m:r></m:e><m:sub><m:r><m:t>2</m:t></m:r></m:sub></m:sSub>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Dice</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> + </m:t></m:r>
  <m:sSub><m:e><m:r><m:t>λ</m:t></m:r></m:e><m:sub><m:r><m:t>3</m:t></m:r></m:sub></m:sSub>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>subclass</m:t></m:r></m:sub></m:sSub>
</m:oMath>
</m:oMathPara>"""

# Equation 2.4: Efficient Self-Attention in SegFormer
OMML_ATTENTION = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Attention</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>Q, K, V</m:t></m:r></m:e></m:d>
  <m:r><m:t> = Softmax</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
    <m:f>
      <m:fPr><m:type m:val="bar"/></m:fPr>
      <m:num>
        <m:r><m:t>Q · </m:t></m:r>
        <m:sSup><m:e><m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
          <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>R</m:t></m:r></m:sub></m:sSub>
          <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>K</m:t></m:r></m:e></m:d>
        </m:e></m:d></m:e><m:sup><m:r><m:t>T</m:t></m:r></m:sup></m:sSup>
      </m:num>
      <m:den>
        <m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>
          <m:sSub><m:e><m:r><m:t>d</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>head</m:t></m:r></m:sub></m:sSub>
        </m:e></m:rad>
      </m:den>
    </m:f>
  </m:e></m:d>
  <m:r><m:t> · </m:t></m:r>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>R</m:t></m:r></m:sub></m:sSub>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>V</m:t></m:r></m:e></m:d>
</m:oMath>
</m:oMathPara>"""

# Equation 2.6: Focal Loss
OMML_LFOCAL = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:sSub><m:e><m:r><m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr><m:t>L</m:t></m:r></m:e><m:sub><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>Focal</m:t></m:r></m:sub></m:sSub>
  <m:r><m:t> = − </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num><m:r><m:t>1</m:t></m:r></m:num>
    <m:den><m:r><m:t>N</m:t></m:r></m:den>
  </m:f>
  <m:nary>
    <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
    <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
    <m:sup><m:r><m:t>N</m:t></m:r></m:sup>
    <m:e>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>c=1</m:t></m:r></m:sub>
        <m:sup><m:r><m:t>C</m:t></m:r></m:sup>
        <m:e>
          <m:sSub><m:e><m:r><m:t>α</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
          <m:sSup><m:e><m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
            <m:r><m:t>1 − </m:t></m:r>
            <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          </m:e></m:d></m:e><m:sup><m:r><m:t>γ</m:t></m:r></m:sup></m:sSup>
          <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t> · log</m:t></m:r>
          <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
            <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>i,c</m:t></m:r></m:sub></m:sSub>
          </m:e></m:d>
        </m:e>
      </m:nary>
    </m:e>
  </m:nary>
</m:oMath>
</m:oMathPara>"""

# Equation 7.1: 2D Hann Blending Window
OMML_HANN = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:t>w</m:t></m:r>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:r><m:t>x, y</m:t></m:r></m:e></m:d>
  <m:r><m:t> = </m:t></m:r>
  <m:sSup><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>sin</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:f>
      <m:fPr><m:type m:val="bar"/></m:fPr>
      <m:num><m:r><m:t>π · x</m:t></m:r></m:num>
      <m:den><m:r><m:t>W</m:t></m:r></m:den>
    </m:f>
  </m:e></m:d>
  <m:r><m:t> · </m:t></m:r>
  <m:sSup><m:e><m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>sin</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
  <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
    <m:f>
      <m:fPr><m:type m:val="bar"/></m:fPr>
      <m:num><m:r><m:t>π · y</m:t></m:r></m:num>
      <m:den><m:r><m:t>H</m:t></m:r></m:den>
    </m:f>
  </m:e></m:d>
</m:oMath>
</m:oMathPara>"""
