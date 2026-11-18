import docx
from docx.oxml import parse_xml

doc = docx.Document()
p = doc.add_paragraph()

omml = """<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
<m:oMath>
  <m:r><m:rPr><m:scr m:val="roman"/><m:sty m:val="p"/></m:rPr><m:t>NDVI</m:t></m:r>
  <m:r><m:t> = </m:t></m:r>
  <m:f>
    <m:fPr><m:type m:val="bar"/></m:fPr>
    <m:num>
      <m:r><m:t>NIR - Red</m:t></m:r>
    </m:num>
    <m:den>
      <m:r><m:t>NIR + Red</m:t></m:r>
    </m:den>
  </m:f>
</m:oMath>
</m:oMathPara>"""

p._p.append(parse_xml(omml))
doc.save('test_math.docx')
print('OMML Math saved successfully!')

