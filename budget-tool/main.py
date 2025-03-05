import os
import camelot

current_dir = "./budget-tool"
pdf_name = "6-Month statement.pdf"
pdf_file = os.path.join(current_dir, pdf_name)

df = camelot.read_pdf(pdf_file, pages="all")
pass
