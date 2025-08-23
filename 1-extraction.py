from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
import sys
import io

# Set stdout to handle UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

#result = converter.convert("https://arxiv.org/pdf/2408.09869")JSKS_Technical Proposal.pdf
result = converter.convert("a.pdf")

document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

# Save to file instead of printing
with open("output_pdf.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)
print("PDF extraction complete. Output saved to output_pdf.md")

