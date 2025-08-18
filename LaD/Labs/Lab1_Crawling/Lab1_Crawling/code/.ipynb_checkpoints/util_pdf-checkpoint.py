import pdfplumber

def convert_pdf_to_txt(pdffile):
    '''Convert PDF content from a file path to text.'''
    extracted_text = ""
    with pdfplumber.open(pdffile) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            print('page', page_num)
            extracted_text += page.extract_text() or ""  # Some pages might have no text, hence the or ""
    return extracted_text