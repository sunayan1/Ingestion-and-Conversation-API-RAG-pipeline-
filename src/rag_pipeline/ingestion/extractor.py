from pypdf import PdfReader
from io import BytesIO

def extract_text_from_pdf(filebytes: bytes) -> str: 
    pdf_file = BytesIO(filebytes) #creates in-memory file-like object
    reader = PdfReader(pdf_file)  # creates a PDF reader object that parses the PDF structure 
    extracted_text = ""

    for pages in reader.pages:
        text = pages.extract_text()
        if text: 
            extracted_text += text + "\n"
    return extracted_text


def extract_text_from_txt(filebytes: bytes) -> str: 
    try: 
        return filebytes.decode('utf-8')
    except UnicodeDecodeError: 
        return filebytes.decode('latin-1')


def extract_text(filename: str, filebytes: bytes): 
    if filename.endswith('.pdf'): 
        return extract_text_from_pdf(filebytes)
    elif filename.endswith('.txt'): 
        return extract_text_from_txt(filebytes)
    else: 
        raise ValueError(f"Unsupported file type: {filename}")