from pdf2image import convert_from_path
import pytesseract
import os

def process_pdf(pdf_path):
    # Convierte el PDF a imágenes
    pages = convert_from_path(pdf_path, 300)
    extracted_text = ""

    for page_num, page in enumerate(pages):
        # Usar Tesseract para extraer texto de la imagen
        text = pytesseract.image_to_string(page, lang='spa')
        extracted_text += text

    # Guardar el texto extraído en un archivo
    text_output_path = pdf_path.replace(".pdf", "_extracted.txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Texto extraído guardado en: {text_output_path}")  # Imprimir el path
    return text_output_path
