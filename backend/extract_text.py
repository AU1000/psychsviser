import pdfplumber

def extract_text_from_pdf(pdf_path):

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                full_text += text + "\n"

    return full_text


if __name__ == "__main__":

    text = extract_text_from_pdf("C:/Users/Admin/Downloads/the-laws-of-human-nature.pdf")

    with open("processed/book_text.txt","w",encoding="utf-8") as f:
        f.write(text)

    print("Text extraction complete")