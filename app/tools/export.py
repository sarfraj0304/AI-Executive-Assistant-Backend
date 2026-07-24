import os
import uuid
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openpyxl import Workbook

load_dotenv()

EXPORT_DIR = "exports"
BASE_URL = os.getenv("BASE_URL")

os.makedirs(EXPORT_DIR, exist_ok=True)


def create_pdf(content: str, file_name: str):

    unique_name = f"{uuid.uuid4().hex}_{file_name}.pdf"

    file_path = os.path.join(
        EXPORT_DIR,
        unique_name,
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
    )

    styles = getSampleStyleSheet()

    story = []

    for line in content.split("\n"):

        story.append(
            Paragraph(
                line or " ",
                styles["BodyText"],
            )
        )

        story.append(Spacer(1, 8))

    doc.build(story)

    return {
        "success": True,
        "file_name": unique_name,
        "file_url": f"{BASE_URL}/exports/{unique_name}",
    }


def create_excel(data: list[dict], file_name: str):

    unique_name = f"{uuid.uuid4().hex}_{file_name}.xlsx"

    file_path = os.path.join(
        EXPORT_DIR,
        unique_name,
    )

    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Export"

    if not data:
        workbook.save(file_path)

        return {
            "success": True,
            "file_name": unique_name,
            "file_url": f"{BASE_URL}/exports/{unique_name}",
        }

    # Headers
    headers = list(data[0].keys())

    sheet.append(headers)

    # Rows
    for item in data:

        sheet.append([item.get(header, "") for header in headers])

    workbook.save(file_path)

    return {
        "success": True,
        "file_name": unique_name,
        "file_url": f"{BASE_URL}/exports/{unique_name}",
    }
