"""Generate the evaluation handbook and a scanned (image-only) sample PDF.

Run from the project root:  python eval/make_samples.py
"""

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    ("1. Working Hours",
     "Standard working hours are 09:00 to 18:00, Monday to Friday. Core hours, during which everyone "
     "must be reachable, are 10:00 to 16:00. Employees may start between 07:30 and 10:00 as long as "
     "they complete a full working day. The lunch break is 45 minutes and is unpaid. Overtime must be "
     "approved in advance by the line manager and is compensated at 1.5 times the hourly rate or as "
     "time off in lieu within 60 days."),
    ("2. Remote Work",
     "After the probation period, employees may work remotely up to two days per week. Fully remote "
     "arrangements require approval from a vice president. Every remote employee receives a one-time "
     "home office stipend of 400 EUR and an internet reimbursement of 30 EUR per month. Remote staff "
     "must be reachable on Slack during core hours. Working from abroad is allowed for a maximum of "
     "20 days per calendar year and requires a prior tax review by the finance team."),
    ("3. Annual Leave",
     "Every full-time employee receives 25 days of paid annual leave per calendar year. Up to 5 unused "
     "days may be carried over and must be taken by March 31 of the following year; anything beyond "
     "that is forfeited. Leave of more than 3 consecutive days must be requested at least 2 weeks in "
     "advance through the HR portal. Public holidays are not counted against annual leave."),
    ("4. Sick Leave",
     "Sick leave is paid in full for up to 15 days per calendar year. Employees must notify their "
     "manager before 10:00 on the first day of absence. A doctor's note is required after three "
     "consecutive days of illness. Absences longer than 4 weeks are handled through an individual "
     "case plan agreed with HR and the occupational health service."),
    ("5. Parental Leave",
     "Birth mothers are entitled to 16 weeks of fully paid maternity leave. Fathers and partners "
     "receive 4 weeks of paid paternity leave, which must be taken within the first 6 months after "
     "the birth. Adoptive parents have the same entitlements. Returning parents may work part-time at "
     "80 percent for up to 6 months while keeping full benefits."),
    ("6. Business Travel",
     "All business trips are booked through the TravelDesk portal at least 14 days in advance. "
     "Economy class applies to flights shorter than 6 hours; business class is permitted for longer "
     "flights. The hotel cap is 160 EUR per night in major cities and 110 EUR elsewhere. Rental cars "
     "require approval from a director. Train travel in first class is not reimbursed."),
    ("7. Expenses",
     "Expense claims must be submitted within 30 days through the finance portal. Meals are reimbursed "
     "up to 40 EUR per day. A receipt is mandatory for every item above 15 EUR. Alcohol is not "
     "reimbursable. Claims above 500 EUR require the approval of the department head. Approved claims "
     "are paid with the next payroll run."),
    ("8. Learning and Development",
     "Each employee has a training budget of 1,200 EUR per year; unused budget does not carry over. "
     "Conference attendance requires manager approval. Certification fees are paid upfront by the "
     "company and are subject to a 12-month retention clause. Employees also receive 5 paid learning "
     "days per year that can be used for courses, reading or internal workshops."),
    ("9. IT Security",
     "Passwords must have at least 14 characters and multi-factor authentication is mandatory for all "
     "company systems. Laptops are fully encrypted. A lost or stolen device must be reported to the "
     "security team within 2 hours. Company data must never be stored in personal cloud accounts. "
     "Suspicious emails are reported with the phishing button. A VPN is required on public Wi-Fi."),
    ("10. Code of Conduct",
     "Gifts from suppliers or clients worth more than 50 EUR must be declared to the compliance "
     "officer. Potential conflicts of interest are disclosed in writing. Harassment can be reported to "
     "HR or through the anonymous hotline, and retaliation against reporters is prohibited. "
     "Investigations begin within 5 working days of a report."),
    ("11. Performance Reviews",
     "Performance reviews take place twice a year, in June and December, using a rating scale from 1 "
     "to 5. Feedback is collected from at least 3 colleagues. Salary adjustments take effect on April 1. "
     "Promotions are decided once per year. A performance improvement plan lasts 90 days and includes "
     "bi-weekly check-ins with the manager."),
    ("12. Leaving the Company",
     "The notice period is 1 month during probation and 3 months afterwards. Company equipment must be "
     "returned within 5 working days. System access is revoked at 18:00 on the last working day. The "
     "final salary, including payout of unused leave, is paid with the next payroll run. An exit "
     "interview with HR is offered to every departing employee."),
]


def make_handbook(path: Path) -> None:
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page()
        page.insert_text((60, 80), "Acme Corp Employee Handbook", fontsize=10, color=(0.4, 0.4, 0.4))
        page.insert_text((60, 120), title, fontsize=20, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(60, 150, 535, 700), body, fontsize=12, lineheight=1.5)
    doc.save(path)
    print("wrote", path)


def make_scanned(src: Path, dst: Path, pages: int = 3) -> None:
    """Rasterise pages into JPEG-only pages: no text layer, like a real scan."""
    source = pymupdf.open(src)
    out = pymupdf.open()
    for i in range(min(pages, len(source))):
        pix = source[i].get_pixmap(dpi=200)
        page = out.new_page(width=source[i].rect.width, height=source[i].rect.height)
        page.insert_image(page.rect, stream=pix.tobytes("jpeg", jpg_quality=80))
    out.save(dst)
    print("wrote", dst)


TICKET_SECTIONS = [
    ("Rezervasyon detayları", 225, [
        ("Rezervasyon tarihi", "13 Ekim 2026, 14:32"),
        ("Ödeme tarihi", "13 Ekim 2026, 14:35"),
        ("Ödeme yöntemi", "Kredi kartı (**** 4821)"),
        ("Toplam tutar", "2.450,00 TL"),
    ]),
    ("Gidiş uçuşu", 390, [
        ("Uçuş tarihi", "21 Ekim 2026, Çarşamba"),
        ("Kalkış", "İzmir Adnan Menderes (ADB) - 07:15"),
        ("Varış", "Berlin Brandenburg (BER) - 10:05"),
        ("Uçuş numarası", "XQ 142"),
    ]),
    ("Dönüş uçuşu", 555, [
        ("Uçuş tarihi", "28 Ekim 2026, Çarşamba"),
        ("Kalkış", "Berlin Brandenburg (BER) - 18:40"),
        ("Varış", "İzmir Adnan Menderes (ADB) - 22:25"),
        ("Uçuş numarası", "XQ 143"),
    ]),
    ("Yolcu ve bagaj", 715, [
        ("Yolcu", "AYŞE DEMİR"),
        ("Bagaj hakkı", "20 kg"),
        ("Check-in", "Uçuştan 24 saat önce açılır"),
    ]),
]


def _turkish_font() -> str:
    for p in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(p).exists():
            return p
    raise SystemExit("No TrueType font with Turkish characters found")


def make_ticket(path: Path) -> None:
    """A one-page booking confirmation. Labels and values are written as two separate column
    streams (all labels first, then all values), the way many browsers print HTML tables, so a
    naive text dump separates `Ödeme tarihi` from its value. This is the failure we guard against."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="TR", fontfile=_turkish_font())

    def put(x, y, s, size=11, color=(0, 0, 0)):
        page.insert_text((x, y), s, fontsize=size, fontname="TR", color=color)

    put(60, 70, "Örnek Hava Yolları", 12, (0.4, 0.4, 0.4))
    put(60, 110, "Rezervasyonunuz onaylandı!", 22)
    put(60, 140, "Örnek Hava Yolları'nı tercih ettiğiniz için teşekkür ederiz.", 11)
    put(60, 185, "Rezervasyon numarası:", 12)
    put(230, 185, "CAGAEJ", 16)

    for title, y, _ in TICKET_SECTIONS:
        put(60, y, title, 15)
    # column 1: every label, top to bottom
    for _, y, rows in TICKET_SECTIONS:
        for i, (label, _) in enumerate(rows):
            put(60, y + 30 + i * 26, label)
    # column 2: every value, top to bottom
    for _, y, rows in TICKET_SECTIONS:
        for i, (_, value) in enumerate(rows):
            put(300, y + 30 + i * 26, value)
    doc.save(path)
    print("wrote", path)


if __name__ == "__main__":
    handbook = ROOT / "eval" / "handbook_eval.pdf"
    make_handbook(handbook)
    make_scanned(handbook, ROOT / "sample_scanned.pdf")
    make_ticket(ROOT / "eval" / "ticket_tr.pdf")
