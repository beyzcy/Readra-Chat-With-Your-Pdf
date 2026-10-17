"""Generate the held-out documents (eval/heldout/*.pdf).

These documents and their questions (eval/dataset_heldout.json) were written ONCE and are never
used to tune the pipeline: their results are reported as they come out. The older dataset
(eval/dataset.json) is a development set, because it was written while fixing known failures.

Run from the project root:  python eval/make_heldout.py
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent / "heldout"
OUT.mkdir(exist_ok=True)


def put(page, x, y, text, size=11, bold=False, color=(0, 0, 0)):
    page.insert_text((x, y), text, fontsize=size, fontname="hebo" if bold else "helv", color=color)


# ------------------------------------------------------------------ invoice (labels and values in separate streams)
def invoice() -> pymupdf.Document:
    doc = pymupdf.open()
    p = doc.new_page()
    put(p, 56, 70, "Northwind Studio GmbH", 14, True)
    put(p, 56, 86, "Elbstrasse 12, 20457 Hamburg", 10, color=(0.4, 0.4, 0.4))
    put(p, 400, 70, "INVOICE", 22, True)

    meta = [("Invoice number", "INV-2026-0417"), ("Issue date", "03 March 2026"), ("Due date", "02 April 2026"),
            ("Purchase order", "PO-88213"), ("Customer", "Alder & Finch Ltd")]
    for i, (label, _) in enumerate(meta):                   # all labels first ...
        put(p, 56, 140 + i * 22, label, 11, True)
    for i, (_, value) in enumerate(meta):                   # ... then all values
        put(p, 190, 140 + i * 22, value, 11)

    cols = [("Description", 56, ["Brand identity workshop", "Website copywriting", "Photo licensing"]),
            ("Qty", 300, ["2", "4", "1"]), ("Unit price", 360, ["480.00", "150.00", "280.00"]),
            ("Amount", 450, ["960.00", "600.00", "280.00"])]
    p.draw_line((56, 276), (540, 276), width=0.8)
    for name, x, _ in cols:
        put(p, x, 272, name, 11, True)
    for name, x, cells in cols:                              # the table, one column at a time
        for i, cell in enumerate(cells):
            put(p, x, 300 + i * 22, cell, 11)

    tot = [("Subtotal", "1,840.00"), ("VAT 19%", "349.60"), ("Total due", "2,189.60 EUR")]
    for i, (label, _) in enumerate(tot):
        put(p, 340, 400 + i * 24, label, 12, i == 2)
    for i, (_, value) in enumerate(tot):
        put(p, 450, 400 + i * 24, value, 12, i == 2)

    put(p, 56, 500, "Payment", 13, True)
    put(p, 56, 522, "Bank: Commerzbank Hamburg", 11)
    put(p, 56, 540, "IBAN: DE44 5001 0517 5407 3249 31", 11)
    put(p, 56, 558, "Payment reference: INV-2026-0417", 11)
    put(p, 56, 590, "Payment within 30 days of the issue date. Late payments incur interest of 1.5% per month.", 10)
    return doc


# ------------------------------------------------------------------ lease contract (3 pages of clauses)
CLAUSES = [
    [("Residential Lease Agreement", None),
     ("Parties", "Landlord: Maria Keller. Tenant: Jonas Weber. Property: 14 Linden Street, Unit 3B, Hamburg."),
     ("1. Term", "The lease runs for 12 months, starting on 1 September 2026 and ending on 31 August 2027."),
     ("2. Rent", "The monthly rent is 950 EUR, due on the third working day of each month."),
     ("3. Deposit", "The tenant pays a deposit of two months' rent (1,900 EUR). It is returned within 8 weeks after move-out.")],
    [("4. Utilities", "Heating and water are included in the rent. Electricity and internet are paid by the tenant."),
     ("5. Late payment", "If rent is not received by the 5th of the month, a fee of 5 EUR per day applies."),
     ("6. Maintenance", "The tenant pays for minor repairs up to 100 EUR per incident. The landlord pays for everything above that."),
     ("7. Pets", "Pets are allowed only with written consent of the landlord. At most one cat is permitted.")],
    [("8. Subletting", "Subletting is prohibited without the written consent of the landlord."),
     ("9. Notice", "Either party may terminate the lease with 3 months' written notice to the end of a month."),
     ("10. Quiet hours", "Quiet hours are from 22:00 to 06:00 on all days."),
     ("11. Smoking", "Smoking is not allowed inside the unit."),
     ("12. Signatures", "Signed in Hamburg on 15 June 2026 by Maria Keller and Jonas Weber.")],
]


def contract() -> pymupdf.Document:
    doc = pymupdf.open()
    for page_clauses in CLAUSES:
        p = doc.new_page()
        y = 80
        for title, body in page_clauses:
            if body is None:
                put(p, 56, y, title, 20, True)
                y += 44
                continue
            put(p, 56, y, title, 13, True)
            rect = pymupdf.Rect(56, y + 8, 540, y + 90)
            p.insert_textbox(rect, body, fontsize=11.5, fontname="helv", lineheight=1.4)
            y += 96
    return doc


# ------------------------------------------------------------------ two-column research paper
LEFT_1 = ("Abstract. We study crack detection on concrete surfaces when only a small share of images is labelled. "
          "A segmentation network is pre-trained with self-supervision and then fine-tuned on the labelled subset.\n\n"
          "1 Introduction. Visual inspection of bridges is slow and expensive. Labelled data is scarce because "
          "annotating thin cracks pixel by pixel takes experts several minutes per image.")
RIGHT_1 = ("2 Method. The encoder is a ResNet-34. Fine-tuning uses the AdamW optimiser with a learning rate of 3e-4, "
           "a batch size of 32 and 40 epochs. Images are cropped to 512 by 512 pixels.\n\n"
           "3 Dataset. We collected 12,400 images from 31 bridges. Of these, 9,300 are used for training and "
           "3,100 for testing. Only 10 percent of the training images carry labels.")
LEFT_2 = ("4 Results. Our model reaches an F1 score of 87.4 percent on the test set. The supervised baseline "
          "trained on the same labelled images reaches 81.2 percent. Without the augmentation step the F1 score "
          "of our model drops to 84.9 percent.")
RIGHT_2 = ("5 Limitations. The images come from daylight inspections only. Performance on night images and on "
           "wet surfaces was not evaluated.\n\n"
           "6 Conclusion. Self-supervised pre-training closes most of the gap to fully supervised training "
           "while using a tenth of the labels.")


def paper() -> pymupdf.Document:
    doc = pymupdf.open()
    for n, (left, right) in enumerate([(LEFT_1, RIGHT_1), (LEFT_2, RIGHT_2)], start=1):
        p = doc.new_page()
        if n == 1:
            put(p, 56, 70, "Robust Crack Detection with Few Labels", 18, True)
        top = 110 if n == 1 else 70
        p.insert_textbox(pymupdf.Rect(56, top, 285, 760), left, fontsize=10.5, fontname="helv", lineheight=1.35)
        p.insert_textbox(pymupdf.Rect(310, top, 540, 760), right, fontsize=10.5, fontname="helv", lineheight=1.35)
    return doc


# ------------------------------------------------------------------ timetable (many look-alike times, column streams)
TRAINS = [("ICE 705", "08:14", "11:02", "7", "Cologne"), ("IC 2218", "08:48", "10:31", "3", "Bremen"),
          ("RE 1041", "09:05", "09:57", "12", "Luebeck"), ("ICE 583", "09:52", "13:40", "7", "Munich"),
          ("IC 2210", "10:14", "12:09", "4", "Hanover"), ("RE 1043", "10:35", "11:29", "12", "Luebeck"),
          ("ICE 691", "11:08", "14:21", "8", "Berlin")]


def timetable() -> pymupdf.Document:
    doc = pymupdf.open()
    p = doc.new_page()
    put(p, 56, 70, "Departures from Hamburg Hbf", 18, True)
    put(p, 56, 92, "Valid on weekdays, 1 October to 31 March", 10, color=(0.4, 0.4, 0.4))
    names = ["Train", "Departs", "Arrives", "Platform", "Destination"]
    xs = [56, 160, 250, 340, 430]
    for name, x in zip(names, xs):
        put(p, x, 140, name, 11, True)
    p.draw_line((56, 148), (540, 148), width=0.8)
    for c, x in enumerate(xs):                      # one column after the other
        for r, row in enumerate(TRAINS):
            put(p, x, 172 + r * 26, row[c], 11)
    return doc


# ------------------------------------------------------------------ scanned copy of the invoice
def scanned(src: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=170)
        q = out.new_page(width=page.rect.width, height=page.rect.height)
        q.insert_image(q.rect, stream=pix.tobytes("jpeg", jpg_quality=75))
    return out


if __name__ == "__main__":
    inv = invoice()
    for name, doc in [("invoice_en.pdf", inv), ("contract_en.pdf", contract()), ("paper_2col.pdf", paper()),
                      ("timetable_en.pdf", timetable()), ("invoice_scan.pdf", scanned(inv))]:
        doc.save(OUT / name)
        print("wrote", OUT / name, f"({len(doc)} page(s))")
