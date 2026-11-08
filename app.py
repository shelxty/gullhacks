import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

DATA_FILE = Path(__file__).with_name("pills.csv")


def search_pills():
    imprint = imprint_entry.get().strip().casefold()
    if not imprint:
        messagebox.showinfo("Enter an imprint", "Type the text printed on the pill.")
        return

    results_box.delete("1.0", tk.END)

    if not DATA_FILE.exists():
        results_box.insert(tk.END, "Reference file pills.csv was not found.")
        return

    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        matches = [
            row for row in csv.DictReader(file)
            if row.get("imprint", "").strip().casefold() == imprint
        ]

    if not matches:
        results_box.insert(tk.END, "No reference matches found.")
        return

    for row in matches:
        results_box.insert(
            tk.END,
            f"Possible match: {row.get('drug_name', 'Unknown')}\n"
        )


window = tk.Tk()
window.title("Pill Reference Lookup")
window.geometry("480x360")

tk.Label(window, text="Tablet imprint:").pack(pady=(18, 4))

imprint_entry = tk.Entry(window, width=30)
imprint_entry.pack()
imprint_entry.bind("<Return>", lambda event: search_pills())

tk.Button(window, text="Search", command=search_pills).pack(pady=10)

results_box = tk.Text(window, wrap="word", height=12, width=55)
results_box.pack(padx=12, pady=8, fill="both", expand=True)

tk.Label(
    window,
    text="Possible matches only. Appearance cannot confirm pill contents.",
    wraplength=440,
).pack(pady=(0, 12))

window.mainloop()