import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from tkinter import scrolledtext
from pill_camera import CameraPage

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

# def show_page(window, page):
#     return lambda: window.pages[page].tkraise()
def show_page(window, page):
    return lambda: window.show_page(page)
    
COLORS = {
    "background": "#FFFFFF",
    "surface": "#477285",
    "text": "#DDFFF7",
    "button": "#DDFFF7",
    "button_active": "#CCDDD4",
    "button_text": "#80A4ED",
    "border": "#33515F",
}

import tkinter as tk


def apply_theme(widget):
    if isinstance(widget, (tk.Tk, tk.Frame, tk.LabelFrame)):
        widget.configure(bg=COLORS["background"])

    elif isinstance(widget, tk.Label):
        widget.configure(
            bg=COLORS["surface"],
            fg=COLORS["text"],
        )

    elif isinstance(widget, tk.Button):
        widget.configure(
            bg=COLORS["button"],
            fg=COLORS["button_text"],
            activebackground=COLORS["button_active"],
            activeforeground=COLORS["button_text"],
        )

    elif isinstance(widget, (tk.Entry, tk.Text)):
        widget.configure(
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
        )

    elif isinstance(widget, tk.Listbox):
        widget.configure(
            bg=COLORS["surface"],
            fg=COLORS["text"],
            selectbackground=COLORS["button"],
            selectforeground=COLORS["button_text"],
        )

    # Apply the theme to this widget's children too.
    for child in widget.winfo_children():
        apply_theme(child)

drug_options = {
    "Aspirin": {},
    "Acetaminophen": {},
    "Caffeine": {},
    "Ibuprofen": {},
    "Vitamin C": {},
    "Meth": {},
    "Cocaine": {},
    "Fentanyl": {}
}

form_options = []

selected_drug = None
selected_form = None

class DrugSelectionPage(tk.Frame):
    def __init__(self, window, on_continue):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        tk.Label(self, text="Type or select an option:").pack(pady=(12, 4))
        
        self.input_text = tk.StringVar()
        
        self.search_entry = tk.Entry(self, textvariable=self.input_text)
        self.search_entry.pack(fill="x", padx=12)
        
        self.list_frame = tk.Frame(self)
        self.list_frame.pack(fill="x", padx=12, pady=8)

        self.scrollbar = tk.Scrollbar(self.list_frame)
        self.scrollbar.pack(side="right", fill="y")
        
        self.options_list = tk.Listbox(self.list_frame, height=5, yscrollcommand=self.scrollbar.set)
        self.options_list.pack(fill="x", padx=12, pady=8)
        self.scrollbar.config(command=self.options_list.yview)
        
        self.status_label = tk.Label(self, text="No option selected")
        self.status_label.pack()
        
        self.continue_button = tk.Button(self, text="Continue",
                                         command=self.continue_to_next_page,
                                         state="disabled")
        self.continue_button.pack(pady=12)
        
        self.input_text.trace_add("write", self.filter_options)
        self.search_entry.bind("<Return>", self.on_option_typed)
        self.options_list.bind("<<ListboxSelect>>", self.on_option_clicked)
        
        self._setting_selection = False
        
        self.filter_options()
        
    def filter_options(self, *_):
        global drug_options, selected_drug
        
        if self._setting_selection:
            return
                
        selected_drug = None
        self.continue_button.config(state="disabled")
        self.status_label.config(text="No option selected")
        
        query = self.input_text.get().casefold()
        matches = [
            option for option in drug_options.keys()
            if query in option.casefold()
        ]
        
        self.options_list.delete(0, tk.END)
        for option in matches:
            self.options_list.insert(tk.END, option)
    
    def on_option_typed(self, *_):
        global drug_options
        
        typed_text = self.input_text.get().strip().casefold()

        for option in drug_options.keys():
            if option.casefold() == typed_text:
                self.on_option_select(option)
                return "break"

        self.status_label.config(text="Type an exact option or click one in the list.")
        return "break"
    
    def on_option_clicked(self, _event):        
        selection = self.options_list.curselection()
        if selection:
            self.on_option_select(self.options_list.get(selection[0]))

    def on_option_select(self, option):
        global selected_drug
        selected_drug = option

        self._setting_selection = True
        try:
            self.input_text.set(selected_drug)
        finally:
            self._setting_selection = False
        self.status_label.config(text=f"Selected: {selected_drug}")
        self.continue_button.config(state="normal")
        
    def continue_to_next_page(self):
        global selected_drug
        
        if selected_drug is not None:
            self.on_continue()

class FormSelectionPage(tk.Frame):
    def __init__(self, window, on_continue, previous_page):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        self.label = tk.Label(self, text="Select the form of drug:")
        self.label.pack(pady=20)
        
        self.tablet_button = tk.Button(self, text="Tablet", command=self.select_tablet)
        self.tablet_button.pack(pady=10)
        
        self.powder_button = tk.Button(self, text="Powder", command=self.select_powder)
        self.powder_button.pack(pady=10)
        
        self.back_button = tk.Button(self, text="Back", command=show_page(window, previous_page))
        self.back_button.pack(pady=10)

    def select_tablet(self):
        global selected_form
        selected_form = "Tablet"
        
        self.on_continue()

    def select_powder(self):
        global selected_form
        selected_form = "Powder"
        
        self.on_continue()

class TabletImprintInstructionsPage(tk.Frame):
    def __init__(self, window, on_continue, previous_page):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        self.back_button = tk.Button(self, text="Back", command=show_page(window, previous_page))
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Check for a letter-number code on the drug tablet.")
        self.label.pack(pady=10)
                
        self.label2 = tk.Label(self, wraplength=350)
        self.label2.pack(pady=10)
        
        self.continue_button = tk.Button(self, text="Continue", command=self.on_continue)
        self.continue_button.pack(pady=10)
        
    def prep_page(self):
        self.label2.config(text="If it has one, enter the code into drugs.com and verify the drug name matches " + selected_drug)

class SolventSelectionPage(tk.Frame):
    def __init__(self, window, on_continue):
        super().__init__(window.container, bg="#FFFFFF")
        
        global selected_solvent
        
        self.on_continue = on_continue
        
        self.back_button = tk.Button(self, text="Back", command=window.backtrack_solvent)
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Select the available liquid and fill half of a transparent shot glass:")
        self.label.pack(pady=20)
                
        self.water_button = tk.Radiobutton(self, text="Warm Tap Water", variable=selected_solvent, value="Water")
        self.water_button.pack(pady=5)
        
        self.alcohol_button = tk.Radiobutton(self, text="Alcohol (Vodka)", variable=selected_solvent, value="Alcohol")
        self.alcohol_button.pack(pady=5)
        
        self.disclaimer = tk.Label(self, text="Vodka will only assist heroine detection. For the best results, use both alcohol and warm water.", wraplength=400)
        self.disclaimer.pack(pady=10)
        
        self.continue_button = tk.Button(self, text="Continue", command=self.on_continue)
        self.continue_button.pack(pady=10)

class TabletInstructionsPage(tk.Frame):
    def __init__(self, window, on_continue, previous_page):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        self.back_button = tk.Button(self, text="Back", command=show_page(window, previous_page))
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Break the tablet into a half-pinky-fingernail sized piece crush it into a powder.", wraplength=350)
        self.label.pack(pady=10)
        
        self.continue_button = tk.Button(self, text="Continue", command=self.on_continue)
        self.continue_button.pack(pady=10)

class SwirlInstructionsPage(tk.Frame):
    def __init__(self, window, on_continue):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        self.back_button = tk.Button(self, text="Back", command=window.backtrack_swirl)
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Swirl the powder in the liquid for 15 seconds.")
        self.label.pack(pady=10)
        
        self.continue_button = tk.Button(self, text="Continue", command=self.on_continue)
        self.continue_button.pack(pady=10)
        
class ImagePage(tk.Frame):
    def __init__(self, window, on_continue, previous_page):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_continue = on_continue
        
        self.back_button = tk.Button(self, text="Back", command=show_page(window, previous_page))
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Take image:")
        self.label.pack(pady=10)
        
        self.continue_button = tk.Button(self, text="Continue", command=self.on_continue)
        self.continue_button.pack(pady=10)

class SummaryPage(tk.Frame):
    def __init__(self, window, on_continue, previous_page):
        super().__init__(window.container, bg="#FFFFFF")

        tk.Button(self, text="Back", command=show_page(window, previous_page)).pack(pady=10)
        tk.Label(self, text="Session overview:").pack(pady=10)

        self.text = scrolledtext.ScrolledText(self, width=50, height=22,
                                              font=("Consolas", 10), wrap="word")
        self.text.pack(padx=12, pady=8, fill="both", expand=True)
        self.text.config(state="disabled")

        tk.Button(self, text="Continue", command=on_continue).pack(pady=10)

    def set_text(self, content):
        self.text.config(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.text.config(state="disabled")

class ResultsPage(tk.Frame):
    def __init__(self, window, previous_page):
        super().__init__(window.container, bg="#FFFFFF")
        
        self.back_button = tk.Button(self, text="Back", command=show_page(window, previous_page))
        self.back_button.pack(pady=10)
        
        self.label = tk.Label(self, text="Results:")
        self.label.pack(pady=10)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        global selected_solvent
        selected_solvent = tk.StringVar(value="Water")
        
        self.configure(bg="#FFFFFF")
        
        width = 920 / 2
        height = 1600 / 2
        self.geometry(f"{int(width)}x{int(height)}")
        
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        
        self.pages = {
            "drug_selection": DrugSelectionPage(self, on_continue=lambda: self.show_page("form_selection")),
            "form_selection": FormSelectionPage(self, on_continue=self.branch_forms, previous_page="drug_selection"),
            
            "tablet_imprint_instructions": TabletImprintInstructionsPage(self, on_continue=lambda: self.show_page("solvent_selection"), previous_page="form_selection"),
            
            "solvent_selection": SolventSelectionPage(self, on_continue=self.branch_solvent),
            
            "tablet_instructions": TabletInstructionsPage(self, on_continue=lambda: self.show_page("swirl_instructions"), previous_page="solvent_selection"),
            "swirl_instructions": SwirlInstructionsPage(self, on_continue=lambda: self.show_page("image_page")),
            "image_page": ImagePage(self,
                        on_continue=lambda: self.show_page("camera_page"),
                        previous_page="swirl_instructions"),
            "camera_page": CameraPage(self,
                          on_next=self.show_summary,
                          on_back=lambda: self.show_page("image_page")),
            "summary_page": SummaryPage(self,
                            on_continue=lambda: self.show_page("results_page"),
                            previous_page="camera_page"),
            "results_page": ResultsPage(self, previous_page="summary_page"),
        }
        
        for page in self.pages.values():
            page.place(x=0, y=40, relwidth=1, relheight=1)
        
        self.title("Pill Reference Lookup")
        self.geometry(f"{int(width)}x{int(height)}")
        
        self.show_page("drug_selection")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        for p in self.pages.values():
            if hasattr(p, "stop"):
                p.stop()
        self.destroy()
        
    # def show_page(self, page):
    #     self.pages[page].tkraise()
    def show_page(self, name):
        for p in self.pages.values():          # stop any camera that's running
            if hasattr(p, "stop"):
                p.stop()
        self.pages[name].tkraise()
        if hasattr(self.pages[name], "start"):
            self.pages[name].start()

    def show_summary(self, text):
        self.pages["summary_page"].set_text(text)
        self.show_page("summary_page")

        
    def show_form_selection(self):
        self.show_page("form_selection")
        
    def branch_forms(self):
        global selected_form
        
        if (selected_form == "Tablet"):
            self.pages["tablet_imprint_instructions"].prep_page()
            self.show_page("tablet_imprint_instructions")
        
        if (selected_form == "Powder"):
            self.show_page("solvent_selection")
    
    def branch_solvent(self):
        global selected_form
        
        if (selected_form == "Tablet"):
            self.show_page("tablet_instructions")
            
        if (selected_form == "Powder"):
            self.show_page("swirl_instructions")
    
    def backtrack_solvent(self):
        global selected_form
        
        if (selected_form == "Tablet"):
            self.show_page("tablet_imprint_instructions")
        if (selected_form == "Powder"):
            self.show_page("form_selection")
            
    def backtrack_swirl(self):
        global selected_form
        
        if (selected_form == "Tablet"):
            self.show_page("tablet_instructions")
        if (selected_form == "Powder"):
            self.show_page("solvent_selection")

if __name__ == "__main__":
    app = App()
    apply_theme(app)
    app.mainloop()

# tk.Label(window, text="Tablet imprint:").pack(pady=(18, 4))

# imprint_entry = tk.Entry(window, width=30)
# imprint_entry.pack()
# imprint_entry.bind("<Return>", lambda event: search_pills())

# tk.Button(window, text="Search", command=search_pills).pack(pady=10)

# results_box = tk.Text(window, wrap="word", height=12, width=55)
# results_box.pack(padx=12, pady=8, fill="both", expand=True)

# tk.Label(
#     window,
#     text="Possible matches only. Appearance cannot confirm pill contents.",
#     wraplength=440,
# ).pack(pady=(0, 12))

# window.mainloop()