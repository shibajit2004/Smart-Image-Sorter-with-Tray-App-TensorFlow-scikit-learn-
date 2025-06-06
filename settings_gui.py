import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os

SETTINGS_FILE = "settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def main():
    root = tk.Tk()
    root.title("Image Organizer Settings")
    root.geometry("500x300")
    root.resizable(False, False)
    root.iconbitmap("assets\\icon.ico")
    settings = load_settings()

    # --- Create labels and entries ---
    entries = {}

    fields = [
        ("Watch Folder", "WATCH_FOLDER"),
        ("Photo DB Folder", "PHOTO_DB_FOLDER"),
    ]

    for idx, (label_text, key) in enumerate(fields):
        label = tk.Label(root, text=label_text)
        label.grid(row=idx, column=0, padx=10, pady=8, sticky="w")
        
        entry = tk.Entry(root, width=40)
        entry.grid(row=idx, column=1, padx=10, pady=8)
        entry.insert(0, settings.get(key, ""))

        entries[key] = entry

        if "Folder" in label_text:
            def make_browse_function(entry_widget):
                def browse():
                    folder_selected = filedialog.askdirectory()
                    if folder_selected:
                        entry_widget.delete(0, tk.END)
                        entry_widget.insert(0, folder_selected)
                return browse

            browse_button = tk.Button(root, text="Browse", command=make_browse_function(entry))
            browse_button.grid(row=idx, column=2, padx=5)

    # --- Save button ---
    def on_save():
        updated_settings = {key: entry.get() for key, entry in entries.items()}
        changed = updated_settings != settings

        save_settings(updated_settings)

        if changed:
            messagebox.showinfo("Settings", "✅ Settings saved and updated!")
        else:
            messagebox.showinfo("Settings", "ℹ️ Settings saved (no changes).")
        
        root.destroy()  # Close window after saving

    save_button = tk.Button(root, text="Save Settings", command=on_save, bg="green", fg="white")
    save_button.grid(row=len(fields), column=0, columnspan=3, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
