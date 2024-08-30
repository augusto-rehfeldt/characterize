import os
import sys
import subprocess
import threading
import time

python = sys.executable if not "python.exe" in os.listdir(
) else os.path.join(os.path.realpath(os.path.dirname(__file__)), 'python.exe')

try:
    import copy
except ImportError:
    subprocess.run(
        f"{python} -m pip install copy")
    import copy

try:
    import tkinter as tk
    import tkinter.ttk as ttk
except ImportError:
    subprocess.run(
        f"{python} -m pip install tkinter")
    import tkinter as tk
    import tkinter.ttk as ttk
finally:
    import tkinter.filedialog as filedialog
    from tkinter import messagebox


def find_lowest_value(numbers: list) -> int:
    numbers.sort()
    for i in range(1, len(numbers)):
        if numbers[i] > numbers[i-1] + 1:
            return numbers[i-1] + 1
    return numbers[-1] + 1


def get_image_fingerprint(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return hash(f.read())


def to_hours_minutes_seconds(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)


def return_recursive(folder_path: str) -> list:
    files = []
    try:
        for item in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, item)):
                files.append(os.path.join(folder_path, item))
            elif os.path.isdir(os.path.join(folder_path, item)):
                files += return_recursive(os.path.join(folder_path, item))
    except FileNotFoundError:
        pass

    return [x.replace("\\", "/") for x in files]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Characterize")
        self.window_width = 780
        self.window_height = 455
        self.root.geometry(
            f"{int(self.window_width)}x{int(self.window_height)}")
        self.root.resizable(False, False)

        self.time = 0

        self.thread_queue = []
        self.flag_time = False

        # initializing style
        s = ttk.Style()
        s.configure('correct.TEntry', foreground='green')
        s.configure('incorrect.TEntry', foreground='red')

        self.frame_tree = ttk.Frame(
            self.root, style='Frame1.TFrame')
        self.frame_tree.grid(row="0", column="0", columnspan="2", sticky="W",)

        self.frame_buttons = ttk.Frame(
            self.root)
        self.frame_buttons.grid(row="1", column="0", sticky="W",)

        self.frame_bottom = ttk.Frame(self.root)
        self.frame_bottom.grid(row="2", column="0",
                               columnspan="2", sticky="W",)

        self.frame_options = ttk.Frame(self.frame_bottom, width=300)
        self.frame_options.grid(row="0", column="0",
                                pady=(10, 10), padx=(10, 200), sticky="W",)

        self.frame_generate = ttk.Frame(self.frame_bottom, width=200)
        self.frame_generate.grid(
            row="0", column="1", pady=(30, 0), sticky="NW",)

        self.create_widgets()

        self.root.after(
            500, self.listen_for_result)

        self.flag = False

    def create_widgets(self):
        self.place_tree(self.frame_tree)
        self.place_buttons(self.frame_buttons)
        self.place_options(self.frame_options)
        self.place_generate(self.frame_generate)

    def help_1(self):
        message = """- languages with lower range values and higher max values are better for black & white images,\n- languages with higher starting values and lower range are better for colour images.\n- the third item represents the dissimilarity index between characters for a default 12 characters list. A value of 0 indicates that the characters on the list have similar brightness levels, while a value of 1 indicates that the characters on the list have very dissimilar brightness levels. A higher dissimilarity index is better.\n\nascii: Brightness range: 79; min and max: (2, 81); dissimility: 0.087.\narabic: Brightness range: 21; min and max: (8, 29); dissimility: 0.094.\nbraille: Brightness range: 15; min and max: (3, 18); dissimility: 0.005.\nchinese: Brightness range: 107; min and max: (0, 107); dissimility: 0.091.\ncyrillic: Brightness range: 52; min and max: (18, 70); dissimility: 0.092.\nemoji: Brightness range: 142; min and max: (31, 173); dissimility: 0.093.\nhangul: Brightness range: 58; min and max: (28, 86); dissimility: 0.091.\nhiragana: Brightness range: 29; min and max: (17, 46); dissimility: 0.103.\nkatakana: Brightness range: 24; min and max: (15, 39); dissimility: 0.092.\nkanji: Brightness range: 86; min and max: (8, 94); dissimility: 0.092.\nlatin: Brightness range: 48; min and max: (15, 63); dissimility: 0.091.\nnumbers: Brightness range: 22; min and max: (21, 43); dissimility: 0.063.\nnumbers+: Brightness range: 48; min and max: (15, 63); dissimility: 0.085.\nroman: Brightness range: 54; min and max: (27, 81); dissimility: 0.103.\nsimple: Brightness range: 79; min and max: (2, 81); dissimility: 0.046."""
        messagebox.showinfo(message=message, title="Script help")

    def OnDoubleClick(self, event):
        try:
            item = self.tree.selection()[0]
        except IndexError:
            pass
        else:
            path_to_open = self.tree.item(item)["values"][-1]
            if path_to_open:
                os.startfile(path_to_open.replace(
                    "\\\\", "/").strip())

    def place_tree(self, master: ttk.Frame):
        try:
            for child in master.get_children():
                child.remove()
        except:
            pass
        self.tree = ttk.Treeview(master, columns=(
            "name", "format", "size", "final size", "status", "output path"), height=10, show='headings',)
        # Set the column headings
        self.tree.heading("#1", text="Name")
        self.tree.heading("#2", text="Format")
        self.tree.heading("#3", text="Size")
        self.tree.heading("#4", text="Final size")
        self.tree.heading("#5", text="Status")
        self.tree.heading("#6", text="Output path")
        # Set the column widths
        self.tree.column("#1", width=int(self.window_width/2.5-8),
                         stretch=tk.NO, anchor="w")
        self.tree.column("#2", width=int(self.window_width/8-7),
                         stretch=tk.NO, anchor="e")
        self.tree.column("#3", width=int(self.window_width/8-7),
                         stretch=tk.NO, anchor="e")
        self.tree.column("#4", width=int(self.window_width/8-7),
                         stretch=tk.NO, anchor="e")
        self.tree.column("#5", width=int(self.window_width/4-8),
                         stretch=tk.NO, anchor="e")
        self.tree.column("#6", width=0,
                         stretch=tk.NO, anchor="e")

        self.tree.bind("<Double-1>", self.OnDoubleClick)

        vscroll = tk.Scrollbar(
            master, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vscroll.set)
        # Pack the treeview and scrollbar
        vscroll.pack(side="right", fill="y", anchor="ne")
        self.tree.pack(side="left", fill="both")

    def place_buttons(self, master: ttk.Frame):
        try:
            for child in master.get_children():
                child.remove()
        except:
            pass

        self.button_add = ttk.Button(master,
                                     text="Add file/s", command=self.file_add_dialog)
        self.button_add.grid(row="0", column="0")
        self.button_add_folder = ttk.Button(master,
                                            text="Add folder", command=self.folder_add_dialog)
        self.button_add_folder.grid(row="0", column="1")
        self.button_remove = ttk.Button(
            master, text="Remove selected", command=self.remove_files)
        self.button_remove.grid(row="0", column="2")
        self.button_clear = ttk.Button(
            master, text="Clear all", command=self.clear_tree)
        self.button_clear.grid(row="0", column="3")
        self.label_length = ttk.Label(master, text="0 file/s in total")
        self.label_length.grid(row="0", column="4")
        self.label_time = ttk.Label(master, text="")
        self.label_time.grid(row="0", column="5")

    def place_options(self, master: ttk.Frame):
        try:
            for child in master.get_children():
                child.remove()
        except:
            pass
        self.label_language = ttk.Label(master, text="Script")
        self.label_language.grid(row="0", column="0", sticky="W",)
        self.language_cb = ttk.Combobox(master, state="readonly", width=12, values=[
            "ascii", "arabic", "braille", "chinese", "cyrillic", "emoji", "hangul", "hiragana", "katakana", "kanji", "latin", "numbers", "numbers+", "roman", "simple"])
        self.language_cb.set(self.language_cb["values"][0])
        self.language_cb.grid(row="0", column="1", pady=(0, 5), ipady=2)
        
        # Add the new checkbox for empty character
        self.empty_char_check = ttk.Checkbutton(master, text="Add empty character")
        self.empty_char_check.state(['!alternate'])
        self.empty_char_check.grid(row="0", column="2", sticky="W")
        
        # Move the help button to the next column
        self.language_help = ttk.Button(
            master, text="Help", width=6, command=self.help_1)
        self.language_help.grid(row="0", column="3", sticky="NW")

        self.label_resolution = ttk.Label(master, text="Resolution")
        self.label_resolution.grid(row="1", column="0", sticky="W",)
        self.resolution_entry = ttk.Entry(master, width=15,)
        self.resolution_entry.bind(
            '<KeyRelease>', lambda e: self.check_value(e.widget, 1, 4000))
        self.resolution_entry.grid(row="1", column="1", pady=(0, 5), ipady=2)
        self.label_resolution_bis = ttk.Label(master, text="characters wide")
        self.label_resolution_bis.grid(row="1", column="2", sticky="W",)

        self.label_complexity = ttk.Label(master, text="Complexity")
        self.label_complexity.grid(row="2", column="0", sticky="W",)
        self.complexity_entry = ttk.Entry(master, width=15,)
        self.complexity_entry.bind(
            '<KeyRelease>', lambda e: self.check_value(e.widget, 1, 40))
        self.complexity_entry.grid(row="2", column="1", pady=(0, 5), ipady=2)
        self.label_complexity_bis = ttk.Label(
            master, text="different characters")
        self.label_complexity_bis.grid(row="2", column="2", sticky="W",)

        self.label_format = ttk.Label(master, text="Output format/s")
        self.label_format.grid(row="3", column="0", padx=(0, 5), sticky="W",)
        self.format_cb = ttk.Combobox(master, state="readonly", width=12, values=[
            "png", "jpg", "txt", "png, jpg", "png, txt", "jpg, txt", "png, jpg, txt"])
        self.format_cb.set(self.format_cb["values"][0])
        self.format_cb.grid(row="3", column="1", pady=(0, 4), ipady=2)

        self.label_color = ttk.Label(master, text="Color")
        self.label_color.grid(row="4", column="0", sticky="W",)
        self.color_check = ttk.Checkbutton(master)
        self.color_check.state(['!alternate'])
        self.color_check.grid(row="4", column="1")

        self.label_divide = ttk.Label(master, text="Subdivide")
        self.label_divide.grid(row="5", column="0", sticky="W",)
        self.divide_check = ttk.Checkbutton(master)
        self.divide_check.state(['!alternate'])
        self.divide_check.grid(row="5", column="1")

        self.label_optimize = ttk.Label(master, text="Optimize")
        self.label_optimize.grid(row="6", column="0", sticky="W",)
        self.optimize_check = ttk.Checkbutton(master)
        self.optimize_check.state(['!alternate'])
        self.optimize_check.grid(row="6", column="1")

        # Add tooltips for better user experience
        self.add_tooltip(self.language_cb, "Select the character set to use")
        self.add_tooltip(self.empty_char_check, "Use an empty character for the darkest pixels")
        self.add_tooltip(self.resolution_entry, "Set the width of the output in characters")
        self.add_tooltip(self.complexity_entry, "Set the number of unique characters to use")
        self.add_tooltip(self.format_cb, "Choose the output file format(s)")
        self.add_tooltip(self.color_check, "Enable color output")
        self.add_tooltip(self.divide_check, "Subdivide large images for processing")
        self.add_tooltip(self.optimize_check, "Optimize output files (if <=300 files)")

    def add_tooltip(self, widget, text):
        tooltip = tk.Label(widget.master, text=text, background="#ffffe0", relief="solid", borderwidth=1)
        tooltip.pack_forget()

        def enter(event):
            tooltip.lift(widget)
            tooltip.place(x=widget.winfo_rootx() - widget.winfo_x(), 
                          y=widget.winfo_rooty() - widget.winfo_y() + widget.winfo_height())

        def leave(event):
            tooltip.place_forget()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def place_generate(self, master: ttk.Frame):
        try:
            for child in master.get_children():
                child.remove()
        except:
            pass
        self.button_generate = ttk.Button(
            master, text="Generate!", width="20", command=self.process)
        self.button_generate.pack(anchor="nw")
        self.label_generate = ttk.Label(
            master, text="", wraplength=130, justify=tk.LEFT)
        self.label_generate.pack(anchor="nw")

    def cycle_buttons(self, disable: bool = False):
        if disable:
            self.button_add.state(["disabled"])
            self.button_add_folder.state(["disabled"])
            self.button_generate.state(["disabled"])
            self.button_clear.state(["disabled"])
            self.button_remove.state(["disabled"])
        else:
            self.button_add.state(["!disabled"])
            self.button_add_folder.state(["!disabled"])
            self.button_generate.state(["!disabled"])
            self.button_clear.state(["!disabled"])
            self.button_remove.state(["!disabled"])

    def file_add_dialog(self):
        self.cycle_buttons(True)

        filetypes = [
            ("image", ".jpeg"),
            ("image", ".png"),
            ("image", ".jpg"),
            ("image", ".jfif"),
            ("image", ".webp")
        ]

        tree_filenames = [self.tree.item(line)['values'][0]
                        for line in self.tree.get_children()]

        filenames = sorted([x for x in filedialog.askopenfilenames(
            title='Open a file', initialdir='/', filetypes=filetypes) if not x in tree_filenames])

        fingerprints = [get_image_fingerprint(x) for x in tree_filenames]

        for file in filenames:
            fingerprint = get_image_fingerprint(file)
            if fingerprint in fingerprints:
                continue
            fingerprints.append(fingerprint)
            file_stats = os.stat(file)
            indexes = [int(self.tree.item(line)['text'])
                    for line in self.tree.get_children()]
            index = find_lowest_value(indexes) if len(indexes) > 0 else 0
            index = str(index)
            self.tree.insert("", "end", text=index, values=(file, file.split(
                ".")[-1], f"{round(file_stats.st_size / 1024, 2)} KB", "", "", ""))

        self.label_length["text"] = f"{len(self.tree.get_children())} file/s in total"
        self.root.title("Characterize")
        self.cycle_buttons(False)

    def folder_add_dialog(self):
        self.cycle_buttons(True)

        tree_filenames = [self.tree.item(line)['values'][0]
                        for line in self.tree.get_children()]

        folder = filedialog.askdirectory(
            initialdir='/', title='Select a folder',)

        files = return_recursive(folder)

        filenames = sorted([x for x in files if not x in tree_filenames and any(
            x.lower().endswith(y) for y in [".png", ".jpg", ".jfif", ".jpeg", ".webp"])])

        fingerprints = [get_image_fingerprint(x) for x in tree_filenames]

        for file in filenames:
            fingerprint = get_image_fingerprint(file)
            if fingerprint in fingerprints:
                continue
            fingerprints.append(fingerprint)
            file_stats = os.stat(file)
            indexes = [int(self.tree.item(line)['text'])
                    for line in self.tree.get_children()]
            index = find_lowest_value(indexes) if len(indexes) > 0 else 0
            index = str(index)
            self.tree.insert("", "end", text=index, values=(file, file.split(
                ".")[-1], f"{round(file_stats.st_size / 1024, 2)} KB", "", folder, ""))  # Store folder path in the output path column

        self.label_length["text"] = f"{len(self.tree.get_children())} file/s in total"
        self.root.title("Characterize")
        self.cycle_buttons(False)

    def remove_files(self):
        self.cycle_buttons(True)
        selected = self.tree.selection()
        for item in selected:
            self.tree.delete(item)

        self.label_length["text"] = f"{len(self.tree.get_children())} file/s in total"
        self.root.title("Characterize")
        self.cycle_buttons(False)

    def clear_tree(self):
        self.cycle_buttons(True)
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.label_length["text"] = f"{len(self.tree.get_children())} file/s in total"
        self.root.title("Characterize")
        self.cycle_buttons(False)

    def check_value(self, entry: ttk.Entry, min_value: int, max_value: int) -> bool:
        try:
            value = float(entry.get().strip())
            valid = min_value <= value <= max_value
        except ValueError:
            valid = False
        entry.config(style='correct.TEntry' if valid else 'incorrect.TEntry')
        return valid

    def listen_for_result(self):
        """ Check if there is something in the queue. """
        queue = copy.deepcopy(self.thread_queue)
        self.thread_queue = []
        if len(queue) > 0:
            queue = sorted(queue, key=lambda x: len(x))
            if self.flag:
                self.label_generate["text"] = "Processing images..."
                self.flag = False
            self.update_tw(queue)
        if self.flag_time:
            self.label_time[
                "text"] = f"/ Time elapsed: {to_hours_minutes_seconds(time.time()-self.time)}"
        self.root.after(300, self.listen_for_result)

    def update_tw(self, queue: list = False):
        for res in queue:
            if isinstance(res, tuple):
                for line in self.tree.get_children():
                    if self.tree.item(line)['values'][0] == res[0]:
                        if len(res) == 2:
                            self.tree.item(line, values=self.tree.item(line)[
                                'values'][:-3]+["", f"Processing...", ""])
                        elif len(res) > 2:
                            self.tree.item(line, values=self.tree.item(line)[
                                'values'][:-3]+["", f"Completed in {res[1]} s", res[2]])
        children = self.tree.get_children()
        pct = round((len([x for x in children if "Completed" in self.tree.item(x)[
                    'values'][-2]])/len(children))*100, 2)
        self.root.title(f"Characterize ({pct}% processed)")
        if "enable" in queue:
            self.cycle_buttons(False)
            self.label_time[
                "text"] = f"/ Time elapsed: {to_hours_minutes_seconds(time.time()-self.time)}"
            self.label_generate["text"] = "Updating final file sizes..."
            # update sizes
            for line in self.tree.get_children():
                if self.tree.item(line)['values'][-1] != "":
                    self.tree.item(line, values=self.tree.item(line)[
                        'values'][:-3]+[f"{round(os.stat(self.tree.item(line)['values'][-1]).st_size / 1024, 2)} KB"]+self.tree.item(line)['values'][-2:])
            self.label_generate["text"] = "Done. Double click on any file on the list to open it."
            self.flag_time = False

    def process(self):
        def run_thread(command, callback):
            # Configure startupinfo to hide the subprocess window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # Execute the command in a subprocess
            with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=startupinfo) as process:
                for line in iter(process.stdout.readline, b""):
                    if not line:
                        break
                    line = line.decode('utf-8').strip()
                    for inf in line.split(">>"):
                        segments = inf.split("<<")
                        if len(segments) == 3:
                            self.thread_queue.append((segments[1], segments[2].replace("\\r\\n", "")))
                        elif len(segments) == 4:
                            self.thread_queue.append((segments[1], segments[2], segments[3].replace("\\\\", "/").replace("\\r\\n", "")))
                process.wait()
            
            callback()
        
        def split_list(lst, chunk_size):
            """Split a list into chunks of a given size."""
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        if not all(self.resolution_entry.get() and self.complexity_entry.get()):
            return False

        try:
            value_a = int(self.resolution_entry.get())
            value_b = int(self.complexity_entry.get())
        except ValueError:
            return False
        else:
            if not 1 <= value_a <= 4000:
                return False
            if not 1 <= value_b <= 40:
                return False

        # Get folder paths if all files in the folder are added
        folder_paths = set()
        individual_files = []

        for line in self.tree.get_children():
            output_path = self.tree.item(line)['values'][-1]
            if output_path:
                if os.path.isdir(output_path):
                    # Check if all files in the directory are added
                    if len([x for x in self.tree.get_children() if self.tree.item(x)['values'][-1] == output_path]) == len(return_recursive(output_path)):
                        folder_paths.add(output_path)
                    else:
                        individual_files.append(self.tree.item(line)['values'][0])
                else:
                    # If it's a file, add it to individual_files
                    individual_files.append(self.tree.item(line)['values'][0])
            else:
                individual_files.append(self.tree.item(line)['values'][0])

        base_command = f"""{python} {os.path.join(os.path.realpath(os.path.dirname(__file__)), 'characterize.py')}"""
        
        # Adding folders
        for folder in folder_paths:
            base_command += f' --i "{folder}"'

        script = self.language_cb.get().strip()
        color = True if len(self.color_check.state()) > 0 else False
        divide = True if len(self.divide_check.state()) > 0 else False
        optimize = True if len(self.optimize_check.state()) > 0 else False
        empty_char = True if len(self.empty_char_check.state()) > 0 else False
        format = self.format_cb.get().strip()

        base_command += f' --cr {value_a} --cl {value_b} --l {script} --c {color} --d {divide} --o {optimize} --f {format} --ec {empty_char} --tk true'

        self.cycle_buttons(True)
        self.label_generate["text"] = "Running engine..."
        self.flag = True
        self.root.title("Characterize")
        self.time = time.time()
        self.label_time["text"] = ""
        self.flag_time = True

        for line in self.tree.get_children():
            self.tree.item(line, values=self.tree.item(line)['values'][:-3] + ["", "", ""])

        # Split individual files into chunks of 50
        file_chunks = list(split_list(individual_files, 50))

        # Track number of processed chunks
        processed_chunks = [0]

        def process_next_chunk(index):
            if index < len(file_chunks):
                files_str = ' '.join([f'"{file}"' for file in file_chunks[index]])
                command = f'{base_command} --i {files_str}'
                t = threading.Thread(target=run_thread, args=(command, lambda: on_chunk_complete(index + 1)), daemon=True)
                t.start()
            else:
                # Re-enable buttons or perform other actions after all chunks are processed
                self.cycle_buttons(False)
                self.label_generate["text"] = f"Completed all {len(file_chunks)} chunks"

        def on_chunk_complete(index):
            processed_chunks[0] += 1
            self.label_generate["text"] = f"Processing... {processed_chunks[0]} out of {len(file_chunks)} chunks completed"
            if processed_chunks[0] >= len(file_chunks):
                self.cycle_buttons(False)
                self.label_generate["text"] = f"Completed all {len(file_chunks)} chunks"
                self.thread_queue.append("enable")
            else:
                process_next_chunk(index)

        process_next_chunk(0)



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
