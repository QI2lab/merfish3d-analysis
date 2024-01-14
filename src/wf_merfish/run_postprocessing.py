from magicgui.widgets import Container, FileEdit, CheckBox, ComboBox, ProgressBar, PushButton, Label
from pathlib import Path
from wf_merfish.postprocess.postprocess import postprocess

def create_gui():
    # Main GUI container
    layout = Container(layout='vertical')

    # File selection widgets
    directory_selector = FileEdit(mode='d', label='Dataset')
    codebook_selector = FileEdit(mode='r', label='Codebook', filter='*.csv', visible=False)
    bit_order_selector = FileEdit(mode='r', label='Bit Order', filter='*.csv', visible=False)
    noise_map_selector = FileEdit(mode='r', label='Noise Map', filter='*.tiff', visible=False)
    darkfield_selector = FileEdit(mode='r', label='Darkfield', filter='*.tiff', visible=False)
    shading_selector = FileEdit(mode='r', label='Shading', filter='*.tiff', visible=False)

    # Correction options dropdown
    correction_options = ComboBox(label="Correction Options", choices=["None", "Hotpixel correct", "Flatfield correct"], visible=False)

    # Other specific options (initially hidden)
    other_options = [CheckBox(text='Register polyDT each tile across rounds', visible=False),
                     CheckBox(text='Register polyDT all tiles first round', visible=False)]

    # Summary label (initially hidden)
    summary_label = Label(value="", visible=False)

    # Default progress bars and labels
    progress_labels = [Label(value="Round", visible=False),
                       Label(value="Tile", visible=False),
                       Label(value="Channel", visible=False)]
    progress_bars = [ProgressBar(value=0, max=100, visible=False) for _ in range(3)]

    # Function to handle directory selection
    def handle_directory_change(event):
        if directory_selector.value:
            directory_path = Path(directory_selector.value)
            csv_files = list(directory_path.glob('*.csv'))
            if csv_files:
                codebook_selector.visible = True
                summary_label.visible = False
            else:
                summary_label.value = "No CSV file found in the directory."
                summary_label.visible = True

    # Function to handle codebook file selection
    def handle_codebook_change(event):
        if codebook_selector.value:
            bit_order_selector.visible = True
            summary_label.visible = False

    # Function to handle bit order file selection
    def handle_bit_order_change(event):
        if bit_order_selector.value:
            correction_options.visible = True
            correction_options.value = "Hotpixel correct"
            correction_options.value = "None"
            summary_label.visible = False

    # Function to handle correction option change
    def handle_correction_option_change(event):
        correction_choice = correction_options.value
        noise_map_selector.visible = correction_choice in ["Hotpixel correct", "Flatfield correct"]
        darkfield_selector.visible = correction_choice == "Flatfield correct"
        shading_selector.visible = correction_choice == "Flatfield correct"
        for option in other_options:
            option.visible = True
        start_button.visible = correction_choice == "None" or check_all_files_valid()

    # Function to validate TIFF file
    def validate_tiff_file(file_path):
        return Path(file_path).is_file() if file_path else False

    # Function to check if all required files are selected and valid
    def check_all_files_valid():
        if correction_options.value in ["Hotpixel correct", "Flatfield correct"]:
            if not validate_tiff_file(noise_map_selector.value):
                return False
        if correction_options.value == "Flatfield correct":
            if not validate_tiff_file(darkfield_selector.value) or not validate_tiff_file(shading_selector.value):
                return False
        return True
        
    # # Worker function with nested loops
    # @thread_worker
    # def nested_task(selected_options, dataset_path, codebook_path, bit_order_path, noise_map_path=None, darkfield_path=None, shading_path=None):
    #     # Implement task logic here using the provided arguments
    #     pass

    # Function to handle progress updates
    def update_progress(value):
        # Implement progress update logic here
        pass

    # Start button (initially visible)
    start_button = PushButton(text="Start Task", visible=False)

    # Exit button
    exit_button = PushButton(text="Exit")

    # Function to close the GUI
    def close_gui():
        layout.native.close()

    # Start the task
    def start_task():
        selected_options = {option.text: option.value for option in other_options if option.value}
        
        # Clear existing progress bars and labels
        for widget in layout:
            if isinstance(widget, ProgressBar) or isinstance(widget, Label):
                layout.remove(widget)

        # Add default progress bars and labels
        for label, bar in zip(progress_labels, progress_bars):
            label.visible = True
            bar.visible = True
            layout.extend([label, bar])

        # Add additional progress bars for specific options
        if selected_options.get('Register polyDT each tile across rounds'):
            layout.extend([Label(value="PolyDT Tile", visible=True), 
                        ProgressBar(value=0, max=100, visible=True)])
        if selected_options.get('Register polyDT all tiles first round'):
            layout.extend([Label(value="PolyDT Round", visible=True), 
                        ProgressBar(value=0, max=100, visible=True)])

        noise_map_path = noise_map_selector.value if validate_tiff_file(noise_map_selector.value) else None
        darkfield_path = darkfield_selector.value if validate_tiff_file(darkfield_selector.value) else None
        shading_path = shading_selector.value if validate_tiff_file(shading_selector.value) else None

        # Start the worker with additional arguments
        worker = postprocess(selected_options, directory_selector.value, 
                             codebook_selector.value, bit_order_selector.value,
                             noise_map_path, darkfield_path, shading_path)
        worker.yielded.connect(update_progress)
        worker.start()

    def update_start_button_visibility():
        start_button.visible = correction_options.value == "None" or check_all_files_valid()

    correction_options.changed.connect(update_start_button_visibility)

    start_button.clicked.connect(start_task)
    exit_button.clicked.connect(close_gui)

    # Connect the file selection events to their handlers
    directory_selector.changed.connect(handle_directory_change)
    codebook_selector.changed.connect(handle_codebook_change)
    bit_order_selector.changed.connect(handle_bit_order_change)
    correction_options.changed.connect(handle_correction_option_change)

    # Add widgets to the layout
    layout.extend([directory_selector, codebook_selector, bit_order_selector, 
                correction_options, noise_map_selector, darkfield_selector, 
                shading_selector, summary_label] + other_options + [start_button, exit_button])

    # Show the GUI with the specified window title
    layout.native.setWindowTitle("Postprocess qi2lab widefield MERFISH")
    layout.show(run=True)

create_gui()
