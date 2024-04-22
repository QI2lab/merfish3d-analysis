from magicgui.widgets import Container, FileEdit, CheckBox, ComboBox, ProgressBar, PushButton, Label
from pathlib import Path
from wf_merfish.postprocess.postprocess import postprocess
from superqt.utils import ensure_main_thread

def create_gui():
    # Main GUI container
    layout = Container(layout='vertical')

    # File selection widgets
    directory_selector = FileEdit(mode='d', label='Dataset')
    codebook_selector = FileEdit(mode='r', label='Codebook', filter='*.csv', visible=False)
    bit_order_selector = FileEdit(mode='r', label='Bit Order', filter='*.csv', visible=False)
    noise_map_selector = FileEdit(mode='r', label='Gain Map', filter='*.tiff, *.tif', visible=False)
    darkfield_selector = FileEdit(mode='r', label='Offset', filter='*.tiff, *.tif', visible=False)
    shading_selector = FileEdit(mode='r', label='Shading', filter='*.tiff, *.tif', visible=False)

    # Correction options dropdown
    correction_options = ComboBox(label="Correction Options", choices=["None", "Hotpixel correct", "Flatfield correct"], visible=False)

    # Other specific options (initially hidden)
    processing_options = [CheckBox(text='Register and process tiles', visible=False),
                        CheckBox(text='Decode tiles', visible=False),
                        CheckBox(text='Write individual polyDT tiffs', visible=False),
                        CheckBox(text='Write fused, downsampled polyDT tiff',visible=False)]

    # Summary label (initially hidden)
    summary_label = Label(value="", visible=False)

    # Default progress bars and labels
    progress_labels = [Label(value="Channel", visible=False),
                       Label(value="Round", visible=False),
                       Label(value="Tile", visible=False)]
    progress_bars = [ProgressBar(value=0, max=100, visible=False) for _ in range(3)]

    # Function to handle directory selection
    def handle_directory_change(event):
        if directory_selector.value:
            directory_path = Path(directory_selector.value)
            csv_files = list(directory_path.glob('*.csv'))
            if csv_files:
                codebook_selector.visible = True
                codebook_selector.value = Path(directory_selector.value) / Path('codebook.csv')
                summary_label.visible = False
            else:
                summary_label.value = "No CSV file found in the directory."
                summary_label.visible = True

    # Function to handle codebook file selection
    def handle_codebook_change(event):
        if codebook_selector.value:
            bit_order_selector.visible = True
            bit_order_selector.value = Path(directory_selector.value) / Path('bit_order.csv')
            summary_label.visible = False

    # Function to handle bit order file selection
    def handle_bit_order_change(event):
        if bit_order_selector.value:
            correction_options.visible = True
            correction_options.value = "Hotpixel correct"
            correction_options.value = "None"
            summary_label.visible = False

    # Function to handle correction option change
    def handle_options_change(event):
        correction_choice = correction_options.value
        noise_map_selector.visible = correction_choice in ["Hotpixel correct", "Flatfield correct"]
        darkfield_selector.visible = correction_choice == "Flatfield correct"
        shading_selector.visible = correction_choice == "Flatfield correct"
        for option in options:
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
        
    def update_progress(progress_updates):
        progress_bar_names = {
            "Round": "Round",
            "Tile": "Tile",
            "Channel": "Channel",
            "Register/Process": "Register/Process",  
            "Decode": "Decode",
        }

        for key, progress_value in progress_updates.items():
            if key in progress_bar_names:
                # Find the progress bar by its label
                for widget in layout:
                    if isinstance(widget, Label) and widget.value == progress_bar_names[key]:
                        # The next widget in the layout after the label should be the progress bar
                        index = layout.index(widget) + 1
                        if index < len(layout) and isinstance(layout[index], ProgressBar) and layout[index].visible:
                            layout[index].value = progress_value
                            break

    # Start button (initially visible)
    start_button = PushButton(text="Start Task", visible=False)

    # Exit button
    exit_button = PushButton(text="Exit")

    # Function to close the GUI
    def close_gui():
        layout.native.close()
    
    @ensure_main_thread
    def threaded_postprocess(stitching_options,noise_map_path,darkfield_path):
        yield from postprocess(correction_options.value, stitching_options, directory_selector.value, 
                                    codebook_selector.value, bit_order_selector.value,
                                    noise_map_path, darkfield_path)

    # Start the task
    def start_task():
        options = {option.text: option.value for option in processing_options if option.value}
        
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
        if options.get('Register and process tiles'):
            layout.extend([Label(value="Register/Process", visible=True), 
                        ProgressBar(value=0, max=100, visible=True)])
        if options.get('Decode'):
            layout.extend([Label(value="Decoding", visible=True), 
                        ProgressBar(value=0, max=100, visible=True)])

        noise_map_path = noise_map_selector.value if validate_tiff_file(noise_map_selector.value) else None
        darkfield_path = darkfield_selector.value if validate_tiff_file(darkfield_selector.value) else None
        shading_path = shading_selector.value if validate_tiff_file(shading_selector.value) else None

        # Start the worker with additional arguments
        worker = threaded_postprocess(options,noise_map_path,darkfield_path)
        worker.yielded.connect(update_progress)
        worker.start()
        worker.finished.connect(close_gui)

    def update_start_button_visibility():
        correction_choice = correction_options.value
        valid_noise_map = validate_tiff_file(noise_map_selector.value) if noise_map_selector.visible else True
        valid_darkfield = validate_tiff_file(darkfield_selector.value) if darkfield_selector.visible else True
        valid_shading = validate_tiff_file(shading_selector.value) if shading_selector.visible else True

        start_button.visible = (correction_choice == "None" or
                                (correction_choice == "Hotpixel correct" and valid_noise_map) or
                                (correction_choice == "Flatfield correct" and valid_noise_map and valid_darkfield and valid_shading))


    correction_options.changed.connect(update_start_button_visibility)
    noise_map_selector.changed.connect(update_start_button_visibility)

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
                shading_selector, summary_label] + options + [start_button, exit_button])

    # Show the GUI with the specified window title
    layout.native.setWindowTitle("Postprocess raw data - qi2lab widefield MERFISH")
    layout.show(run=True)

if __name__ == '__main__':
    create_gui()