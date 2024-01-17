from magicgui.widgets import Container, FileEdit, ComboBox, PushButton, ProgressBar, Label
from pathlib import Path
from superqt.utils import thread_worker
from wf_merfish.postprocess.localization import visualize_tile, localize_tile, batch_localize

def create_gui():
    # Main GUI container
    layout = Container(layout='vertical')

    # File selection widget
    directory_selector = FileEdit(mode='d', label='Select Directory')

    # Dropdown menu for directories (initially hidden)
    dropdown_menu = ComboBox(label="Directories", visible=False)

    # Progress bars (initially hidden)
    progress_labels = [Label(value="Bit", visible=False),
                       Label(value="Tile", visible=False)]
    progress_bars = [ProgressBar(value=0, max=100, visible=False) for _ in range(2)]

    # # Placeholder functions for thread workers
    @thread_worker
    def visualize_worker(path: Path):
        # Your visualization logic here
        pass

    @thread_worker
    def localize_worker(path: Path):
        # Your localization logic here
        pass

    @thread_worker
    def batch_localize_worker(directory: Path):
        # Your batch localization logic here
        yield {"Bit": 50, "Tile": 50}  # Example yield, replace with actual progress
        
    def get_directories(directory: Path):
        # Replace with your actual function to list directories
        if not directory.is_dir():
            return []

        return [subdir for subdir in directory.iterdir() if subdir.is_dir()]

    # Function to handle directory selection
    def handle_directory_change(event):
        if directory_selector.value:
            directory_path = Path(directory_selector.value)
            list_of_dirs = get_directories(directory_path)  # Replace with your actual function
            dropdown_menu.choices = [p.name for p in list_of_dirs]
            dropdown_menu.visible = True
            visualize_button.visible = True
            localize_button.visible = True

    # Function to update progress bars
    def update_progress(progress):
        bit_progress_bar.value = progress["Bit"]
        tile_progress_bar.value = progress["Tile"]

    # Button click handlers
    def on_visualize_clicked():
        visualize_worker = visualize_worker(Path(dropdown_menu.value))
        visualize_worker.start()

    def on_localize_clicked():
        localize_worker = localize_worker(Path(dropdown_menu.value))
        localize_worker.start()

    def on_batch_localize_clicked():
        batch_localize_worker = batch_localize_worker(directory_selector.value)
        batch_localize_worker.yielded.connect(update_progress)
        batch_localize_worker.start()
        # Add default progress bars and labels
        for label, bar in zip(progress_labels, progress_bars):
            label.visible = True
            bar.visible = True
            layout.extend([label, bar])

    # Buttons (initially hidden)
    visualize_button = PushButton(text="Visualize", visible=False, clicked=on_visualize_clicked)
    localize_button = PushButton(text="Localize This Tile", visible=False, clicked=on_localize_clicked)
    batch_localize_button = PushButton(text="Batch Localize All Tiles", visible=False, clicked=on_batch_localize_clicked)
    exit_button = PushButton(text="Exit", clicked=lambda: layout.native.close())

    # Connect the directory selector change event
    directory_selector.changed.connect(handle_directory_change)

    # Add widgets to the layout
    layout.extend([directory_selector, dropdown_menu, visualize_button, localize_button, 
                   batch_localize_button, bit_progress_bar, tile_progress_bar, exit_button])

    # Show the GUI
    layout.native.setWindowTitle("Localize spots - qi2lab widefield MERFISH")
    layout.show(run=True)

create_gui()