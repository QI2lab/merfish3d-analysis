import napari
from magicgui.widgets import Container, FileEdit, ComboBox, PushButton, ProgressBar, Label
from superqt.utils import thread_worker
from pathlib import Path
import time
from PyQt5.QtWidgets import QApplication

# Custom worker functions (to be defined in a separate file, e.g., workers.py)
@thread_worker
def visualize_worker(path: Path, prepare_and_show_napari):
    # Your visualization logic here
    prepare_and_show_napari()  # Prepare and show Napari
    yield  # Pause execution until Napari is closed
    # Continue with the rest of the logic

def prepare_and_show_napari_for_localization(tile_id,em_wvl, voxel_zyx_um):
    viewer = napari.Viewer()

    # Add the plugin dock widget
    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
        "napari-spot-detection", "Spot detection"
    )

    # Set the plugin widget parameters
    plugin_widget.txt_ri.setText('1.51')
    plugin_widget.txt_lambda_em.setText(str(em_wvl * 100))
    plugin_widget.txt_dc.setText(str(voxel_zyx_um[1]))
    plugin_widget.txt_dstage.setText(str(voxel_zyx_um[0]))
    plugin_widget.chk_skewed.setChecked(False)

    viewer.show()
    wait_for_viewer_to_close(viewer)

def wait_for_viewer_to_close(viewer):
    # Wait for the viewer to close
    while viewer.window._qt_window.isVisible():
        time.sleep(0.1)  # Small delay to prevent freezing
        QApplication.processEvents()  # Process other Qt events

# Example usage in a worker function
@thread_worker
def visualize_worker(path: Path):
    # Your visualization logic here
    prepare_and_show_napari_for_localization(em_wvl, voxel_zyx_um)
    yield  # Pause execution until Napari is closed
    # Continue with the rest of the logic

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

    # Function to open Napari and wait for it to close
    def open_napari_and_wait():
        viewer = napari.Viewer()
        viewer.show()

        # Wait for the viewer to close
        while viewer.window._qt_window.isVisible():
            time.sleep(0.1)  # Small delay to prevent freezing
            QApplication.processEvents()  # Process other Qt events

    # Function to handle directory selection
    def handle_directory_change(event):
        if directory_selector.value:
            directory_path = Path(directory_selector.value)
            csv_files = list(directory_path.glob('*.csv'))
            if csv_files:
                dropdown_menu.choices = [p.name for p in csv_files]
                dropdown_menu.visible = True

    # Function to update progress bars
    def update_progress(progress):
        bit_progress_bar.value = progress["Bit"]
        tile_progress_bar.value = progress["Tile"]

    # Button click handlers
    def on_visualize_clicked():
        worker = visualize_worker(Path(dropdown_menu.value), open_napari_and_wait)
        worker.start()

    def on_localize_clicked():
        worker = localize_worker(Path(dropdown_menu.value), open_napari_and_wait)
        worker.start()

    # Buttons (initially hidden)
    visualize_button = PushButton(text="Visualize", visible=False, clicked=on_visualize_clicked)
    localize_button = PushButton(text="Localize This Tile", visible=False, clicked=on_localize_clicked)
    exit_button = PushButton(text="Exit", clicked=lambda: layout.native.close())

    # Connect the directory selector change event
    directory_selector.changed.connect(handle_directory_change)

    # Add widgets to the layout
    layout.extend([directory_selector, dropdown_menu, visualize_button, localize_button, 
                   bit_progress_bar, tile_progress_bar, exit_button])

    # Show the GUI
    layout.native.setWindowTitle("Localize spots - qi2lab widefield MERFISH")
    layout.show(run=True)

create_gui()