import time

import numpy as onp
import plotly.express as px
import plotly.graph_objects as go

import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

class SparseMapGui:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
            titlebar_content=None, 
            control_layout="collapsible",
            show_share_button=False,
            brand_color=(180, 180, 30),)

        self.need_update = True

        # Load the colmap info.
        self.gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        self.gui_point_size = self.server.gui.add_slider(
            "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.05
        )

        @self.gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
                [0.0, -1.0, 0.0]
            )

        @self.gui_point_size.on_update
        def _(_) -> None:
            # point_cloud.point_size = gui_point_size.value
            self.need_update = True

    def update(self):
        if self.need_update:
            self.need_update = False

    def run(self):
        while True:
            self.update()
            time.sleep(0.1)

def main() -> None:
    gui = SparseMapGui()
    gui.run()

if __name__ == "__main__":
    main()
