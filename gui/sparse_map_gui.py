import time
from typing import List, Optional
import numpy as np
import viser
import viser.transforms as tf
import plotly.express as px
from pyslamcpts import SparseMap

class SparseMapGui:
    def __init__(self):
        self.server = viser.ViserServer()
        self.configure_gui()

        self.map = SparseMap()
        self.map_points = None
        self.cameras = []
        self.frames: List[viser.FrameHandle] = []
        self.need_update = True
        self.displayed_3d_container: Optional[viser.Gui3dContainerHandle] = None
        self.displayed_features = None
        self.displayed_reproj = None

    def configure_gui(self):
        """Configure GUI elements."""
        self.server.gui.configure_theme(
            titlebar_content=None,
            control_layout="collapsible",
            control_width="large",
            show_share_button=False,
            brand_color=(180, 180, 30),
        )

        self.gui_info_panel = self.server.gui.add_text(
            "Map Info", initial_value="Frames: 0 | Points: 0"
        )
        self.gui_info_panel.disabled = True

        with self.server.gui.add_folder("Control"):
            self.gui_show_frames = self.server.gui.add_checkbox("Show frames", initial_value=True)
            self.gui_show_points = self.server.gui.add_checkbox("Show points", initial_value=True)
            self.gui_show_grid = self.server.gui.add_checkbox("Show grid", initial_value=True)
            self.gui_point_size = self.server.gui.add_slider(
                "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.05
            )
            self.gui_curr_frame = self.server.gui.add_slider(
                "Current Frame ID", min=0, max=500, step=1, initial_value=0, disabled=True
            )
            self.gui_reset = self.server.gui.add_button(
                "Reset View", hint="Reset the camera view to the first frame",
            )
            self.gui_prev_frame = self.server.gui.add_button(
                "Prev Frame View", hint="Set the camera view to the previous frame",
            )
            self.gui_next_frame = self.server.gui.add_button(
                "Next Frame View", hint="Set the camera view to the next frame",
            )
        self.gui_upload_button = self.server.gui.add_upload_button("Upload map file")
        self.attach_gui_callbacks()

    def attach_gui_callbacks(self):
        """Attach callbacks to GUI elements."""
        def mark_update_needed(_):
            self.need_update = True

        self.gui_point_size.on_update(mark_update_needed)
        self.gui_show_frames.on_update(mark_update_needed)
        self.gui_show_points.on_update(mark_update_needed)
        self.gui_show_grid.on_update(mark_update_needed)

        @self.gui_reset.on_click
        def _(_) -> None:
            self.move_to_current_frame()

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            self.gui_curr_frame.value = (self.gui_curr_frame.value - 1) % len(self.frames)
            self.move_to_current_frame()

        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.gui_curr_frame.value = (self.gui_curr_frame.value + 1) % len(self.frames)
            self.move_to_current_frame()

        @self.gui_upload_button.on_upload
        def _(_) -> None:
            file = self.gui_upload_button.value
            print(f"Uploaded file: {file.name}, size: {len(file.content)} bytes")

    def update_info_panel(self):
        """Update the information panel with current data."""
        frame_count = len(self.map.get_frame_ids())
        point_count = self.map_points.shape[0] if self.map_points is not None else 0
        self.gui_info_panel.value = f"Frames: {frame_count} | Points: {point_count}"

    def move_frame(self, client: viser.ClientHandle, frame: viser.FrameHandle) -> None:
        """Move smoothly to a specified frame."""
        T_world_current = tf.SE3.from_rotation_and_translation(
            tf.SO3(client.camera.wxyz), client.camera.position
        )
        T_world_target = tf.SE3.from_rotation_and_translation(
            tf.SO3(frame.wxyz), frame.position
        ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

        T_current_target = T_world_current.inverse() @ T_world_target

        # Smooth animation with adaptive steps based on frame rate
        steps = 20
        for j in range(steps):
            T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / (steps - 1))
            with client.atomic():
                client.camera.wxyz = T_world_set.rotation().wxyz
                client.camera.position = T_world_set.translation()
            time.sleep(1.0 / 120.0)  # Faster, smoother update

        client.camera.look_at = frame.position

    def move_to_current_frame(self):
        """Move to the currently selected frame."""
        if self.frames:
            frame = self.frames[self.gui_curr_frame.value]
            for client in self.server.get_clients().values():
                self.move_frame(client, frame)

    def attach_callback(self, frame: viser.FrameHandle, frame_id: int, cam_id: int) -> None:
        """Attach a click event callback to a camera frustum."""
        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None

            if self.displayed_3d_container is not None:
                self.displayed_3d_container.remove()

            self.displayed_3d_container = client.scene.add_3d_gui_container(
                f"/sparse_map/frame_{frame_id}/{cam_id}/gui"
            )
            self.attach_container_buttons(client, frame, frame_id, cam_id)

    def attach_container_buttons(self, client, frame, frame_id, cam_id):
        """Add buttons to a 3D container for frame navigation and visualization."""
        with self.displayed_3d_container:
            go_to = client.gui.add_button("Go to")
            feature_img = client.gui.add_button("Show Features")
            reproj_img = client.gui.add_button("Show Reproject Image")
            close = client.gui.add_button("Close")

        @go_to.on_click
        def _(_) -> None:
            self.move_frame(client, frame)

        @feature_img.on_click
        def _(_) -> None:
            if self.displayed_features is not None:
                self.displayed_features.remove()
            img = self.map.draw_keypoint(frame_id, cam_id)
            fig = px.imshow(img)
            fig.update_layout(title=f"Feature Extraction ({frame_id}/{cam_id})", margin=dict(l=20, r=20))
            self.displayed_features = self.server.gui.add_plotly(figure=fig, aspect=1.0)

        @reproj_img.on_click
        def _(_) -> None:
            if self.displayed_reproj is not None:
                self.displayed_reproj.remove()
            img = self.map.draw_reproj_keypoint(frame_id, cam_id)
            fig = px.imshow(img)
            fig.update_layout(title=f"Feature Reprojection ({frame_id}/{cam_id})", margin=dict(l=20, r=20))
            self.displayed_reproj = self.server.gui.add_plotly(figure=fig, aspect=1.0)

        @close.on_click
        def _(_) -> None:
            self.clear_displayed_images()

    def clear_displayed_images(self):
        """Clear the displayed features and reprojection images."""
        if self.displayed_3d_container is not None:
            self.displayed_3d_container.remove()
            self.displayed_3d_container = None
        if self.displayed_features is not None:
            self.displayed_features.remove()
            self.displayed_features = None
        if self.displayed_reproj is not None:
            self.displayed_reproj.remove()
            self.displayed_reproj = None

    def visualize_frames(self) -> None:
        """Visualize frames in the sparse map."""
        frame_ids = self.map.get_frame_ids()
        downsample_factor = 2

        positions = []
        for frame_id in frame_ids:
            frame = self.map.get_frame(frame_id)
            pose_wb = np.array(frame.get_body_pose())

            for cam_id, img in enumerate(frame.imgs()):
                pose_bc = np.array(self.cameras[cam_id].get_extrinsic())
                pose_wc = pose_wb @ pose_bc
                T_world_camera = tf.SE3.from_rotation_and_translation(
                    tf.SO3.from_matrix(pose_wc[:3, :3]), pose_wc[:3, 3]
                )

                frame_handle = self.server.scene.add_frame(
                    f"/sparse_map/frame_{frame_id}/{cam_id}",
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    axes_length=0.1,
                    axes_radius=0.005,
                    visible=self.gui_show_frames.value
                )
                self.server.scene.add_label(
                    f"/sparse_map/frame_{frame_id}/{cam_id}/label",
                    text=f"Frame {frame_id} / Camera {cam_id}",
                    visible=self.gui_show_frames.value
                )
                self.add_camera_frustum(frame_id, cam_id, img, downsample_factor)
                self.attach_callback(frame_handle, frame_id, cam_id)
                self.frames.append(frame_handle)
                if cam_id == 0:
                    positions.append(tuple(T_world_camera.translation()))

        positions = tuple(positions)
        self.server.scene.add_spline_catmull_rom(
            f"/sparse_map/frame_trajectory",
            positions,
            tension=0.5,
            line_width=3.0,
            color=(180, 0, 180),
        )

    def add_camera_frustum(self, frame_id: int, cam_id: int, img: np.ndarray, downsample_factor: int):
        """Add a camera frustum visualization to the scene."""
        H, W = img.shape[:2]
        fy = max(H, W) / 2
        image = img[::downsample_factor, ::downsample_factor]
        self.server.scene.add_camera_frustum(
            f"/sparse_map/frame_{frame_id}/{cam_id}/frustum",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.15,
            image=image,
        )

    def load_map(self, file: str) -> None:
        """Load the sparse map from a file."""
        if not self.map.load(file):
            print("Failed to load map")
            return

        self.map_points = np.array(self.map.get_world_points()).reshape(-1, 3)
        calibration = self.map.get_calibration()
        self.cameras = [calibration.get_camera(cam_id) for cam_id in range(calibration.cam_num())]
        self.need_update = True

    def update(self) -> None:
        """Update the visualizations if necessary."""
        if self.need_update:
            self.need_update = False
            self.visualize_frames()
            if self.map_points is not None:
                self.server.scene.add_point_cloud(
                    name="/sparse_map/pcd",
                    points=self.map_points,
                    colors=np.random.randint(0, 256, size=(self.map_points.shape[0], 3)),
                    point_size=self.gui_point_size.value,
                    visible=self.gui_show_points.value
                )
            self.update_info_panel()
            self.server.scene.add_grid(
                "/grid",
                width=10.0,
                height=20.0,
                width_segments=10,
                height_segments=20,
                plane="xy",
                visible=self.gui_show_grid.value
            )

    def run(self, map_file) -> None:
        """Run the main loop."""
        self.load_map(map_file)
        while True:
            self.update()
            time.sleep(0.1)

def main(map_file) -> None:
    gui = SparseMapGui()
    gui.run(map_file)

if __name__ == "__main__":
    map_file = "/mnt/i/project/slam/datasets/TXPJ/test2/extract/sparse_map.bin"
    main(map_file)
