#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for viewing Physics objects in the DM Control viewer."""

import abc
import enum
import sys
from typing import Dict, Optional

import numpy as np
from dm_control.mujoco import wrapper
from mujoco_kitchen.adept_envs.simulation import module

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

DEFAULT_WINDOW_TITLE = "MuJoCo Viewer"

_MAX_RENDERBUFFER_SIZE = 2048


class RenderMode(enum.Enum):
    """Rendering modes for offscreen rendering."""

    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2


class Renderer(abc.ABC):
    """Base interface for rendering simulations."""

    def __init__(self, camera_settings: Optional[Dict] = None):
        self._camera_settings = camera_settings

    @abc.abstractmethod
    def close(self):
        """Cleans up any resources being used by the renderer."""

    @abc.abstractmethod
    def render_to_window(self):
        """Renders the simulation to a window."""

    @abc.abstractmethod
    def render_offscreen(
        self,
        width: int,
        height: int,
        mode: RenderMode = RenderMode.RGB,
        camera_id: int = -1,
    ) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """

    def _update_camera(self, camera):
        """Updates the given camera to move to the initial settings."""
        if not self._camera_settings:
            return
        distance = self._camera_settings.get("distance")
        azimuth = self._camera_settings.get("azimuth")
        elevation = self._camera_settings.get("elevation")
        lookat = self._camera_settings.get("lookat")

        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat


class MjPyRenderer(Renderer):
    """Class for rendering mujoco_py simulations."""

    def __init__(self, sim, **kwargs):
        assert isinstance(
            sim, module.get_mujoco_py().MjSim
        ), "MjPyRenderer takes a mujoco_py MjSim object."
        super().__init__(**kwargs)
        self._sim = sim
        self._onscreen_renderer = None
        self._offscreen_renderer = None

    def render_to_window(self):
        """Renders the simulation to a window."""
        if not self._onscreen_renderer:
            self._onscreen_renderer = module.get_mujoco_py().MjViewer(self._sim)
            self._update_camera(self._onscreen_renderer.cam)

        self._onscreen_renderer.render()

    def render_offscreen(
        self,
        width: int,
        height: int,
        mode: RenderMode = RenderMode.RGB,
        camera_id: int = -1,
    ) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        if not self._offscreen_renderer:
            self._offscreen_renderer = module.get_mujoco_py().MjRenderContextOffscreen(
                self._sim
            )

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(self._offscreen_renderer.cam)

        self._offscreen_renderer.render(width, height, camera_id)
        if mode == RenderMode.RGB:
            data = self._offscreen_renderer.read_pixels(width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == RenderMode.DEPTH:
            data = self._offscreen_renderer.read_pixels(width, height, depth=True)[1]
            # Original image is upside-down, so flip it
            return data[::-1, :]
        else:
            raise NotImplementedError(mode)

    def close(self):
        """Cleans up any resources being used by the renderer."""


class DMRenderer(Renderer):
    """Class for rendering DM Control Physics objects."""

    def __init__(
        self,
        physics,
        clear_geom_group_0=False,
        camera_select_next=False,
        mjpy_sim=None,
        **kwargs
    ):
        assert isinstance(
            physics, module.get_dm_mujoco().Physics
        ), "DMRenderer takes a DM Control Physics object."
        super().__init__(**kwargs)
        self._physics = physics
        self._window = None
        self.clear_geom_group_0 = clear_geom_group_0
        self.camera_select_next = camera_select_next
        # Set the camera to lookat the center of the geoms. (mujoco_py does
        # this automatically.
        if "lookat" not in self._camera_settings:
            self._camera_settings["lookat"] = [
                np.median(self._physics.data.geom_xpos[:, i]) for i in range(3)
            ]
        self.mjpy_sim = mjpy_sim

    def render_to_window(self):
        """Renders the Physics object to a window.

        The window continuously renders the Physics in a separate thread.

        This function is a no-op if the window was already created.
        """
        if self.mjpy_sim:
            for i, val in enumerate(self.mjpy_sim.data.qpos):
                self._physics.data.qpos[i] = val
            for i, val in enumerate(self.mjpy_sim.data.qvel):
                self._physics.data.qvel[i] = val
            for i, val in enumerate(self.mjpy_sim.model.body_pos):
                self._physics.model.body_pos[i] = val
            for i, val in enumerate(self.mjpy_sim.model.body_quat):
                self._physics.model.body_quat[i] = val
            self._physics.forward()
        if not self._window:
            self._window = DMRenderWindow(
                clear_geom_group_0=self.clear_geom_group_0,
                camera_select_next=self.camera_select_next,
            )
            self._window.load_model(self._physics)
            self._update_camera(self._window.camera)

        # useful for tuning camera parameters!
        # print(
        #     dict(
        #         distance=self._window.camera.distance,
        #         lookat=self._window.camera.lookat,
        #         azimuth=self._window.camera.azimuth,
        #         elevation=self._window.camera.elevation,
        #     )
        # )

        self._window.run_frame()

    def render_offscreen(
        self,
        width: int,
        height: int,
        mode: RenderMode = RenderMode.RGB,
        camera_id: int = -1,
    ) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        if self.mjpy_sim:
            for i, val in enumerate(self.mjpy_sim.data.qpos):
                self._physics.data.qpos[i] = val
            for i, val in enumerate(self.mjpy_sim.data.qvel):
                self._physics.data.qvel[i] = val
            for i, val in enumerate(self.mjpy_sim.model.body_pos):
                self._physics.model.body_pos[i] = val
            for i, val in enumerate(self.mjpy_sim.model.body_quat):
                self._physics.model.body_quat[i] = val
            self._physics.forward()
        mujoco = module.get_dm_mujoco()
        # TODO(michaelahn): Consider caching the camera.
        camera = mujoco.Camera(
            physics=self._physics, height=height, width=width, camera_id=camera_id
        )
        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(
                camera._render_camera,  # pylint: disable=protected-access
            )

        scene_option = wrapper.MjvOption()
        if self.clear_geom_group_0:
            scene_option.geomgroup[0] = 0
            scene_option.sitegroup[1] = 0
        image = camera.render(
            depth=(mode == RenderMode.DEPTH),
            segmentation=(mode == RenderMode.SEGMENTATION),
            scene_option=scene_option,
        )
        camera._scene.free()  # pylint: disable=protected-access
        return image

    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(
        self,
        width: int = DEFAULT_WINDOW_WIDTH,
        height: int = DEFAULT_WINDOW_HEIGHT,
        title: str = DEFAULT_WINDOW_TITLE,
        clear_geom_group_0: bool = False,
        camera_select_next: bool = False,
    ):
        """Creates a graphical render window.

        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        dmv = module.get_dm_viewer()
        self._viewport = dmv.renderer.Viewport(width, height)
        self._window = dmv.gui.RenderWindow(width, height, title)
        self._viewer = dmv.viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard
        )
        self._draw_surface = None
        self._renderer = dmv.renderer.NullRenderer()
        self.camera_select_next = camera_select_next
        if clear_geom_group_0:
            # for robosuite this gets rid of extraneous collision/visual meshes
            self._viewer._render_settings.geom_groups[0] = 0
            self._viewer._render_settings.site_groups[1] = 0

    @property
    def camera(self):
        return self._viewer._camera._camera

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()

        self._draw_surface = module.get_dm_render().Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE
        )
        self._renderer = module.get_dm_viewer().renderer.OffScreenRenderer(
            physics.model, self._draw_surface
        )

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation.

        NOTE: This is extremely slow at the moment.
        """
        if self.camera_select_next:
            # for robosuite we want to shift the camera idx by 1
            self._viewer._camera_select.select_next()
        glfw = module.get_dm_viewer().gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window, pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()
