from __future__ import annotations

from pathlib import Path


class VideoRecorder:
    def __init__(
        self,
        model: object,
        path: str | Path,
        *,
        camera: str,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        sample_stride: int = 1,
    ) -> None:
        try:
            import imageio.v2 as imageio
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "Video export dependencies are not installed. "
                "Install them with: pip install -e '.[sim]'"
            ) from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.sample_stride = max(1, sample_stride)
        self.sample_count = 0
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.writer = imageio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
        )
        self.frame_count = 0

    def record(self, data: object) -> None:
        self.sample_count += 1
        if self.sample_count % self.sample_stride:
            return
        self.renderer.update_scene(data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frame_count += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
