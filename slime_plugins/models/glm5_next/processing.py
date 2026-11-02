from sglang.srt.multimodal.glm5_next_processing import Glm5NextProcessor as ServingGlm5NextProcessor


class Glm5NextProcessor(ServingGlm5NextProcessor):
    """Training uses exactly the same image expansion as rollout serving."""

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if "pixel_values_videos" in output:
            raise NotImplementedError("GLM53 training processor currently supports images, not videos")
        return output
