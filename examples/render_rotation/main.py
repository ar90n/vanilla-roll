import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import skimage.io

import vanilla_roll as vr
from vanilla_roll.rendering.transfer_function import Preset, get_preset

# from A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie
# https://www.openfmri.org/dataset/ds000113/
MRA_FILE_URL = "https://s3.amazonaws.com/openneuro/ds000113/ds000113_unrevisioned/uncompressed/sub003/angio/angio001.nii.gz"  # noqa: E501


def fetch_mra_volume() -> vr.volume.Volume:
    with TemporaryDirectory() as tmpdir:
        mra_file = Path(tmpdir) / "mra.nii.gz"
        urllib.request.urlretrieve(MRA_FILE_URL, mra_file)
        return vr.io.read_nifti(mra_file)


def save_result(ret: vr.rendering.types.RenderingResult, path: str):
    img_array = vr.rendering.convert_image_to_array(ret.image)
    skimage.io.imsave(path, np.from_dlpack(img_array))  # type: ignore


def main():
    volume = fetch_mra_volume()

    # mode = vr.rendering.mode.MIP()
    # mode = vr.rendering.mode.MinP()
    mode = vr.rendering.mode.VR(get_preset(Preset.MR_ANGIO))

    for i, ret in enumerate(
        # vr.render_horizontal_rotations(volume, mode=mode, n=12, spacing=0.3)
        vr.render_vertical_rotations(volume, mode=mode, n=12, spacing=0.3)
    ):
        save_result(ret, f"mra_{i:03}.png")


if __name__ == "__main__":
    main()
