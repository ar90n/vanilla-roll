# render_rotation
Rendering MRA data with horizontal or vertical camera rotation.

![output](https://raw.githubusercontent.com/ar90n/vanilla-roll/assets/images/render_rotation_result.gif)


## Install vanilla-roll
```bash
$ pip install -r requirements.txt
```

## Select rendering mode
Select rendering mode by removing comment out.

```python
# mode = vr.rendering.mode.MIP()
# mode = vr.rendering.mode.MinP()
mode = vr.rendering.mode.VR(get_preset(Preset.MR_ANGIO))
```

## Choose rotation direction
Choose rotation direction by removing comment out.

```python
for i, ret in enumerate(
    vr.render_horizontal_rotations(volume, mode=mode, n=12, spacing=0.3)
    # vr.render_vertical_rotations(volume, mode=mode, n=12, spacing=0.3)
):
```
