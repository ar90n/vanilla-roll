from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, TypeAlias

import vanilla_roll.array_api as xp


@dataclass(frozen=True)
class Result:
    r: xp.Array
    g: xp.Array
    b: xp.Array
    opacity: xp.Array


@dataclass(frozen=True)
class OpacityControlPoint:
    intensity: float
    opacity: float


@dataclass(frozen=True)
class ColorControlPoint:
    intensity: float
    r: float
    g: float
    b: float


TransferFunction: TypeAlias = Callable[[xp.Array], Result]


def make_transfer_function(
    opacity_control_points: Iterable[OpacityControlPoint],
    color_control_points: Iterable[ColorControlPoint],
) -> TransferFunction:
    color_control_points = list(color_control_points)
    opacity_control_points = list(opacity_control_points)

    def _f(x: xp.Array) -> Result:
        absorption = xp.zeros_like(x)
        for (beg, end) in zip(opacity_control_points, opacity_control_points[1:]):
            mask = xp.logical_and(beg.intensity <= x, x < end.intensity)
            ratio = (x[mask] - beg.intensity) / (end.intensity - beg.intensity)
            absorption[mask] = ratio * (end.opacity - beg.opacity) + beg.opacity

        r = xp.zeros_like(x)
        g = xp.zeros_like(x)
        b = xp.zeros_like(x)
        for (beg, end) in zip(color_control_points, color_control_points[1:]):
            mask = xp.logical_and(beg.intensity <= x, x < end.intensity)
            ratio = (x[mask] - beg.intensity) / (end.intensity - beg.intensity)
            r[mask] = ratio * (end.r - beg.r) + beg.r
            g[mask] = ratio * (end.g - beg.g) + beg.g
            b[mask] = ratio * (end.b - beg.b) + beg.b

        return Result(r=r, g=g, b=b, opacity=absorption)

    return _f


class Preset(Enum):
    CT_AAA = "CT-AAA"
    CT_AAA2 = "CT-AAA2"
    CT_BONE = "CT-Bone"
    CT_BONES = "CT-Bones"
    CT_CARDIAC = "CT-Cardiac"
    CT_CARDIAC2 = "CT-Cardiac2"
    CT_CARDIAC3 = "CT-Cardiac3"
    CT_CHEST_CONTRAST_ENHANCED = "CT-Chest-Contrast-Enhanced"
    CT_CHEST_VESSELS = "CT-Chest-Vessels"
    CT_CORONARY_ARTERIES = "CT-Coronary-Arteries"
    CT_CORONARY_ARTERIES_2 = "CT-Coronary-Arteries-2"
    CT_CORONARY_ARTERIES_3 = "CT-Coronary-Arteries-3"
    CT_CROPPED_VOLUME_BONE = "CT-Cropped-Volume-Bone"
    CT_FAT = "CT-Fat"
    CT_LIVER_VASCULATURE = "CT-Liver-Vasculature"
    CT_LUNG = "CT-Lung"
    CT_MIP = "CT-MIP"
    CT_MUSCLE = "CT-Muscle"
    CT_PULMONARY_ARTERIES = "CT-Pulmonary-Arteries"
    CT_SOFT_TISSUE = "CT-Soft-Tissue"
    MR_ANGIO = "MR-Angio"
    MR_DEFAULT = "MR-Default"
    MR_MIP = "MR-MIP"
    MR_T2_BRAIN = "MR-T2-Brain"
    DTI_FA_BRAIN = "DTI-FA-Brain"


def get_preset(preset: Preset) -> TransferFunction:
    opacity_control_points, color_control_points = _preset_control_points[preset]
    return make_transfer_function(opacity_control_points, color_control_points)


# folloing parameters from glance
# https://github.com/Kitware/glance

_preset_control_points: dict[
    Preset, tuple[list[OpacityControlPoint], list[ColorControlPoint]]
] = {
    Preset.CT_AAA: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=143.556, opacity=0),
            OpacityControlPoint(intensity=166.222, opacity=0.686275),
            OpacityControlPoint(intensity=214.389, opacity=0.696078),
            OpacityControlPoint(intensity=419.736, opacity=0.833333),
            OpacityControlPoint(intensity=3071, opacity=0.803922),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=143.556, r=0.615686, g=0.356863, b=0.184314),
            ColorControlPoint(intensity=166.222, r=0.882353, g=0.603922, b=0.290196),
            ColorControlPoint(intensity=214.389, r=1, g=1, b=1),
            ColorControlPoint(intensity=419.736, r=1, g=0.937033, b=0.954531),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_AAA2: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=129.542, opacity=0),
            OpacityControlPoint(intensity=145.244, opacity=0.166667),
            OpacityControlPoint(intensity=157.02, opacity=0.5),
            OpacityControlPoint(intensity=169.918, opacity=0.627451),
            OpacityControlPoint(intensity=395.575, opacity=0.8125),
            OpacityControlPoint(intensity=1578.73, opacity=0.8125),
            OpacityControlPoint(intensity=3071, opacity=0.8125),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=129.542, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=145.244, r=0.6, g=0.627451, b=0.843137),
            ColorControlPoint(intensity=157.02, r=0.890196, g=0.47451, b=0.6),
            ColorControlPoint(intensity=169.918, r=0.992157, g=0.870588, b=0.392157),
            ColorControlPoint(intensity=395.575, r=1, g=0.886275, b=0.658824),
            ColorControlPoint(intensity=1578.73, r=1, g=0.829256, b=0.957922),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_BONE: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-16.4458, opacity=0),
            OpacityControlPoint(intensity=641.385, opacity=0.715686),
            OpacityControlPoint(intensity=3071, opacity=0.705882),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-16.4458, r=0.729412, g=0.254902, b=0.301961),
            ColorControlPoint(intensity=641.385, r=0.905882, g=0.815686, b=0.552941),
            ColorControlPoint(intensity=3071, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_BONES: (
        [
            OpacityControlPoint(intensity=-1000, opacity=0),
            OpacityControlPoint(intensity=152.19, opacity=0),
            OpacityControlPoint(intensity=278.93, opacity=0.190476),
            OpacityControlPoint(intensity=952, opacity=0.2),
        ],
        [
            ColorControlPoint(intensity=-1000, r=0.3, g=0.3, b=1),
            ColorControlPoint(intensity=-488, r=0.3, g=1, b=0.3),
            ColorControlPoint(intensity=463.28, r=1, g=0, b=0),
            ColorControlPoint(intensity=659.15, r=1, g=0.912535, b=0.0374849),
            ColorControlPoint(intensity=953, r=1, g=0.3, b=0.3),
        ],
    ),
    Preset.CT_CARDIAC: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-77.6875, opacity=0),
            OpacityControlPoint(intensity=94.9518, opacity=0.285714),
            OpacityControlPoint(intensity=179.052, opacity=0.553571),
            OpacityControlPoint(intensity=260.439, opacity=0.848214),
            OpacityControlPoint(intensity=3071, opacity=0.875),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-77.6875, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=94.9518, r=0.882353, g=0.603922, b=0.290196),
            ColorControlPoint(intensity=179.052, r=1, g=0.937033, b=0.954531),
            ColorControlPoint(intensity=260.439, r=0.615686, g=0, b=0),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_CARDIAC2: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=42.8964, opacity=0),
            OpacityControlPoint(intensity=163.488, opacity=0.428571),
            OpacityControlPoint(intensity=277.642, opacity=0.776786),
            OpacityControlPoint(intensity=1587, opacity=0.754902),
            OpacityControlPoint(intensity=3071, opacity=0.754902),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=42.8964, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=163.488, r=0.917647, g=0.639216, b=0.0588235),
            ColorControlPoint(intensity=277.642, r=1, g=0.878431, b=0.623529),
            ColorControlPoint(intensity=1587, r=1, g=1, b=1),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_CARDIAC3: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-86.9767, opacity=0),
            OpacityControlPoint(intensity=45.3791, opacity=0.169643),
            OpacityControlPoint(intensity=139.919, opacity=0.589286),
            OpacityControlPoint(intensity=347.907, opacity=0.607143),
            OpacityControlPoint(intensity=1224.16, opacity=0.607143),
            OpacityControlPoint(intensity=3071, opacity=0.616071),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-86.9767, r=0, g=0.25098, b=1),
            ColorControlPoint(intensity=45.3791, r=1, g=0, b=0),
            ColorControlPoint(intensity=139.919, r=1, g=0.894893, b=0.894893),
            ColorControlPoint(intensity=347.907, r=1, g=1, b=0.25098),
            ColorControlPoint(intensity=1224.16, r=1, g=1, b=1),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_CHEST_CONTRAST_ENHANCED: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=67.0106, opacity=0),
            OpacityControlPoint(intensity=251.105, opacity=0.446429),
            OpacityControlPoint(intensity=439.291, opacity=0.625),
            OpacityControlPoint(intensity=3071, opacity=0.616071),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=67.0106, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=251.105, r=0.882353, g=0.603922, b=0.290196),
            ColorControlPoint(intensity=439.291, r=1, g=0.937033, b=0.954531),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_CHEST_VESSELS: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-1278.35, opacity=0),
            OpacityControlPoint(intensity=22.8277, opacity=0.428571),
            OpacityControlPoint(intensity=439.291, opacity=0.625),
            OpacityControlPoint(intensity=3071, opacity=0.616071),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-1278.35, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=22.8277, r=0.882353, g=0.603922, b=0.290196),
            ColorControlPoint(intensity=439.291, r=1, g=0.937033, b=0.954531),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_CORONARY_ARTERIES: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=136.47, opacity=0),
            OpacityControlPoint(intensity=159.215, opacity=0.258929),
            OpacityControlPoint(intensity=318.43, opacity=0.571429),
            OpacityControlPoint(intensity=478.693, opacity=0.776786),
            OpacityControlPoint(intensity=3661, opacity=1),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=136.47, r=0, g=0, b=0),
            ColorControlPoint(intensity=159.215, r=0.159804, g=0.159804, b=0.159804),
            ColorControlPoint(intensity=318.43, r=0.764706, g=0.764706, b=0.764706),
            ColorControlPoint(intensity=478.693, r=1, g=1, b=1),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_CORONARY_ARTERIES_2: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=142.677, opacity=0),
            OpacityControlPoint(intensity=145.016, opacity=0.116071),
            OpacityControlPoint(intensity=192.174, opacity=0.5625),
            OpacityControlPoint(intensity=217.24, opacity=0.776786),
            OpacityControlPoint(intensity=384.347, opacity=0.830357),
            OpacityControlPoint(intensity=3661, opacity=0.830357),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=142.677, r=0, g=0, b=0),
            ColorControlPoint(intensity=145.016, r=0.615686, g=0, b=0.0156863),
            ColorControlPoint(intensity=192.174, r=0.909804, g=0.454902, b=0),
            ColorControlPoint(intensity=217.24, r=0.972549, g=0.807843, b=0.611765),
            ColorControlPoint(intensity=384.347, r=0.909804, g=0.909804, b=1),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_CORONARY_ARTERIES_3: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=128.643, opacity=0),
            OpacityControlPoint(intensity=129.982, opacity=0.0982143),
            OpacityControlPoint(intensity=173.636, opacity=0.669643),
            OpacityControlPoint(intensity=255.884, opacity=0.857143),
            OpacityControlPoint(intensity=584.878, opacity=0.866071),
            OpacityControlPoint(intensity=3661, opacity=1),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=128.643, r=0, g=0, b=0),
            ColorControlPoint(intensity=129.982, r=0.615686, g=0, b=0.0156863),
            ColorControlPoint(intensity=173.636, r=0.909804, g=0.454902, b=0),
            ColorControlPoint(intensity=255.884, r=0.886275, g=0.886275, b=0.886275),
            ColorControlPoint(intensity=584.878, r=0.968627, g=0.968627, b=0.968627),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_CROPPED_VOLUME_BONE: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=-451, opacity=0),
            OpacityControlPoint(intensity=-450, opacity=1),
            OpacityControlPoint(intensity=1050, opacity=1),
            OpacityControlPoint(intensity=3661, opacity=1),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=-451, r=0, g=0, b=0),
            ColorControlPoint(intensity=-450, r=0.0556356, g=0.0556356, b=0.0556356),
            ColorControlPoint(intensity=1050, r=1, g=1, b=1),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_FAT: (
        [
            OpacityControlPoint(intensity=-1000, opacity=0),
            OpacityControlPoint(intensity=-100, opacity=0),
            OpacityControlPoint(intensity=-99, opacity=0.15),
            OpacityControlPoint(intensity=-60, opacity=0.15),
            OpacityControlPoint(intensity=-59, opacity=0),
            OpacityControlPoint(intensity=101.2, opacity=0),
            OpacityControlPoint(intensity=952, opacity=0),
        ],
        [
            ColorControlPoint(intensity=-1000, r=0.3, g=0.3, b=1),
            ColorControlPoint(intensity=-497.5, r=0.3, g=1, b=0.3),
            ColorControlPoint(intensity=-99, r=0, g=0, b=1),
            ColorControlPoint(intensity=-76.946, r=0, g=1, b=0),
            ColorControlPoint(intensity=-65.481, r=0.835431, g=0.888889, b=0.0165387),
            ColorControlPoint(intensity=83.89, r=1, g=0, b=0),
            ColorControlPoint(intensity=463.28, r=1, g=0, b=0),
            ColorControlPoint(intensity=659.15, r=1, g=0.912535, b=0.0374849),
            ColorControlPoint(intensity=2952, r=1, g=0.300267, b=0.299886),
        ],
    ),
    Preset.CT_LIVER_VASCULATURE: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=149.113, opacity=0),
            OpacityControlPoint(intensity=157.884, opacity=0.482143),
            OpacityControlPoint(intensity=339.96, opacity=0.660714),
            OpacityControlPoint(intensity=388.526, opacity=0.830357),
            OpacityControlPoint(intensity=1197.95, opacity=0.839286),
            OpacityControlPoint(intensity=3661, opacity=0.848214),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=149.113, r=0, g=0, b=0),
            ColorControlPoint(intensity=157.884, r=0.501961, g=0.25098, b=0),
            ColorControlPoint(intensity=339.96, r=0.695386, g=0.59603, b=0.36886),
            ColorControlPoint(intensity=388.526, r=0.854902, g=0.85098, b=0.827451),
            ColorControlPoint(intensity=1197.95, r=1, g=1, b=1),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_LUNG: (
        [
            OpacityControlPoint(intensity=-1000, opacity=0),
            OpacityControlPoint(intensity=-600, opacity=0),
            OpacityControlPoint(intensity=-599, opacity=0.15),
            OpacityControlPoint(intensity=-400, opacity=0.15),
            OpacityControlPoint(intensity=-399, opacity=0),
            OpacityControlPoint(intensity=2952, opacity=0),
        ],
        [
            ColorControlPoint(intensity=-1000, r=0.3, g=0.3, b=1),
            ColorControlPoint(intensity=-600, r=0, g=0, b=1),
            ColorControlPoint(intensity=-530, r=0.134704, g=0.781726, b=0.0724558),
            ColorControlPoint(intensity=-460, r=0.929244, g=1, b=0.109473),
            ColorControlPoint(intensity=-400, r=0.888889, g=0.254949, b=0.0240258),
            ColorControlPoint(intensity=2952, r=1, g=0.3, b=0.3),
        ],
    ),
    Preset.CT_MIP: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-637.62, opacity=0),
            OpacityControlPoint(intensity=700, opacity=1),
            OpacityControlPoint(intensity=3071, opacity=1),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-637.62, r=1, g=1, b=1),
            ColorControlPoint(intensity=700, r=1, g=1, b=1),
            ColorControlPoint(intensity=3071, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_MUSCLE: (
        [
            OpacityControlPoint(intensity=-3024, opacity=0),
            OpacityControlPoint(intensity=-155.407, opacity=0),
            OpacityControlPoint(intensity=217.641, opacity=0.676471),
            OpacityControlPoint(intensity=419.736, opacity=0.833333),
            OpacityControlPoint(intensity=3071, opacity=0.803922),
        ],
        [
            ColorControlPoint(intensity=-3024, r=0, g=0, b=0),
            ColorControlPoint(intensity=-155.407, r=0.54902, g=0.25098, b=0.14902),
            ColorControlPoint(intensity=217.641, r=0.882353, g=0.603922, b=0.290196),
            ColorControlPoint(intensity=419.736, r=1, g=0.937033, b=0.954531),
            ColorControlPoint(intensity=3071, r=0.827451, g=0.658824, b=1),
        ],
    ),
    Preset.CT_PULMONARY_ARTERIES: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=-568.625, opacity=0),
            OpacityControlPoint(intensity=-364.081, opacity=0.0714286),
            OpacityControlPoint(intensity=-244.813, opacity=0.401786),
            OpacityControlPoint(intensity=18.2775, opacity=0.607143),
            OpacityControlPoint(intensity=447.798, opacity=0.830357),
            OpacityControlPoint(intensity=3592.73, opacity=0.839286),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=-568.625, r=0, g=0, b=0),
            ColorControlPoint(intensity=-364.081, r=0.396078, g=0.301961, b=0.180392),
            ColorControlPoint(intensity=-244.813, r=0.611765, g=0.352941, b=0.0705882),
            ColorControlPoint(intensity=18.2775, r=0.843137, g=0.0156863, b=0.156863),
            ColorControlPoint(intensity=447.798, r=0.752941, g=0.752941, b=0.752941),
            ColorControlPoint(intensity=3592.73, r=1, g=1, b=1),
        ],
    ),
    Preset.CT_SOFT_TISSUE: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=-167.01, opacity=0),
            OpacityControlPoint(intensity=-160, opacity=1),
            OpacityControlPoint(intensity=240, opacity=1),
            OpacityControlPoint(intensity=3661, opacity=1),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=-167.01, r=0, g=0, b=0),
            ColorControlPoint(intensity=-160, r=0.0556356, g=0.0556356, b=0.0556356),
            ColorControlPoint(intensity=240, r=1, g=1, b=1),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.MR_ANGIO: (
        [
            OpacityControlPoint(intensity=-2048, opacity=0),
            OpacityControlPoint(intensity=151.354, opacity=0),
            OpacityControlPoint(intensity=158.279, opacity=0.4375),
            OpacityControlPoint(intensity=190.112, opacity=0.580357),
            OpacityControlPoint(intensity=200.873, opacity=0.732143),
            OpacityControlPoint(intensity=3661, opacity=0.741071),
        ],
        [
            ColorControlPoint(intensity=-2048, r=0, g=0, b=0),
            ColorControlPoint(intensity=151.354, r=0, g=0, b=0),
            ColorControlPoint(intensity=158.279, r=0.74902, g=0.376471, b=0),
            ColorControlPoint(intensity=190.112, r=1, g=0.866667, b=0.733333),
            ColorControlPoint(intensity=200.873, r=0.937255, g=0.937255, b=0.937255),
            ColorControlPoint(intensity=3661, r=1, g=1, b=1),
        ],
    ),
    Preset.MR_DEFAULT: (
        [
            OpacityControlPoint(intensity=0, opacity=0),
            OpacityControlPoint(intensity=20, opacity=0),
            OpacityControlPoint(intensity=40, opacity=0.15),
            OpacityControlPoint(intensity=120, opacity=0.3),
            OpacityControlPoint(intensity=220, opacity=0.375),
            OpacityControlPoint(intensity=1024, opacity=0.5),
        ],
        [
            ColorControlPoint(intensity=0, r=0, g=0, b=0),
            ColorControlPoint(intensity=20, r=0.168627, g=0, b=0),
            ColorControlPoint(intensity=40, r=0.403922, g=0.145098, b=0.0784314),
            ColorControlPoint(intensity=120, r=0.780392, g=0.607843, b=0.380392),
            ColorControlPoint(intensity=220, r=0.847059, g=0.835294, b=0.788235),
            ColorControlPoint(intensity=1024, r=1, g=1, b=1),
        ],
    ),
    Preset.MR_MIP: (
        [
            OpacityControlPoint(intensity=0, opacity=0),
            OpacityControlPoint(intensity=98.3725, opacity=0),
            OpacityControlPoint(intensity=416.637, opacity=1),
            OpacityControlPoint(intensity=2800, opacity=1),
        ],
        [
            ColorControlPoint(intensity=0, r=1, g=1, b=1),
            ColorControlPoint(intensity=98.3725, r=1, g=1, b=1),
            ColorControlPoint(intensity=416.637, r=1, g=1, b=1),
            ColorControlPoint(intensity=2800, r=1, g=1, b=1),
        ],
    ),
    Preset.MR_T2_BRAIN: (
        [
            OpacityControlPoint(intensity=0, opacity=0),
            OpacityControlPoint(intensity=36.05, opacity=0),
            OpacityControlPoint(intensity=218.302, opacity=0.171429),
            OpacityControlPoint(intensity=412.406, opacity=1),
            OpacityControlPoint(intensity=641, opacity=1),
        ],
        [
            ColorControlPoint(intensity=0, r=0, g=0, b=0),
            ColorControlPoint(intensity=98.7223, r=0.956863, g=0.839216, b=0.192157),
            ColorControlPoint(intensity=412.406, r=0, g=0.592157, b=0.807843),
            ColorControlPoint(intensity=641, r=1, g=1, b=1),
        ],
    ),
    Preset.DTI_FA_BRAIN: (
        [
            OpacityControlPoint(intensity=0, opacity=0),
            OpacityControlPoint(intensity=0, opacity=0),
            OpacityControlPoint(intensity=0.3501, opacity=0.0158),
            OpacityControlPoint(intensity=0.49379, opacity=0.7619),
            OpacityControlPoint(intensity=0.6419, opacity=1),
            OpacityControlPoint(intensity=0.992, opacity=1),
            OpacityControlPoint(intensity=0.995, opacity=0),
            OpacityControlPoint(intensity=0.995, opacity=0),
        ],
        [
            ColorControlPoint(intensity=0, r=1, g=0, b=0),
            ColorControlPoint(intensity=0, r=1, g=0, b=0),
            ColorControlPoint(intensity=0.24974, r=0.4941, g=1, b=0),
            ColorControlPoint(intensity=0.49949, r=0, g=0.9882, b=1),
            ColorControlPoint(intensity=0.7492, r=0.51764, g=0, b=1),
            ColorControlPoint(intensity=0.995, r=1, g=0, b=0),
            ColorControlPoint(intensity=0.995, r=1, g=0, b=0),
        ],
    ),
}
