"""Reliability diagram SVG rendering, no external dependencies (design.md 4.3 ece; REQ-040)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from omj.bench.types import ResultRow, is_correct

SVG_WIDTH = 480
SVG_HEIGHT = 360

_MARGIN_LEFT = 56
_MARGIN_RIGHT = 20
_MARGIN_TOP = 30
_MARGIN_BOTTOM = 56

_PLOT_LEFT = _MARGIN_LEFT
_PLOT_RIGHT = SVG_WIDTH - _MARGIN_RIGHT
_PLOT_TOP = _MARGIN_TOP
_PLOT_BOTTOM = SVG_HEIGHT - _MARGIN_BOTTOM
_PLOT_WIDTH = _PLOT_RIGHT - _PLOT_LEFT
_PLOT_HEIGHT = _PLOT_BOTTOM - _PLOT_TOP


def _x(fraction: float) -> float:
    return _PLOT_LEFT + fraction * _PLOT_WIDTH


def _y(fraction: float) -> float:
    return _PLOT_BOTTOM - fraction * _PLOT_HEIGHT


def bins_from_rows(rows: list[ResultRow], n_bins: int = 15) -> list[tuple[float, float, int]]:
    """Bucket scored rows into `n_bins` equal-width confidence bins.

    Mirrors the binning rule in metrics.top_label_ece: bins are half-open
    except the last, which is closed on both ends so confidence == 1.0 lands
    in the final bin. Rows without an expected label or probability vector
    are excluded. Returns (mean_confidence, accuracy, count) per non-empty
    bin, in bin order.
    """
    scored = [r for r in rows if r.expected is not None and r.probs]
    confidences = [max(r.probs) for r in scored]  # type: ignore[arg-type]
    corrects = [bool(is_correct(r)) for r in scored]

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, correct in zip(confidences, corrects):
        idx = min(int(confidence * n_bins), n_bins - 1)
        buckets[idx].append((confidence, correct))

    result: list[tuple[float, float, int]] = []
    for bucket in buckets:
        if not bucket:
            continue
        bucket_confidences = [c for c, _ in bucket]
        bucket_corrects = [c for _, c in bucket]
        mean_confidence = sum(bucket_confidences) / len(bucket_confidences)
        accuracy = sum(1.0 for c in bucket_corrects if c) / len(bucket_corrects)
        result.append((mean_confidence, accuracy, len(bucket)))
    return result


def reliability_svg(bins: list[tuple[float, float, int]], title: str = "Reliability") -> str:
    """Render a confidence-vs-accuracy reliability diagram as a standalone SVG string.

    One <rect> is emitted per bin, sized so bar width is proportional to
    1/len(bins); no rects are emitted for anything else, so a caller can
    count <rect> elements to verify one-per-bin.
    """
    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(SVG_WIDTH),
            "height": str(SVG_HEIGHT),
            "viewBox": f"0 0 {SVG_WIDTH} {SVG_HEIGHT}",
        },
    )

    ET.SubElement(
        svg,
        "text",
        {
            "x": f"{SVG_WIDTH / 2:.2f}",
            "y": "18",
            "text-anchor": "middle",
            "font-size": "14",
            "fill": "#1a1a1a",
        },
    ).text = title

    ET.SubElement(
        svg,
        "line",
        {
            "x1": f"{_x(0):.2f}",
            "y1": f"{_y(0):.2f}",
            "x2": f"{_x(1):.2f}",
            "y2": f"{_y(0):.2f}",
            "stroke": "#333333",
            "stroke-width": "1",
        },
    )
    ET.SubElement(
        svg,
        "line",
        {
            "x1": f"{_x(0):.2f}",
            "y1": f"{_y(0):.2f}",
            "x2": f"{_x(0):.2f}",
            "y2": f"{_y(1):.2f}",
            "stroke": "#333333",
            "stroke-width": "1",
        },
    )

    ET.SubElement(
        svg,
        "line",
        {
            "x1": f"{_x(0):.2f}",
            "y1": f"{_y(0):.2f}",
            "x2": f"{_x(1):.2f}",
            "y2": f"{_y(1):.2f}",
            "stroke": "#999999",
            "stroke-width": "1",
            "stroke-dasharray": "4,4",
        },
    )

    n_bins = len(bins) if bins else 1
    bar_width = _PLOT_WIDTH / n_bins
    for i, (mean_confidence, accuracy, count) in enumerate(bins):
        bar_x = _PLOT_LEFT + i * bar_width
        bar_top = _y(max(0.0, min(1.0, accuracy)))
        bar_height = max(_PLOT_BOTTOM - bar_top, 0.0)
        ET.SubElement(
            svg,
            "rect",
            {
                "x": f"{bar_x:.2f}",
                "y": f"{bar_top:.2f}",
                "width": f"{max(bar_width - 2, 1.0):.2f}",
                "height": f"{bar_height:.2f}",
                "fill": "#4c78a8",
                "data-mean-confidence": f"{mean_confidence:.4f}",
                "data-count": str(count),
            },
        )
        label_y = bar_top - 4 if bar_top - 4 > _PLOT_TOP else bar_top + 10
        ET.SubElement(
            svg,
            "text",
            {
                "x": f"{bar_x + bar_width / 2:.2f}",
                "y": f"{label_y:.2f}",
                "text-anchor": "middle",
                "font-size": "9",
                "fill": "#1a1a1a",
            },
        ).text = str(count)

    for frac in (0.0, 0.5, 1.0):
        ET.SubElement(
            svg,
            "text",
            {
                "x": f"{_x(frac):.2f}",
                "y": f"{_y(0) + 16:.2f}",
                "text-anchor": "middle",
                "font-size": "10",
                "fill": "#1a1a1a",
            },
        ).text = f"{frac:.1f}"
        ET.SubElement(
            svg,
            "text",
            {
                "x": f"{_x(0) - 8:.2f}",
                "y": f"{_y(frac) + 3:.2f}",
                "text-anchor": "end",
                "font-size": "10",
                "fill": "#1a1a1a",
            },
        ).text = f"{frac:.1f}"

    ET.SubElement(
        svg,
        "text",
        {
            "x": f"{(_PLOT_LEFT + _PLOT_RIGHT) / 2:.2f}",
            "y": f"{SVG_HEIGHT - 8:.2f}",
            "text-anchor": "middle",
            "font-size": "10",
            "fill": "#1a1a1a",
        },
    ).text = "Confidence"
    ET.SubElement(
        svg,
        "text",
        {
            "x": "14",
            "y": f"{(_PLOT_TOP + _PLOT_BOTTOM) / 2:.2f}",
            "text-anchor": "middle",
            "font-size": "10",
            "fill": "#1a1a1a",
            "transform": f"rotate(-90 14 {(_PLOT_TOP + _PLOT_BOTTOM) / 2:.2f})",
        },
    ).text = "Accuracy"

    return ET.tostring(svg, encoding="unicode")
