from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DashboardConfig:
    canvas_w: int = 2600
    canvas_h: int = 1800
    pad: int = 28
    header_h: int = 70
    panel_title_h: int = 52
    border: int = 2
    bg: Tuple[int, int, int] = (255, 255, 255)
    border_color: Tuple[int, int, int] = (40, 40, 40)
    text_color: Tuple[int, int, int] = (0, 0, 0)

def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _fit_image(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    tw, th = target
    img = img.convert("RGB")
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        raise ValueError("Invalid image size")

    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = img.resize((nw, nh), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas.paste(resized, (x0, y0))
    return canvas

def _draw_panel_frame(
    canvas: Image.Image,
    xy: Tuple[int, int],
    size: Tuple[int, int],
    title: str,
    cfg: DashboardConfig,
) -> Tuple[int, int, int, int]:
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([x, y, x + w, y + h], outline=cfg.border_color, width=cfg.border)

    title_font = _load_font(20)
    draw.text((x + 14, y + 14), title, fill=cfg.text_color, font=title_font)

    img_x = x + cfg.pad // 2
    img_y = y + cfg.panel_title_h
    img_w = w - cfg.pad
    img_h = h - cfg.panel_title_h - (cfg.pad // 2)
    return img_x, img_y, img_w, img_h

def _paste_single_image_panel(
    canvas: Image.Image,
    xy: Tuple[int, int],
    size: Tuple[int, int],
    title: str,
    img_path: Path,
    cfg: DashboardConfig,
) -> None:
    if not img_path.exists():
        raise FileNotFoundError(f"Missing required image: {img_path}")

    img_x, img_y, img_w, img_h = _draw_panel_frame(canvas, xy, size, title, cfg)

    img = Image.open(img_path)
    fitted = _fit_image(img, (img_w, img_h))
    canvas.paste(fitted, (img_x, img_y))

def _paste_side_by_side_panel(
    canvas: Image.Image,
    xy: Tuple[int, int],
    size: Tuple[int, int],
    title: str,
    left_title: str,
    left_path: Path,
    right_title: str,
    right_path: Path,
    cfg: DashboardConfig,
) -> None:
    if not left_path.exists():
        raise FileNotFoundError(f"Missing required image: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Missing required image: {right_path}")

    img_x, img_y, img_w, img_h = _draw_panel_frame(canvas, xy, size, title, cfg)

    gap = cfg.pad // 2
    half_w = (img_w - gap) // 2
    sub_h = img_h

    left_img = _fit_image(Image.open(left_path), (half_w, sub_h))
    right_img = _fit_image(Image.open(right_path), (img_w - gap - half_w, sub_h))

    canvas.paste(left_img, (img_x, img_y))
    canvas.paste(right_img, (img_x + half_w + gap, img_y))

    draw = ImageDraw.Draw(canvas)
    sub_font = _load_font(18)
    draw.text((img_x + 10, img_y + 10), left_title, fill=(0, 0, 0), font=sub_font)
    draw.text((img_x + half_w + gap + 10, img_y + 10), right_title, fill=(0, 0, 0), font=sub_font)

def main() -> None:
    cfg = DashboardConfig()

    out_dir = Path("outputs/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = Path("outputs/phase3/confusion_matrix.png")
    pred_path = Path("outputs/phase3/prediction_gallery_all.png")
    mis_path = Path("outputs/phase3/misclassification_gallery.png")
    gradcam_path = Path("outputs/phase4/gradcam_gallery.png")
    overlay_path = Path("outputs/phase5/landuse_overlay_map.png")

    required = [cm_path, pred_path, mis_path, gradcam_path, overlay_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required images: {missing}")

    canvas = Image.new("RGB", (cfg.canvas_w, cfg.canvas_h), cfg.bg)
    draw = ImageDraw.Draw(canvas)

    header_font = _load_font(30)
    draw.text(
        (cfg.pad, 18),
        "Satellite Land Use Classification Output Dashboard",
        fill=cfg.text_color,
        font=header_font,
    )

    grid_x0 = cfg.pad
    grid_y0 = cfg.header_h
    grid_w = cfg.canvas_w - cfg.pad * 2
    grid_h = cfg.canvas_h - cfg.header_h - cfg.pad

    gap = cfg.pad
    panel_w = (grid_w - gap) // 2
    panel_h = (grid_h - gap) // 2

    p1_xy = (grid_x0, grid_y0)
    p2_xy = (grid_x0 + panel_w + gap, grid_y0)
    p3_xy = (grid_x0, grid_y0 + panel_h + gap)
    p4_xy = (grid_x0 + panel_w + gap, grid_y0 + panel_h + gap)

    panel_size = (panel_w, panel_h)

    _paste_single_image_panel(
        canvas=canvas,
        xy=p1_xy,
        size=panel_size,
        title="Confusion Matrix",
        img_path=cm_path,
        cfg=cfg,
    )

    _paste_single_image_panel(
        canvas=canvas,
        xy=p2_xy,
        size=panel_size,
        title="Prediction Gallery",
        img_path=pred_path,
        cfg=cfg,
    )

    _paste_single_image_panel(
        canvas=canvas,
        xy=p3_xy,
        size=panel_size,
        title="Misclassification Gallery",
        img_path=mis_path,
        cfg=cfg,
    )

    _paste_side_by_side_panel(
        canvas=canvas,
        xy=p4_xy,
        size=panel_size,
        title="Model Focus and Land Use Map",
        left_title="GradCAM",
        left_path=gradcam_path,
        right_title="Overlay Map",
        right_path=overlay_path,
        cfg=cfg,
    )

    out_path = out_dir / "final_dashboard.png"
    canvas.save(out_path)

    print("\nDASHBOARD COMPLETE")
    print(f"Saved: {out_path}\n")

if __name__ == "__main__":
    main()
