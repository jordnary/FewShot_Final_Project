# -*- coding: utf-8 -*-
"""Generate a polished overview poster for the project demo."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "demo" / "assets" / "project_overview.png"
WIDTH, HEIGHT = 1920, 1080


PALETTE = {
    "bg_top": (248, 250, 248),
    "bg_bottom": (232, 241, 239),
    "ink": (26, 34, 44),
    "muted": (91, 106, 121),
    "line": (207, 219, 222),
    "card": (255, 255, 255),
    "green": (30, 115, 96),
    "teal": (21, 140, 143),
    "blue": (47, 111, 187),
    "orange": (217, 121, 61),
    "purple": (108, 92, 170),
    "red": (184, 74, 98),
    "soft_green": (226, 243, 236),
    "soft_blue": (230, 239, 253),
    "soft_orange": (253, 238, 226),
    "soft_purple": (238, 235, 250),
}


def font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/msyhbd.ttc"),
                Path("C:/Windows/Fonts/segoeuib.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONTS = {
    "eyebrow": font(21, True),
    "title": font(62, True),
    "subtitle": font(25),
    "section": font(31, True),
    "label": font(24, True),
    "body": font(22),
    "small": font(18),
    "tiny": font(15),
    "stat": font(56, True),
    "stat_small": font(28, True),
}


def lerp(a, b, t):
    return int(a + (b - a) * t)


def draw_gradient_background(image):
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        t = y / (HEIGHT - 1)
        color = tuple(lerp(PALETTE["bg_top"][i], PALETTE["bg_bottom"][i], t) for i in range(3))
        draw.line([(0, y), (WIDTH, y)], fill=color)

    grid_color = (207, 219, 222, 72)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for x in range(64, WIDTH, 96):
        odraw.line([(x, 0), (x, HEIGHT)], fill=grid_color, width=1)
    for y in range(64, HEIGHT, 96):
        odraw.line([(0, y), (WIDTH, y)], fill=grid_color, width=1)

    # A faint ViT patch-token grid motif.
    start_x, start_y = 1500, 78
    cell = 22
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["green"]]
    for row in range(8):
        for col in range(10):
            idx = (row * 3 + col * 5) % len(colors)
            alpha = 22 + ((row + col) % 5) * 10
            x = start_x + col * (cell + 7)
            y = start_y + row * (cell + 7)
            odraw.rounded_rectangle(
                [x, y, x + cell, y + cell],
                radius=5,
                fill=(*colors[idx], alpha),
            )
    image.alpha_composite(overlay)


def text_bbox(draw, xy, text, font_obj):
    return draw.textbbox(xy, text, font=font_obj)


def text_width(draw, text, font_obj):
    box = text_bbox(draw, (0, 0), text, font_obj)
    return box[2] - box[0]


def card(image, xy, radius=28, fill=None, outline=None, shadow=True):
    x1, y1, x2, y2 = xy
    if fill is None:
        fill = PALETTE["card"]
    if shadow:
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_layer)
        sdraw.rounded_rectangle(
            [x1 + 8, y1 + 12, x2 + 8, y2 + 12],
            radius=radius,
            fill=(18, 45, 50, 34),
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(14))
        image.alpha_composite(shadow_layer)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline or PALETTE["line"], width=1)


def wrapped_lines(draw, text, font_obj, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if text_width(draw, trial, font_obj) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
            if text_width(draw, word, font_obj) <= max_width:
                current = word
            else:
                current = ""
                chunk = ""
                for char in word:
                    trial_chunk = f"{chunk}{char}"
                    if text_width(draw, trial_chunk, font_obj) <= max_width:
                        chunk = trial_chunk
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = char
                current = chunk
        if current:
            lines.append(current)
    return lines


def draw_wrapped(draw, xy, text, font_obj, fill, max_width, line_gap=8):
    x, y = xy
    for line in wrapped_lines(draw, text, font_obj, max_width):
        draw.text((x, y), line, font=font_obj, fill=fill)
        box = text_bbox(draw, (x, y), line, font_obj)
        y += box[3] - box[1] + line_gap
    return y


def draw_badge(draw, x, y, text, fill, fg, pad_x=16, pad_y=7):
    w = text_width(draw, text, FONTS["small"]) + pad_x * 2
    h = 34
    draw.rounded_rectangle([x, y, x + w, y + h], radius=17, fill=fill)
    draw.text((x + pad_x, y + pad_y - 1), text, font=FONTS["small"], fill=fg)
    return x + w + 12


def draw_arrow(draw, x, y1, y2, color):
    draw.line([(x, y1), (x, y2 - 12)], fill=color, width=5)
    draw.polygon([(x, y2), (x - 10, y2 - 16), (x + 10, y2 - 16)], fill=color)


def draw_header(draw):
    draw.text((72, 54), "FEW-SHOT CLASSIFICATION / CUB-200", font=FONTS["eyebrow"], fill=PALETTE["green"])
    draw.text((72, 88), "Few-Shot Learning 项目总览", font=FONTS["title"], fill=PALETTE["ink"])
    draw.text(
        (76, 168),
        "LibFewShot ProtoNet baseline · FroFA reproduction · CLIP ViT-B/16 frozen patch-token improvement",
        font=FONTS["subtitle"],
        fill=PALETTE["muted"],
    )
    x = 76
    for text, fill, fg in [
        ("5-way 1-shot", PALETTE["soft_blue"], PALETTE["blue"]),
        ("5-way 5-shot", PALETTE["soft_orange"], PALETTE["orange"]),
        ("Frozen features", PALETTE["soft_green"], PALETTE["green"]),
        ("Validation sweep", PALETTE["soft_purple"], PALETTE["purple"]),
    ]:
        x = draw_badge(draw, x, 206, text, fill, fg)


def draw_pipeline(image):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = 64, 270, 646, 1002
    card(image, [x1, y1, x2, y2], radius=24)
    draw.text((100, 304), "实验路线", font=FONTS["section"], fill=PALETTE["ink"])
    draw.text((102, 348), "From reproducible baseline to stronger frozen representation.", font=FONTS["small"], fill=PALETTE["muted"])

    nodes = [
        ("01", "CUB Episodes", "CUB 细粒度鸟类图像；统一 5-way 1-shot / 5-shot episodic evaluation.", PALETTE["blue"]),
        ("02", "ProtoNet Baseline", "在 LibFewShot 中建立 Conv64F 与 ResNet12 baseline，形成后续对照.", PALETTE["teal"]),
        ("03", "FroFA Reproduction", "接入 FroFA-style support feature augmentation，比较 joint / frozen 设置.", PALETTE["orange"]),
        ("04", "CLIP-FroFA Improvement", "使用 CLIP ViT-B/16 frozen global features 与 post-LN patch tokens.", PALETTE["green"]),
    ]
    node_x = 118
    content_x = 176
    top = 410
    gap = 120
    for idx, (num, title, body, color) in enumerate(nodes):
        cy = top + idx * gap
        if idx < len(nodes) - 1:
            draw_arrow(draw, node_x, cy + 28, cy + gap - 34, color)
        draw.ellipse([node_x - 27, cy - 27, node_x + 27, cy + 27], fill=color)
        num_w = text_width(draw, num, FONTS["small"])
        draw.text((node_x - num_w / 2, cy - 12), num, font=FONTS["small"], fill=(255, 255, 255))
        draw.text((content_x, cy - 34), title, font=FONTS["label"], fill=PALETTE["ink"])
        draw_wrapped(draw, (content_x, cy), body, FONTS["small"], PALETTE["muted"], 390, line_gap=4)

    draw.rounded_rectangle([100, 895, 610, 958], radius=18, fill=(242, 247, 246), outline=PALETTE["line"], width=1)
    draw.text((126, 912), "Artifacts", font=FONTS["label"], fill=PALETTE["ink"])
    draw.text((248, 916), "configs · scripts · results · report · demo", font=FONTS["small"], fill=PALETTE["muted"])


def draw_accuracy_chart(image):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = 690, 270, 1856, 704
    card(image, [x1, y1, x2, y2], radius=24)
    draw.text((730, 304), "Accuracy Landscape (%)", font=FONTS["section"], fill=PALETTE["ink"])
    draw.text((732, 348), "Key submitted results across the main experiment lines.", font=FONTS["small"], fill=PALETTE["muted"])

    legend_y = 310
    draw.rounded_rectangle([1522, legend_y + 2, 1544, legend_y + 24], radius=5, fill=PALETTE["blue"])
    draw.text((1554, legend_y), "1-shot", font=FONTS["small"], fill=PALETTE["muted"])
    draw.rounded_rectangle([1638, legend_y + 2, 1660, legend_y + 24], radius=5, fill=PALETTE["orange"])
    draw.text((1670, legend_y), "5-shot", font=FONTS["small"], fill=PALETTE["muted"])

    data = [
        ("ProtoNet\nResNet12", 73.376, 85.945),
        ("CLIP global\nno-FroFA", 86.975, 96.379),
        ("Post-LN\nMAP", 42.187, 68.878),
        ("Post-LN\nFroFA+MAP", 43.960, 72.893),
    ]
    chart_left, chart_top = 932, 406
    chart_w = 792
    row_gap = 68
    bar_h = 19
    for tick in [0, 25, 50, 75, 100]:
        tx = chart_left + chart_w * tick / 100
        draw.line([(tx, chart_top - 26), (tx, chart_top + row_gap * 3 + 46)], fill=(218, 228, 230), width=1)
        tick_text = str(tick)
        draw.text((tx - text_width(draw, tick_text, FONTS["tiny"]) / 2, chart_top + row_gap * 3 + 56), tick_text, font=FONTS["tiny"], fill=PALETTE["muted"])

    for idx, (label, one, five) in enumerate(data):
        cy = chart_top + idx * row_gap
        label_lines = label.split("\n")
        draw.text((732, cy - 18), label_lines[0], font=FONTS["body"], fill=PALETTE["ink"])
        draw.text((732, cy + 8), label_lines[1], font=FONTS["small"], fill=PALETTE["muted"])
        for offset, value, color in [(-13, one, PALETTE["blue"]), (14, five, PALETTE["orange"])]:
            by = cy + offset
            bw = chart_w * value / 100
            draw.rounded_rectangle([chart_left, by, chart_left + bw, by + bar_h], radius=10, fill=color)
            value_text = f"{value:.1f}"
            draw.text((chart_left + bw + 10, by - 4), value_text, font=FONTS["small"], fill=PALETTE["ink"])

    draw.line([(chart_left, chart_top - 26), (chart_left, chart_top + row_gap * 3 + 46)], fill=PALETTE["line"], width=2)


def draw_takeaway(image):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = 690, 734, 1856, 1002
    card(image, [x1, y1, x2, y2], radius=24)
    draw.text((730, 768), "Main Takeaway", font=FONTS["section"], fill=PALETTE["ink"])
    draw.text(
        (732, 812),
        "Best final route: post-LN patch tokens + validation-selected brightness c2FroFA + MAP head.",
        font=FONTS["small"],
        fill=PALETTE["muted"],
    )

    stat_blocks = [
        (730, 854, "+1.773 pp", "1-shot gain", "42.187 → 43.960", PALETTE["blue"], PALETTE["soft_blue"]),
        (1038, 854, "+4.016 pp", "5-shot gain", "68.878 → 72.893", PALETTE["orange"], PALETTE["soft_orange"]),
    ]
    for sx, sy, stat, label, detail, color, fill in stat_blocks:
        draw.rounded_rectangle([sx, sy, sx + 270, sy + 110], radius=20, fill=fill, outline=PALETTE["line"], width=1)
        draw.text((sx + 22, sy + 14), stat, font=FONTS["stat_small"], fill=color)
        draw.text((sx + 24, sy + 55), label, font=FONTS["body"], fill=PALETTE["ink"])
        draw.text((sx + 24, sy + 82), detail, font=FONTS["small"], fill=PALETTE["muted"])

    cfg_x, cfg_y = 1376, 846
    draw.rounded_rectangle([cfg_x, cfg_y, 1814, 966], radius=20, fill=(244, 248, 247), outline=PALETTE["line"], width=1)
    draw.text((cfg_x + 24, cfg_y + 18), "Selected config", font=FONTS["label"], fill=PALETTE["ink"])
    config_text = "alpha=0.30 · num_aug=8 · brightness\ntrain_steps=40 · weight_decay=0.01"
    draw_wrapped(draw, (cfg_x + 24, cfg_y + 56), config_text, FONTS["small"], PALETTE["muted"], 382, line_gap=4)


def draw_footer(draw):
    draw.text(
        (72, 1030),
        "Source: experiments/*/results/*.csv · report/tables/*.md · Generated for demo/assets/project_overview.png",
        font=FONTS["tiny"],
        fill=(95, 111, 123),
    )


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw_gradient_background(image)
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    draw_pipeline(image)
    draw_accuracy_chart(image)
    draw_takeaway(image)
    draw_footer(draw)
    image.convert("RGB").save(OUT_PATH, "PNG", optimize=True)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
