# 역할: 원본 사본의 박스·랜드마크 및 얼굴 모음 이미지 생성과 HTML 생성 호출
"""Local-only HTML, full oriented copies, annotated images and paged contact sheets."""
from collections import defaultdict
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size):
    # Only known Korean fonts are considered Korean-capable; otherwise ASCII IDs.
    for path in (Path("C:/Windows/Fonts/malgun.ttf"), Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size), True
    return ImageFont.load_default(size=size), False


def display_label(f, korean):
    return f["label"] if korean else (f["group_id"] or "unknown")


def render(run, photos, faces, options=None):
    text_font, korean = font(16)
    (run / 'geometry').mkdir(exist_ok=True)
    by_photo, groups = defaultdict(list), defaultdict(list)
    for f in faces:
        by_photo[f["photo_id"]].append(f)
        groups[f["group_id"] or "unknown"].append(f)
    for p in photos:
        src = run / "photos" / (p["photo_id"] + ".jpg")
        if not src.exists():
            continue
        with Image.open(src) as im:
            image = im.convert("RGB")
        draw = ImageDraw.Draw(image)
        image_font, _ = font(max(16, round(max(image.size)/100)))
        for f in by_photo[p["photo_id"]]:
            x, y, w, h = [f[k] for k in ("x", "y", "width", "height")]
            if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h) <= 0:
                continue
            color = "orange" if f["status"] == "unknown" else "lime"
            draw.rectangle((x, y, x+w, y+h), outline=color, width=max(2, image.width//500))
            points = f.get('landmarks', [])
            if isinstance(points, str):
                try:
                    points = json.loads(points)
                except (ValueError, TypeError):
                    points = []
            for n in range(0, len(points)-1, 2):
                px, py = points[n:n+2]
                if math.isfinite(px) and math.isfinite(py):
                    r = max(2, image.width//700)
                    draw.ellipse((px-r, py-r, px+r, py+r), fill=('red','cyan','yellow','magenta','blue')[(n//2)%5])
            label = f"{f['face_id']} {display_label(f, korean)}"
            draw.text((max(0, x), max(0, y-24)), label, font=image_font, fill=color, stroke_width=1, stroke_fill="black")
        for f in by_photo[p['photo_id']]:
            x,y,w,h = [f[k] for k in ('x','y','width','height')]
            if all(math.isfinite(v) for v in (x,y,w,h)) and min(w,h)>0:
                region = (max(0,int(x-.25*w)),max(0,int(y-.3*h)),min(image.width,int(x+1.25*w)+1),min(image.height,int(y+1.3*h)+1))
                if region[2] > region[0] and region[3] > region[1]:
                    ImageOps.contain(image.crop(region), (300,300)).save(run/'geometry'/f"{f['face_id']}.jpg", quality=92)
        image.save(run / "annotated" / (p["photo_id"] + ".jpg"), quality=92)
    for gid, members in sorted(groups.items()):
        for offset in range(0, len(members), 80):
            batch = members[offset:offset+80]
            cols = min(5, len(batch))
            sheet = Image.new("RGB", (cols*240, ((len(batch)+cols-1)//cols)*164), "white")
            draw = ImageDraw.Draw(sheet)
            for n, f in enumerate(batch):
                x, y = (n % cols)*240, (n//cols)*164
                if f["crop_path"] and (run / f["crop_path"]).exists():
                    with Image.open(run / f["crop_path"]) as crop:
                        sheet.paste(ImageOps.contain(crop.convert("RGB"), (112,112)), (x,y))
                else:
                    draw.text((x,y), "no aligned crop", font=text_font, fill="black")
                draw.text((x,y+114), f["photo_id"], font=text_font, fill="black")
                draw.text((x,y+136), f["face_id"], font=text_font, fill="black")
            sheet.save(run / f"montages/{gid}_{offset//80+1:03d}.jpg", quality=92)
    from .report_ui import render_html
    render_html(run, photos, faces, options)
