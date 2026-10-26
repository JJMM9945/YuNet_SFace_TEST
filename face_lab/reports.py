"""Local-only HTML, full oriented copies, annotated images and paged contact sheets."""
from collections import defaultdict
from html import escape
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


def render(run, photos, faces):
    text_font, korean = font(16)
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
            color = "orange" if f["status"] == "unknown" else "lime"
            draw.rectangle((x, y, x+w, y+h), outline=color, width=max(2, image.width//500))
            label = f"{f['face_id']} {display_label(f, korean)}"
            draw.text((max(0, x), max(0, y-24)), label, font=image_font, fill=color, stroke_width=1, stroke_fill="black")
        image.save(run / "annotated" / (p["photo_id"] + ".jpg"), quality=92)
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
            '<title>Local face review</title><style>body{font:16px sans-serif;margin:24px;background:#fafafa;color:#222}img{max-width:100%}.faces{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;padding:8px;background:white;width:180px}figure img{width:112px}a{color:#135da0}td{padding:6px;border-bottom:1px solid #ddd}</style>',
            '<h1>로컬 인물 검토</h1><p>번호는 이번 실행의 임시 그룹입니다. 검출 수·그룹 수는 정확도가 아닙니다. 주황색은 미확인입니다.</p>',
            '<p><a href="summary.md">요약</a> · <a href="faces.csv">얼굴 CSV</a> · <a href="photos.csv">사진 CSV</a> · <a href="ground_truth_template.csv">정답 양식</a> · <a href="photo_review_template.csv">미검출 검토 양식</a></p>']
    if any(f.get('label_origin','').startswith(('user_','manual')) for f in faces):
        page.append('<p><strong>사용자 확인을 반영한 결과입니다.</strong> 명시적으로 확인한 얼굴만 수정했으며 나머지 자동 그룹은 미검토입니다. 모델 재학습이나 자동 정확도 향상을 의미하지 않습니다.</p>')
    for gid, members in sorted(groups.items()):
        title = members[0]["label"] if gid != "unknown" else "미확인"
        page.append(f'<h2>{escape(title)} ({len(members)})</h2>')
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
            sheet_path = f"montages/{gid}_{offset//80+1:03d}.jpg"
            sheet.save(run / sheet_path, quality=92)
            page.append(f'<p><a href="{sheet_path}">얼굴 모음 {offset//80+1}</a></p><img src="{sheet_path}" alt="contact sheet">')
        page.append('<div class="faces">')
        for f in members:
            pid, fid = f["photo_id"], f["face_id"]
            # URLs contain generated ASCII IDs only, never user paths or filenames.
            image = f'<img src="{f["crop_path"]}" alt="{fid}">' if f["crop_path"] else '<p>정렬 실패</p>'
            page.append(f'<figure>{image}<figcaption>{fid}<br>{escape(f["relative_path"])}<br>{escape(f["reason"])}<br>{escape(f["review"])}<br>'
                        f'{escape(f.get("manual_person_id", ""))} {escape(f.get("label_origin", ""))}<br>'
                        f'<a href="photos/{pid}.jpg">원본 내용 사본</a> · <a href="annotated/{pid}.jpg">검출 확인</a></figcaption></figure>')
        page.append('</div>')
    page.append('<h2>모든 입력 사진 (미검출·실패 포함)</h2><table><tr><th>사진</th><th>상태</th><th>얼굴 수</th><th>인물</th></tr>')
    for p in photos:
        pid = p["photo_id"]
        link = f'<a href="annotated/{pid}.jpg">{escape(p["relative_path"])}</a>' if (run / "annotated" / (pid+".jpg")).exists() else escape(p["relative_path"])
        page.append(f'<tr><td>{link}</td><td>{escape(p["status"])} {escape(p["error"])}</td><td>{p["detected_faces"]}</td><td>{escape(p["people"])}</td></tr>')
    page.append('</table></html>')
    (run / "index.html").write_text("\n".join(page), encoding="utf-8")
