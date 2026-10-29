"""Local-only HTML, full oriented copies, annotated images and paged contact sheets."""
from collections import defaultdict
import json
import math
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
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
            '<title>Local face review</title><style>body{font:16px sans-serif;margin:24px;background:#fafafa;color:#222}img{max-width:100%}.faces{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;padding:8px;background:white;width:180px}figure img{width:112px}figure{width:300px}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure:target{outline:3px solid blue}a{color:#135da0}td{padding:6px;border-bottom:1px solid #ddd}</style>',
            '<h1>로컬 인물 검토</h1><p>번호는 이번 실행의 임시 그룹입니다. 검출 수·그룹 수는 정확도가 아닙니다. 주황색은 미확인입니다.</p>',
            '<p><a href="summary.md">요약</a> · <a href="faces.csv">얼굴 CSV</a> · <a href="photos.csv">사진 CSV</a> · <a href="ground_truth_template.csv">정답 양식</a> · <a href="photo_review_template.csv">미검출 검토 양식</a></p>']
    if (run/'diagnostics.json').exists():
        page.append('<p><a href="diagnostics.json">전체 단계 진단 JSON</a> · <a href="diagnostics.csv">진단 CSV</a></p>')
    if any(f.get('label_origin','').startswith(('user_','manual')) for f in faces):
        page.append('<p><strong>사용자 확인을 반영한 결과입니다.</strong> 명시적으로 확인한 얼굴만 수정했으며 나머지 자동 그룹은 미검토입니다. 모델 재학습이나 자동 정확도 향상을 의미하지 않습니다.</p>')
    for gid, members in sorted(groups.items()):
        title = members[0]["label"] if gid != "unknown" else "미확인"
        page.append(f'<h2 id="group-{escape(gid)}">{escape(title)} ({len(members)})</h2>')
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
            d = f.get('diagnostic', {})
            diagnostic = ''
            if d:
                def candidates_html(cs):
                    lines = []
                    for c in cs[:3]:
                        gid = c['group_id']
                        score = c['score']
                        exists = any(other['group_id'] == gid for other in faces)
                        dest = f'#group-{gid}' if exists else '#'+c.get('face_ids',[''])[0]
                        suffix = '' if exists else ' (해체된 초기 그룹)'
                        lines.append(f'<a href="{escape(dest)}">{escape(gid)}</a>: {score:.5f}{suffix}')
                    return '<br>'.join(lines) or '후보 없음'
                diagnostic = f"<p>상태: {escape(d['state'])}; 크기 {d['face_size']:.1f}px; 검출 {d['detection_score']}; 흐림 {d['blur_variance']}</p>"
                diagnostic += '<p>초기 그룹 점수 ('+escape(d['group_score_definition'])+')<br>'+candidates_html(d['candidates'])+'</p>'
                diagnostic += f"<p>후보 차이: {d['candidate_gap']}; 임계값: {d['score_threshold']}; 폭: {d['margin_threshold']}<br>자기 그룹−경쟁 그룹: {d['own_competitor_gap']}</p>"
                nearest = d['nearest_face_id']
                diagnostic += f'<p>가장 가까운 개별 얼굴: <a href="#{escape(nearest or "")}">{escape(nearest or "없음")}</a>; cosine={d["nearest_cosine_similarity"]}</p>'
                r = d.get('reassignment', {})
                if 'candidates' in r:
                    diagnostic += '<p><strong>재배정: '+escape(r['reason'])+'</strong><br>'+candidates_html(r['candidates'])+f"<br>차이 {r['candidate_gap']}; 기준 {r['score_threshold']}; 폭 {r['margin_threshold']}</p>"
                diagnostic += '<details><summary>단계별 판정·전체 후보</summary><pre>'+escape(json.dumps(d, ensure_ascii=False, indent=2))+'</pre></details>'
            geometry = f'<img src="geometry/{fid}.jpg" alt="box and five landmarks">' if (run/'geometry'/f'{fid}.jpg').exists() else ''
            page.append(f'<figure id="{fid}">{geometry}{image}<figcaption>{fid}<br>{escape(f["relative_path"])}<br>{escape(f["reason"])}<br>{escape(f["review"])}<br>'
                        f'{escape(f.get("manual_person_id", ""))} {escape(f.get("label_origin", ""))}<br>'
                        f'{diagnostic}<a href="photos/{pid}.jpg">원본 내용 사본</a> · <a href="annotated/{pid}.jpg">검출 확인</a></figcaption></figure>')
        page.append('</div>')
    page.append('<h2>모든 입력 사진 (미검출·실패 포함)</h2><table><tr><th>사진</th><th>상태</th><th>얼굴 수</th><th>인물</th></tr>')
    for p in photos:
        pid = p["photo_id"]
        link = f'<a href="annotated/{pid}.jpg">{escape(p["relative_path"])}</a>' if (run / "annotated" / (pid+".jpg")).exists() else escape(p["relative_path"])
        page.append(f'<tr><td>{link}</td><td>{escape(p["status"])} {escape(p["error"])}</td><td>{p["detected_faces"]}</td><td>{escape(p["people"])}</td></tr>')
    page.append('</table></html>')
    (run / "index.html").write_text("\n".join(page), encoding="utf-8")
