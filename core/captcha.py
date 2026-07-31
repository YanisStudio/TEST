"""
測謊字元格辨識：定位「請按照號碼順序輸入文字」下方的雜訊字元圖並用 OCR 辨識。
依賴：pip install ddddocr
"""

import io
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

if getattr(sys, 'frozen', False):
    # 跟 gui/app.py 同樣的道理：PyInstaller onedir 打包後隨附資源實際在 _internal
    # 子資料夾底下，要用 sys._MEIPASS 才能取得正確路徑，不能假設是 exe 所在目錄。
    _RESOURCE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    _RESOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR      = os.path.join(_RESOURCE_DIR, 'assets')
ENTER_BOX_PATH  = os.path.join(ASSETS_DIR, 'anchors', 'Enter_Box.png')

def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread 在 Windows 上路徑含中文字時常直接讀失敗（回傳 None），改用
    np.fromfile 讀 bytes 再用 cv2.imdecode 解碼，不受路徑編碼影響。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


_OCR = None
_enter_template = None

_DIGIT_CHARSET = "123"  # 順序永遠只會是 1、2、3，收窄字元集讓模型不可能選到其他數字
# 曾經試過收窄成只留大寫（提示文字寫「不區分大小寫」，理論上應該無妨），
# 但實測拿 assets/samples 的標記樣本驗證後發現準確率從 79.2% 掉到 29.2%：
# 這個 OCR 模型明顯對小寫字母辨識力遠優於大寫，兩種大小寫都留著時模型幾乎
# 都自己選小寫；硬收窄成大寫反而拿掉模型比較準的選項，逼它常常誤判成數字。
# 保留兩種大小寫都收，讓模型自己選它比較有把握的那個。
_ALNUM_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_LOW_CONF_THRESHOLD = 0.96
_ROTATE_SWITCH_MARGIN = 0.03
_COLOR_FILTER_NAMES = ["red", "green", "blue", "orange", "purple", "yellow"]
_ROTATIONS = [
    ("90",     lambda im: im.rotate(90, expand=True)),
    ("180",    lambda im: im.rotate(180, expand=True)),
    ("270",    lambda im: im.rotate(270, expand=True)),
    ("fliplr", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
    ("flipud", lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)),
]


def _get_enter_template():
    """讀取「輸入框＋確認鈕」樣板圖（只給本模組內部定位用，不會被 _load_templates 掃到）。"""
    global _enter_template
    if _enter_template is None:
        img = _imread_unicode(ENTER_BOX_PATH, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'找不到錨點樣板: {ENTER_BOX_PATH}')
        _enter_template = img
    return _enter_template


def _find_enter_box(screen_bgr, region, scales=np.linspace(0.5, 1.8, 20)):
    """在指定 region=(x1,y1,x2,y2) 內找輸入框/確認鈕，回傳 (bbox, score)。bbox 為畫面座標。"""
    x1, y1, x2, y2 = region
    if x2 <= x1 or y2 <= y1:
        return None, -1.0
    roi = screen_bgr[y1:y2, x1:x2]
    tmpl = _get_enter_template()
    th, tw = tmpl.shape[:2]

    best_score, best_bbox = -1.0, None
    for scale in scales:
        rw, rh = int(tw * scale), int(th * scale)
        if rw < 8 or rh < 8 or rw > roi.shape[1] or rh > roi.shape[0]:
            continue
        resized = cv2.resize(tmpl, (rw, rh))
        result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score > best_score:
            best_score, best_bbox = score, (x1 + loc[0], y1 + loc[1], rw, rh)
    return best_bbox, best_score


def _get_ocr():
    global _OCR
    if _OCR is None:
        import ddddocr
        _OCR = ddddocr.DdddOcr(show_ad=False)
    return _OCR


# ─── 定位字元圖框（雜訊小方形） ──────────────────────────────────────────────

def find_captcha_boxes(bgr, min_size=40, max_size=180):
    """在圖片中找高雜訊小方形（測謊字元圖片的特徵）。
    回傳 [(x, y, w, h, box_bgr), ...]，依 x 排序（座標相對於傳入的 bgr）。"""
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 25, 90)
    k     = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_size < w < max_size and min_size < h < max_size):
            continue
        if not (0.5 < w / h < 2.0):
            continue
        roi = gray[y:y+h, x:x+w]
        if roi.std() < 15:           # 雜訊夠高才算字元圖片
            continue
        candidates.append((x, y, w, h, bgr[y:y+h, x:x+w].copy()))

    # 去重疊，保留較大的框
    candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
    kept = []
    for c in candidates:
        cx, cy, cw, ch, _ = c
        overlap = any(
            max(cx, kx) < min(cx+cw, kx+kw) and max(cy, ky) < min(cy+ch, ky+kh)
            for kx, ky, kw, kh, _ in kept
        )
        if not overlap:
            kept.append(c)

    kept.sort(key=lambda c: c[0])
    return kept


# ─── OCR ─────────────────────────────────────────────────────────────────────

def _to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return buf.getvalue()


def _color_candidates(pil_img, ocr):
    """把原圖放大 2 倍，依序用 ddddocr 內建的顏色過濾（隔離出字元筆畫的顏色，
    濾掉不同色的雜訊線/雜點），收集每種顏色洗出的乾淨單一字元結果。
    實測對「答案筆畫顏色夠飽和」的樣本能明顯拉高正確答案的信心分數，
    避免它在信心門檻/旋轉比較時被雜訊污染的錯誤答案覆蓋掉；
    顏色太淡或雜訊過重的樣本則通常洗不出結果，直接被忽略，不影響其他前處理管道。"""
    w, h = pil_img.size
    base = pil_img.resize((w*2, h*2), Image.LANCZOS)
    out = []
    for color in _COLOR_FILTER_NAMES:
        try:
            info = ocr.classification(_to_bytes(base), probability=True,
                                       color_filter_colors=[color])
        except Exception:
            continue
        text = info['text'].strip()
        if len(text) == 1:
            out.append((text, info['confidence']))
    return out


def _ocr_attempts(pil_img, ocr):
    """多種前處理依序嘗試，取第一個乾淨單字元結果；同時另外收集顏色過濾的候選，
    兩邊全部候選一起比信心分數，回傳最高的。全部失敗回傳 (None, 0.0)。

    這裡刻意只取「第一個」成功結果，不是漏改：實測拿 assets/samples 的標記樣本
    比較過「全部跑完再選信心最高」的版本，結果是打平（救回一筆、卻也弄壞另一筆），
    代表後段更強的前處理有時反而會生出信心更高但答案錯誤的雜訊結果，蓋掉前段
    已經正確的答案。既然沒有實測證據支持「全部比較」更準，就維持原本較簡單、
    較快的做法。"""
    w, h = pil_img.size
    attempts = [pil_img]
    attempts.append(pil_img.resize((w*3, h*3), Image.LANCZOS))
    gray = pil_img.convert("L")
    enh  = ImageEnhance.Contrast(gray).enhance(3.0)
    attempts.append(enh.resize((w*3, h*3), Image.LANCZOS))
    attempts.append(enh.resize((w*3, h*3), Image.LANCZOS).filter(ImageFilter.SHARPEN))

    candidates = []
    for img in attempts:
        info = ocr.classification(_to_bytes(img), probability=True)
        r = info['text'].strip()
        if len(r) == 1:
            candidates.append((r, info['confidence']))
            break

    candidates.extend(_color_candidates(pil_img, ocr))

    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda c: c[1])


def _ocr_best(pil_img, fallback_img=None, low_conf_threshold=_LOW_CONF_THRESHOLD,
              switch_margin=_ROTATE_SWITCH_MARGIN):
    """多種預處理輪流嘗試，回傳最佳結果。答案只會是英數字，限制字元集可避免辨識成中文。

    這款遊戲的雜訊字元圖有時方向會讓字母判斷錯（例如 B 被讀成 a），但盲目每次都
    嘗試旋轉會有副作用：某些字母（b/d/p/q、m/w 這類本身就會因旋轉互相混淆的字母）
    可能反而被换成錯誤答案，即使原始角度才是對的。所以只有在信心分數低於門檻時，
    才代表模型自己也沒把握，這時才值得多花成本試各種旋轉/翻轉角度。

    實測發現光是「信心分數比較高就換」還不夠：曾出現原始角度才是對的，但旋轉後
    的錯誤答案只贏了 0.0008 這種雜訊等級的差距，就整個蓋掉正確答案。所以旋轉後的
    結果必須贏過目前最佳結果至少 switch_margin，才會真的採用，避免被這種誤差內
    的雜訊分數波動誤導。

    遮蓋角落數字後的圖如果判斷不出來，會退回用未遮蓋的原圖再試一次，
    盡量避免整格判斷失敗（"?"）導致答案漏掉一個字元、後面全部錯位。"""
    ocr = _get_ocr()
    ocr.set_ranges(_ALNUM_CHARSET)

    text, conf = _ocr_attempts(pil_img, ocr)

    if not text or conf < low_conf_threshold:
        # m/w 是垂直鏡像關係（上下翻轉後兩者會互換）。m/w 字形本身左右大致對稱，
        # 所以「180 度旋轉」（＝水平翻轉+垂直翻轉）對它們來說效果跟單純垂直翻轉
        # 差不多，一樣會把 m 換成 w；實測發現原始角度已經讀到 m/w 時，flipud 和
        # 180 度都常常用比原本更高的信心「修正」成另一個錯誤答案——對這兩個字母
        # 來說，這兩種翻轉不會帶來新資訊，只會製造混淆，所以已經有 m/w 讀數時
        # 兩者都不採用，避免正確答案被换掉。
        skip_vertical_mirror = bool(text) and text.lower() in ("m", "w")

        # 所有旋轉/翻轉都先跑完、互相比較選出全域最佳，再拿去跟「原始角度」這個固定
        # 基準比較，而不是逐一比對一個會一直往上墊高的暫定最佳值——後者順序一變，
        # 门檻就跟著變，會讓真正信心最高的候選被中途出現的次佳候選擋下來。
        rotated = []
        for rname, transform in _ROTATIONS:
            if skip_vertical_mirror and rname in ("flipud", "180"):
                continue
            r_text, r_conf = _ocr_attempts(transform(pil_img), ocr)
            if r_text:
                rotated.append((r_text, r_conf))
        if rotated:
            best_text, best_conf = max(rotated, key=lambda r: r[1])
            if not text:
                text = best_text
            elif best_conf > conf + switch_margin:
                text = best_text

    if text:
        return text
    if fallback_img is not None:
        text, _ = _ocr_attempts(fallback_img, ocr)
        if text:
            return text
    return "?"


def _find_corner_digit(pil_img):
    """四角找小數字（作答順序），回傳 (digit, corner_name)。"""
    ocr = _get_ocr()
    ocr.set_ranges(_DIGIT_CHARSET)
    w, h = pil_img.size
    cw, ch = max(w//3, 12), max(h//3, 12)
    corners = {
        "TL": (0,    0,    cw,   ch),
        "TR": (w-cw, 0,    w,    ch),
        "BL": (0,    h-ch, cw,   h),
        "BR": (w-cw, h-ch, w,    h),
    }
    for name, box in corners.items():
        crop = pil_img.crop(box)
        # LANCZOS 放大細筆畫小數字時會產生模糊，有時反而讓模型判斷不出任何文字；
        # NEAREST 不做平滑、保留銳利邊緣，實測對這種小尺寸數字更容易辨識成功，
        # 所以 LANCZOS 讀不到時再退回 NEAREST 試一次。
        attempts = [
            crop.resize((crop.width*5, crop.height*5), Image.LANCZOS),
            crop.resize((crop.width*5, crop.height*5), Image.NEAREST),
        ]
        for big in attempts:
            txt = ocr.classification(_to_bytes(big)).strip()
            for c in txt:
                if c.isdigit() and c != "0":
                    return c, name
    return "?", "?"


def recognize_box(box_bgr):
    """回傳 (order_str, char_str)。"""
    original = Image.fromarray(cv2.cvtColor(box_bgr, cv2.COLOR_BGR2RGB))
    w, h = original.size

    order, corner = _find_corner_digit(original)

    cw, ch = max(w//3, 12), max(h//3, 12)
    mask_map = {
        "TL": (0,    0,    cw,   ch),
        "TR": (w-cw, 0,    w,    ch),
        "BL": (0,    h-ch, cw,   h),
        "BR": (w-cw, h-ch, w,    h),
    }
    masked = original
    if corner in mask_map:
        masked = original.copy()
        draw = ImageDraw.Draw(masked)
        draw.rectangle(mask_map[corner], fill=(255, 255, 255))

    char = _ocr_best(masked, fallback_img=original)
    return order, char


# ─── 整合：以文字樣板錨點定位字元格並辨識 ────────────────────────────────────

def solve_captcha_region(screen_bgr, anchor_bbox, x_pad_ratio=0.4,
                          max_search_ratio=15.0, enter_conf=0.5):
    """
    anchor_bbox：偵測樣板（Text_Box.png，固定提示文字）在畫面上比對到的
    (x, y, w, h)，取自 _scan_templates 的 best_bbox。

    字元圖夾在「提示文字」與「輸入框/確認鈕」之間，兩邊都是固定不變的內容，
    所以不用猜一個高度倍數，而是先在文字下方一段夠寬鬆的範圍內找 Enter_Box
    （輸入框＋確認鈕）樣板，用它的上緣當作搜尋帶的下邊界——這樣搜尋帶幾乎
    就是實際的字元格空間，不會吃到對話框外的背景，也不受答案格數影響。
    若一時沒找到 Enter_Box（信心 < enter_conf），退回原本用比例猜高度的作法
    當保底，避免因為單一樣板沒對到就完全找不到字元格。

    回傳 (answer_str, results, enter_bbox)：
    results 為 [(order, char, x, y, w, h), ...]（畫面座標）；
    enter_bbox 是這次有信心找到的「輸入框＋確認鈕」位置 (x, y, w, h)，沒找到則為 None，
    給呼叫端用來點擊輸入框、確保輸入焦點在正確的地方（見 enter_box_click_point）。
    """
    H, W = screen_bgr.shape[:2]
    ax, ay, aw, ah = anchor_bbox

    pad = int(aw * x_pad_ratio)
    rx1 = max(0, ax - pad)
    rx2 = min(W, ax + aw + pad)
    ry1 = min(H, ay + ah)                              # 從文字下緣開始找
    search_y2 = min(H, ry1 + int(ah * max_search_ratio))  # 找 Enter_Box 用的寬鬆範圍

    enter_bbox, enter_score = None, -1.0
    try:
        enter_bbox, enter_score = _find_enter_box(screen_bgr, (rx1, ry1, rx2, search_y2))
    except FileNotFoundError:
        pass

    if enter_bbox and enter_score >= enter_conf:
        ry2 = enter_bbox[1]          # 用輸入框上緣當下邊界
    else:
        ry2 = search_y2              # 找不到就退回寬鬆範圍保底

    valid_enter_bbox = enter_bbox if (enter_bbox and enter_score >= enter_conf) else None

    if rx2 <= rx1 or ry2 <= ry1:
        return "", [], valid_enter_bbox

    roi = screen_bgr[ry1:ry2, rx1:rx2]
    boxes = find_captcha_boxes(roi)
    if not boxes:
        return "", [], valid_enter_bbox

    results = []
    for bx, by, bw, bh, box_img in boxes:
        order, char = recognize_box(box_img)
        results.append((order, char, rx1 + bx, ry1 + by, bw, bh))

    results.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 99)
    answer = "".join(c for _, c, *_ in results if c not in ("?", ""))
    return answer, results, valid_enter_bbox


def enter_box_click_point(enter_bbox):
    """回傳 Enter_Box（輸入框＋確認鈕）裡，輸入框範圍內安全可點擊的座標。
    輸入框大約佔樣板寬度的前 80%，抓 40% 處夠靠近中間、又不會太靠近跟確認鈕的交界。"""
    ex, ey, ew, eh = enter_bbox
    return ex + int(ew * 0.4), ey + int(eh * 0.5)
