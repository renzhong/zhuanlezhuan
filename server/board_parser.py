import cv2
import numpy as np
import os
from models import BoardRect, Cell, TypeImg
from utils import find_cell_type
from collections import Counter

def load_type_imgs(dir_path):
    type_imgs = []
    img_type = 1
    for fname in os.listdir(dir_path):
        if fname.lower().endswith('.png'):
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(dir_path, fname)
            img = cv2.imread(img_path)
            if img is not None:
                type_imgs.append({'img': img, 'name': name, 'img_type': img_type})
                img_type += 1
    return type_imgs

def find_board_roi(img, top_n=10, select_idx=0):
    # img: OpenCV 图像对象
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    rect_set = set()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1000:
            continue
        key = (x, y, w, h)
        if key in rect_set:
            continue
        rect_set.add(key)
        board_img = img[y:y+h, x:x+w]
        rects.append(BoardRect(board_img, x, y, w, h, area))
    rects = sorted(rects, key=lambda r: r.area, reverse=True)[:top_n]
    if len(rects) > select_idx:
        return rects[select_idx]
    elif rects:
        return rects[0]
    else:
        return None

def split_board_cells_v2(board_rect, rows=14, cols=10):
    img = board_rect.img
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        cx = x + rw // 2
        cy = y + rh // 2
        rects.append({'x': x, 'y': y, 'w': rw, 'h': rh, 'cx': cx, 'cy': cy, 'img': img[y:y+rh, x:x+rw]})
    cell_w = w / cols
    cell_h = h / rows
    cell_centers = [[{'row': i, 'col': j, 'cx': int((j + 0.5) * cell_w), 'cy': int((i + 0.5) * cell_h)} for j in range(cols)] for i in range(rows)]
    cells = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, rect in enumerate(rects):
        min_dist = float('inf')
        min_cell = None
        for i in range(rows):
            for j in range(cols):
                dist = np.hypot(rect['cx'] - cell_centers[i][j]['cx'], rect['cy'] - cell_centers[i][j]['cy'])
                if dist < min_dist:
                    min_dist = dist
                    min_cell = (i, j)
        if min_cell:
            cells[min_cell[0]][min_cell[1]] = Cell(min_cell[0], min_cell[1], rect['x'], rect['y'], rect['w'], rect['h'], rect['img'], -1)
    return cells

def split_board_cells_v3(board_rect, min_cells=10):
    img = board_rect.img
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < 100:  # 忽略极小噪点
            continue
        rects.append({'x': x, 'y': y, 'w': rw, 'h': rh, 'area': area, 'img': img[y:y+rh, x:x+rw], 'cx': x+rw//2, 'cy': y+rh//2})
    if len(rects) < min_cells:
        return None
    rects = sorted(rects, key=lambda r: r['area'])
    mid_idx = len(rects) // 2
    ref_rect = rects[mid_idx]
    ref_w, ref_h = ref_rect['w'], ref_rect['h']
    # 选出和模板尺寸相近的所有棋子
    valid_rects = [r for r in rects if abs(r['w']-ref_w)<10 and abs(r['h']-ref_h)<10]
    if len(valid_rects) < min_cells:
        return None

    cols = 10
    rows = 14
    # 计算理论中心点
    offset_x = (w - ref_w * cols) // 2
    offset_y = (h - ref_h * rows) // 2
    theory_centers = [[(int(offset_x + (j+0.5)*ref_w), int(offset_y + (i+0.5)*ref_h)) for j in range(cols)] for i in range(rows)]
    cells = [[None for _ in range(cols)] for _ in range(rows)]
    for rect in valid_rects:
        min_dist = float('inf')
        min_cell = None
        for i in range(rows):
            for j in range(cols):
                tcx, tcy = theory_centers[i][j]
                dist = np.hypot(rect['cx']-tcx, rect['cy']-tcy)
                if dist < min_dist:
                    min_dist = dist
                    min_cell = (i, j)
        if min_cell:
            i, j = min_cell
            if cells[i][j] is None or min_dist < ref_w//2:
                cells[i][j] = Cell(i, j, rect['x'], rect['y'], rect['w'], rect['h'], rect['img'], -1)
    return cells

def parse_board_image(img, cell_split_version='v3'):
    # img: OpenCV 图像对象
    type_imgs = load_type_imgs(os.path.join('static', 'type_imgs'))
    if cell_split_version == 'v3':
        board_rect = find_board_roi(img)
        if board_rect is None:
            raise Exception('未检测到棋盘')
        cells = split_board_cells_v3(board_rect)
        if cells is None:
            board_rect = find_board_roi(img, 2)
            if board_rect is None:
                raise Exception('未检测到棋盘')
            # 回退到v2
            cells = split_board_cells_v2(board_rect)
    else:
        board_rect = find_board_roi(img, 2)
        if board_rect is None:
            raise Exception('未检测到棋盘')
        cells = split_board_cells_v2(board_rect)
    bg_img = cv2.imread(os.path.join('static', 'bg.png'))
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            if cells[i][j] is None:
                cells[i][j] = Cell(i, j, 0, 0, 103, 107, bg_img, 0)
    for row in cells:
        for cell in row:
            if cell.cell_type == 0:
                continue
            type_img, score = find_cell_type(cell, [TypeImg(**t) for t in type_imgs], threshold=0.0)
            if type_img is not None and score >= 0.9:
                cell.cell_type = type_img.img_type
            else:
                cell.cell_type = 0
    board = [[cell.cell_type for cell in row] for row in cells]
    return board, type_imgs

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法: python board_parser.py <图片路径>')
        exit(1)
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图片: {img_path}')
        exit(1)
    try:
        board, _ = parse_board_image(img, cell_split_version='v3')
        print('识别到的棋盘:')
        for row in board:
            print(row)
    except Exception as e:
        print(f'识别失败: {e}')