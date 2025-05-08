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

def find_board_roi(img, top_n=10, select_idx=2):
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

def parse_board_image(img):
    # img: OpenCV 图像对象
    type_imgs = load_type_imgs(os.path.join('static', 'type_imgs'))
    board_rect = find_board_roi(img)
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