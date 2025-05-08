import cv2
import numpy as np
import hashlib

def get_patch_hash(img, x, y, size=5):
    patch = img[y:y+size, x:x+size]
    if patch.shape != (size, size):
        return None
    return hashlib.md5(patch.flatten()).hexdigest()

def is_same_cell_ncc(cell1, cell2, threshold=0.9):
    img1 = cell1.img
    img2 = cell2.img
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    score = res[0][0]
    return score > threshold, score

def copy_board(game_board):
    return [row[:] for row in game_board]

def hash_game_board(game_board):
    return hash(tuple(tuple(row) for row in game_board))

def find_cell_type(cell, type_imgs, threshold=0.7):
    best_score = -1
    best_type_img = None
    cell_img = cell.img
    if len(cell_img.shape) == 3:
        cell_img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        cell_img_gray = cell_img
    for type_img in type_imgs:
        tpl = type_img.img
        if len(tpl.shape) == 3:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        else:
            tpl_gray = tpl
        if cell_img_gray.shape != tpl_gray.shape:
            tpl_gray_resized = cv2.resize(tpl_gray, (cell_img_gray.shape[1], cell_img_gray.shape[0]))
        else:
            tpl_gray_resized = tpl_gray
        res = cv2.matchTemplate(cell_img_gray, tpl_gray_resized, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        if score > best_score:
            best_score = score
            best_type_img = type_img
    if best_score >= threshold:
        return best_type_img, best_score
    else:
        return None, None 