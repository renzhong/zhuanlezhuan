import cv2
import numpy as np
import sys
from collections import deque
import hashlib
from dataclasses import dataclass
from collections import Counter
import os

def get_patch_hash(img, x, y, size=5):
    patch = img[y:y+size, x:x+size]
    if patch.shape != (size, size):
        return None  # 边界不够
    return hashlib.md5(patch.flatten()).hexdigest()

def find_same_patch(img_path, start_x, start_y, max_right, max_down, patch_size=5):
    # 0. 读取图片，转为灰度
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败")
        return
    if len(img.shape) == 3:
        print("图片是彩色")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("图片是灰色")
        gray = img

    # 1. 计算起始patch的hash
    base_hash = get_patch_hash(gray, start_x, start_y, patch_size)
    if base_hash is None:
        print("起始patch超出边界")
        return

    print(f"起始patch hash: {base_hash}")

    # 2. 横向查找
    for dx in range(start_x + 1, max_right):
        for dy in range(start_y + 1, max_down):
            x = dx
            y = dy
            h = get_patch_hash(gray, x, y, patch_size)
            if h is None:
                print(f"get_path_hash is none: ({x}, {y}, {patch_size})")
                continue
            if h == base_hash:
                print(f"找到相同patch: ({x}, {y})")

def save_cell(cell, save_path):
    cv2.imwrite(save_path, cell.img)
    print(f"img已保存到: {save_path}")

def save_patch(img_path, x, y, width, height, save_path):
    """
    从img_path读取图片，截取(x, y)为左上角，大小为patch_size的区域，保存到save_path。
    """
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败")
        return
    print("save_patch img size width: ", width, " height: ", height, " x: ", x, " y: ", y)
    patch = img[y:y+height, x:x+width]
    if patch.shape[0] != height or patch.shape[1] != width:
        print("patch超出边界")
        return
    cv2.imwrite(save_path, patch)
    print(f"patch已保存到: {save_path}")

def print_potted(img_path, x, y, end_x, end_y):
    """
    打印以(x, y)为左上角，(end_x, end_y)为右下角的长方形区域的像素值
    """
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败")
        return
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 检查边界
    h, w = gray.shape
    if x < 0 or y < 0 or end_x > w or end_y > h or end_x <= x or end_y <= y:
        print("区域超出图片边界")
        return

    patch = gray[y:end_y, x:end_x]
    print(f"patch from ({x}, {y}) to ({end_x}, {end_y}):")
    print(patch)
    print("flatten:", patch.flatten())

@dataclass
class BoardRect:
    img: any
    x: int
    y: int
    w: int
    h: int
    area: int

@dataclass
class Cell:
    row: int
    col: int
    x: int      # 左上角x
    y: int      # 左上角y
    w: int      # 宽
    h: int      # 高
    img: any    # 该cell的图像
    cell_type: -1 # 棋子的类型

@dataclass
class TypeImg:
    img: any
    name: str
    img_type: int

def find_board_roi(image_path, top_n=10, select_idx=1):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("contours: ", len(contours))
    rects = []
    rect_set = set()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1000:  # 过滤太小的噪声框
            continue
        key = (x, y, w, h)
        if key in rect_set:
            continue
        rect_set.add(key)
        board_img = img[y:y+h, x:x+w]
        rects.append(BoardRect(board_img, x, y, w, h, area))
    # 按面积排序，保留前top_n个
    rects = sorted(rects, key=lambda r: r.area, reverse=True)[:top_n]
    for idx, r in enumerate(rects):
        print(f"Top{idx+1}: area={r.area}, x={r.x}, y={r.y}, w={r.w}, h={r.h}")
    if len(rects) > select_idx:
        r = rects[select_idx]
        print(f"返回第{select_idx+1}大矩形")
        return r
    elif rects:
        print("只找到一个大矩形，返回最大")
        return rects[0]
    else:
        print("未检测到合适的矩形区域")
        return None

def split_board_cells(board_rect, rows, cols):
    w = board_rect.w
    h = board_rect.h
    cell_h = h // rows
    cell_w = w // cols
    offset_x = (w - cols * cell_w) // 2
    offset_y = (h - rows * cell_h) // 2
    print(f"offset_x: {offset_x}, offset_y: {offset_y}")

    # 预定义 x_vec 和 y_vec 数组
    x_vec = [1, 1, 15, 20, 25, 30, 35, 40, 45, 50]  # 10列
    y_vec = [
        2,
        2+111,
        2+111*2,
        2+111*3,
        2+111*4,
        1+111*5,
        1+111*6,
        1+111*7,
        1+111*8,
        1+111*9,
        1+111*10,
        0+111*11,
        0+111*12,
        0+111*13
    ]  # 14行

    cells = []
    for i in range(rows):
        row_cells = []
        for j in range(cols):
            x0 = j * cell_w + offset_x
            y0 = y_vec[i] + offset_y
            cell_img = board_rect.img[y0:y0+cell_h, x0:x0+cell_w]
            # print(f"i: {i}, j: {j}, x0: {x0}, y0: {y0}, cell_w: {cell_w}, cell_h: {cell_h} cell_img: {cell_img.shape}")
            # cv2.imwrite(f"debug_{i}_{j}.png", cell_img)
            row_cells.append(Cell(i, j, x0, y0, cell_w, cell_h, cell_img, -1))
        cells.append(row_cells)
    return cells  # 返回二维列表 cells[row][col]

def is_same_cell_ncc(img, cell1, cell2, threshold=0.9):
    """
    使用归一化互相关判断两个 cell 是否相同。
    threshold: 相似度阈值,越接近1越严格。
    返回 (bool, score)
    """
    # img1 = img[cell1.y:cell1.y+cell1.h, cell1.x:cell1.x+cell1.w]
    # img2 = img[cell2.y:cell2.y+cell2.h, cell2.x:cell2.x+cell2.w]
    img1 = cell1.img
    img2 = cell2.img

    # 转灰度
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 尺寸归一化
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # NCC模板匹配
    res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    score = res[0][0]
    return score > threshold, score

def find_same_cell(game_board, i, j):
    """寻找 game_board[i][j] 的相同棋子，返回一个列表"""
    result = []
    target_type = game_board[i][j]
    if target_type == 0:  # 空地
        return result

    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if (row != i or col != j) and game_board[row][col] == target_type:
                result.append((row, col))
    return result

# 1. 从 i 向 0 遍历，寻找一个空地，如果没找到则不能向这个方向移动
# 1.1 如果找到空地，则继续向 0 遍历，直到找到一个棋子，或者边界。这个连续的空地长度，就是棋子可以在这个方向上可以移动的步长。
# 2. 从 i 向 13 遍历，寻找一个空地，如果没找到则不能向这个方向移动
# 2.1 如果找到空地，则继续向 13 遍历，直到找到一个棋子，或者边界。这个连续的空地长度，就是棋子可以在这个方向上可以移动的步长。
# 3. 从 j 向 0 遍历，寻找一个空地，如果没找到则不能向这个方向移动
# 3.1 如果找到空地，则继续向 0 遍历，直到找到一个棋子，或者边界。这个连续的空地长度，就是棋子可以在这个方向上可以移动的步长。
# 4. 从 j 向 9 遍历，寻找一个空地，如果没找到则不能向这个方向移动
# 4.1 如果找到空地，则继续向 9 遍历，直到找到一个棋子，或者边界。这个连续的空地长度，就是棋子可以在这个方向上可以移动的步长。
# 返回一个(i,j) 的列表，是这个棋子可以到达的位置。
def find_step_list(game_board, i, j):
    """寻找 game_board[i][j] 可以移动到的位置，返回一个列表"""
    result = []
    rows = len(game_board)
    cols = len(game_board[0])

    # 向上搜索
    for up in range(i-1, -1, -1):
        if game_board[up][j] == 0:  # 找到空地
            # 继续向上找到边界或非空格子
            max_step = 1
            for up2 in range(up-1, -1, -1):
                if game_board[up2][j] == 0:
                    # 连续空地
                    max_step += 1
                else:
                    break
            for up_step in range(1, max_step+1):
                # 可以向上移动的位置
                result.append((i - up_step, j))
            break

    # 向下搜索
    for down in range(i+1, rows):
        if game_board[down][j] == 0:  # 找到空地
            # 继续向下找到边界或非空格子
            max_step = 1
            for down2 in range(down+1, rows):
                if game_board[down2][j] == 0:
                    # 连续空地
                    max_step += 1
                else:
                    break
            for down_step in range(1, max_step+1):
                # 可以向下移动的位置
                result.append((i + down_step, j))
            break

    # 向左搜索
    for left in range(j-1, -1, -1):
        if game_board[i][left] == 0:  # 找到空地
            # 继续向左找到边界或非空格子
            max_step = 1
            for left2 in range(left-1, -1, -1):
                if game_board[i][left2] == 0:
                    # 连续空地
                    max_step += 1
                else:
                    break
            for left_step in range(1, max_step+1):
                # 可以向左移动的位置
                result.append((i, j - left_step))
            break

    # 向右搜索
    for right in range(j+1, cols):
        if game_board[i][right] == 0:  # 找到空地
            # 继续向右找到边界或非空格子
            max_step = 1
            for right2 in range(right+1, cols):
                if game_board[i][right2] == 0:
                    # 连续空地
                    max_step += 1
                else:
                    break
            for right_step in range(1, max_step+1):
                # 可以向右移动的位置
                result.append((i, j + right_step))
            break

    return result

def check_match_line(game_board, pos1, pos2):
    i, j = pos1
    i2, j2 = pos2
    if i == i2:
        start = min(j, j2)
        end = max(j, j2)
        for col in range(start + 1, end):
            if game_board[i][col] != 0:
                return False
        return True
    if j == j2:
        start = min(i, i2)
        end = max(i, i2)
        for row in range(start + 1, end):
            if game_board[row][j] != 0:
                return False
        return True
    return False

# 检查 game_board[i][j] 和 game_board[i2][j2] 是否可消除
# 算法:
# 1. 如果 i = i2,判断从j 到 j2 之间是否都是空地, 如果是空地则可以消除
# 2. 如果 j = j2,判断从i 到 i2 之间是否都是空地, 如果是空地则可以消除
def check_match(game_board, pos1, pos2):
    """检查 game_board[i][j] 和 game_board[i2][j2] 是否可消除"""
    i, j = pos1
    i2, j2 = pos2


    # 同一行检查
    if i == i2:
        start = min(j, j2)
        end = max(j, j2)
        # 检查中间是否都是空地
        for col in range(start + 1, end):
            if game_board[i][col] != 0:
                return False
        return True

    # 同一列检查
    if j == j2:
        start = min(i, i2)
        end = max(i, i2)
        # 检查中间是否都是空地
        for row in range(start + 1, end):
            if game_board[row][j] != 0:
                return False
        return True

    return False

def move_cell_2(game_board, old_board, pos1, pos2):
    # print(f"move_cell_2 {pos1} -> {pos2}")
    i, j = pos1
    i2, j2 = pos2
    if i == i2:
        if j < j2:
            # 向右移动
            for x in range(j, j2):
                game_board[i][x] = 0
            fill_index = 0
            while(1):
                if old_board[i][j+fill_index] != 0:
                    game_board[i][j2+fill_index] = old_board[i][j+fill_index]
                    fill_index += 1
                else:
                    break
        else:
            # 向左移动
            for x in range(j, j2, -1):
                game_board[i][x] = 0
            fill_index = 0
            while(1):
                if old_board[i][j-fill_index] != 0:
                    game_board[i][j2-fill_index] = old_board[i][j-fill_index]
                    fill_index += 1
                else:
                    break
    elif j == j2:
        if i < i2:
            # 向下移动
            for x in range(i, i2):
                game_board[x][j] = 0
            fill_index = 0
            while(1):
                if old_board[i+fill_index][j] != 0:
                    game_board[i2+fill_index][j] = old_board[i+fill_index][j]
                    fill_index += 1
                else:
                    break
        else:
            # 向上移动
            for x in range(i, i2, -1):
                game_board[x][j] = 0
            fill_index = 0
            while(1):
                if old_board[i-fill_index][j] != 0:
                    game_board[i2-fill_index][j] = old_board[i-fill_index][j]
                    fill_index += 1
                else:
                    break
    return game_board

# 2. 如果 i = i2,
# 2.1 如果 j < j2, 则从 j 向 9 遍历，找到第一个空地(i, m), 则
#  x = m + j2 - j - 1
# for x -> j2:
#   tmp_board[i][x] = tmp_board[i][x-(j2-j)]
# for j2 - 1 -> j:
#   tmp_board[i][x] = 空地
# 3. 其他方向同理。
def move_cell(game_board, pos1, pos2):
    """移动 game_board[i][j] 到 (i2,j2)"""
    i, j = pos1
    i2, j2 = pos2

    # 创建临时棋盘
    # tmp_board = [row[:] for row in game_board]

    # 如果在同一行
    if i == i2:
        if j < j2:  # 向右移动
            step = j2 - j
            for x in range(j2 + step, j2 - 1, -1):
                game_board[i][x] = game_board[i][x-step]
            for x in range(j2 - 1, j-1, -1):
                game_board[i][x] = 0
        else:  # 向左移动
            step = j - j2
            for x in range(j2 - step, j2 + 1):
                game_board[i][x] = game_board[i][x+step]
            for x in range(j2 + 1, j+1, -1):
                game_board[i][x] = 0
    # 如果在同一列
    elif j == j2:
        if i < i2:  # 向下移动
            step = i2 - i
            for x in range(i2 + step, i2 - 1, -1):
                game_board[x][j] = game_board[x-step][j]
            for x in range(i2 - 1, i-1, -1):
                game_board[x][j] = 0
        else:  # 向上移动
            step = i - i2
            for x in range(i2 - step, i2 + 1):
                game_board[x][j] = game_board[x-step][j]
            for x in range(i2 - 1, i+1, -1):
                game_board[x][j] = 0

    return game_board

def copy_board(game_board):
    """返回一个 game_board 的深拷贝"""
    return [row[:] for row in game_board]

def hash_game_board(game_board):
    """将 game_board 转换为 hash 值"""
    return hash(tuple(tuple(row) for row in game_board))

result = []  # 全局变量存储结果路径
hash_board_set = set()
count = 0

# 算法：
# 1. 从 (0,0) 开始，找到第一个棋子A，
# 1.1 find_same_cell(game_board, A.i, A.j) 找到A的相同棋子列表c_list
# 1.2 find_step_list(game_board, A.i, A.j) 找到A可以移动的步长列表 step_list
# 2 遍历 c_list
# 2.1 遍历 step_list
# 2.2 check_match(game_board, c_list[i], step_list[j])
# 2.3.1 如果 match，
# 2.3.1.1 深拷贝当前 game_board 到 tmp_board
# 2.3.1.2 在 tmp_board 中移动棋子A到 step_list[j]
# 2.3.1.3 在 tmp_board 中将 step_list[j] 和 c_list[i] 修改为空地
# 2.3.1.4 调用 dfs(tmp_board)
# 2.3.1.5 如果 dfs 返回 True，则将上面的移动操作((A.i, A.j), (step_list[j].i, step_list[j].j))写入到 result.append()
# 2.3.1.6 return True
# 2.3.1.7 如果 dfs 返回 False，则继续遍历.
# 2.3.2 如果不 match，则继续遍历
# 3. 如果棋盘中没有棋子，返回 True
# 4. 如果 c_list && step_list 的双重遍历结束，返回 False
def dfs(game_board):
    """深度遍历解决消除问题"""
    global count, result, hash_board_set  # 声明所有要使用的全局变量

    # 检查是否所有棋子都已消除
    all_cleared = True
    first_piece = None

    count += 1  # 注意这里不要写成 global count += 1
    # if (count % 1000 == 0):
    #     print(f"count: {count}")
    #     show_game_board(game_board, f"count: {count}")

    hash_board = hash_game_board(game_board)
    if hash_board in hash_board_set:
        return False
    hash_board_set.add(hash_board)

    # 寻找第一个棋子
    for i in range(len(game_board)):
        for j in range(len(game_board[0])):
            if game_board[i][j] != 0:
                all_cleared = False
                first_piece = (i, j)
                break
        if first_piece:
            break

    # 如果所有棋子都已消除，返回成功
    if all_cleared:
        return True

    # 剪枝 1: 记录已经找过的棋子类型
    for i in range(len(game_board)):
        for j in range(len(game_board[0])):
            if game_board[i][j] == 0:
                continue
            # 找到相同的棋子
            same_cells = find_same_cell(game_board, i, j)
            # 找到可以移动到的位置
            step_list = find_step_list(game_board, i, j)

            # 遍历所有可能的移动
            for same_cell in same_cells:
                i2, j2 = same_cell
                # print(f"check {i},{j} vs {i2},{j2}")
                if i == i2:
                    # print(f"check_match_line {i},{j} vs {i2},{j2}")
                    if check_match_line(game_board, (i, j), (i2, j2)):
                        # print(f"match: {i},{j} vs {i2},{j2}")
                        new_board = copy_board(game_board)
                        new_board[i][j] = 0
                        new_board[i2][j2] = 0
                        # 递归求解
                        if dfs(new_board):
                            # 记录这步移动
                            result.append(((i, j), (i,j), (i2,j2)))
                            return True
                elif j == j2:
                    # print(f"check_match_clone {i},{j} vs {i2},{j2}")
                    if check_match_line(game_board, (i, j), (i2, j2)):
                        # print(f"match: {i},{j} vs {i2},{j2}")
                        new_board = copy_board(game_board)
                        new_board[i][j] = 0
                        new_board[i2][j2] = 0
                        # 递归求解
                        if dfs(new_board):
                            # 记录这步移动
                            result.append(((i, j), (i,j), (i2,j2)))
                            return True
                else:
                    # print(f"check_match_line {i},{j} vs {i2},{j2}")
                    for step in step_list:
                        # print(f"check_match {step} vs {i2},{j2}")
                        if check_match(game_board, step, same_cell):
                            # print(f"match: {step} vs {i2},{j2}")
                            # 消除这两个棋子
                            new_board = copy_board(game_board)
                            # move_cell(new_board, (i, j), step)
                            move_cell_2(new_board, game_board, (i, j), step)
                            new_board[step[0]][step[1]] = 0
                            new_board[same_cell[0]][same_cell[1]] = 0

                            # 递归求解
                            if dfs(new_board):
                                # 记录这步移动
                                result.append(((i, j), step, same_cell))
                                return True

    # show_game_board(game_board, "end")
    return False

def show_game_board(game_board, debug_info=""):
    print(f"-------game_board--{debug_info}-----")
    for row in game_board:
        row_str = ''
        for cell in row:
            # 每个 cell_type 占3个字符，居中对齐
            row_str += f"{str(cell).center(3)}"
        print(row_str)

def show_game_board_with_highlight(game_board, debug_info, from_pos, to_pos):
    # ANSI 颜色代码
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    print(f"-------{debug_info}-----")
    for i, row in enumerate(game_board):
        row_str = ''
        for j, cell in enumerate(row):
            cell_str = str(cell).center(3)
            if (i, j) == from_pos:
                row_str += f"{RED}{cell_str}{RESET}"
            elif (i, j) == to_pos:
                row_str += f"{GREEN}{cell_str}{RESET}"
            else:
                row_str += cell_str
        print(row_str)

def split_board_cells_v2(board_rect, rows=14, cols=10):
    img = board_rect.img
    h, w = img.shape[:2]

    # 1. 检测所有小矩形
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        print(f"area: {area}, x: {x}, y: {y}, rw: {rw}, rh: {rh}, rows*cols: {rows*cols}")
        # if area < 100 or area > (w*h)//(rows*cols)//2:  # 过滤噪声
        #     continue
        cx = x + rw // 2
        cy = y + rh // 2
        rects.append({'x': x, 'y': y, 'w': rw, 'h': rh, 'cx': cx, 'cy': cy, 'img': img[y:y+rh, x:x+rw]})
        # cv2.imwrite(f"one_{x}_{y}.png", img[y:y+rh, x:x+rw])

    # 2. 计算理论 cell 中心点
    cell_w = w / cols
    cell_h = h / rows
    cell_centers = []
    for i in range(rows):
        row = []
        for j in range(cols):
            cx = int((j + 0.5) * cell_w)
            cy = int((i + 0.5) * cell_h)
            row.append({'row': i, 'col': j, 'cx': cx, 'cy': cy})
        cell_centers.append(row)

    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(None)
        cells.append(row)

    # 3. 匹配
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
            print(f"match: {idx}, {min_cell[0]}, {min_cell[1]}")
            # save_cell(cells[min_cell[0]][min_cell[1]], f"cell_{min_cell[0]}_{min_cell[1]}.png")

    # cells = []
    # used_rects = set()
    # for i in range(rows):
    #     row_cells = []
    #     for j in range(cols):
    #         min_dist = float('inf')
    #         min_rect = None
    #             if idx in used_rects:
    #                 continue
    #             dist = np.hypot(rect['cx'] - cell_centers[i][j]['cx'], rect['cy'] - cell_centers[i][j]['cy'])
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 min_rect = (idx, rect)
    #         # 距离阈值可根据实际调整
    #         if min_rect and min_dist < min(cell_w, cell_h) * 0.6:
    #             idx, rect = min_rect
    #             used_rects.add(idx)
    #             cell_img = rect['img']
    #             row_cells.append(Cell(i, j, rect['x'], rect['y'], rect['w'], rect['h'], cell_img, -1))
    #         else:
    #             # 空节点
    #             row_cells.append(Cell(i, j, 0, 0, int(cell_w), int(cell_h), np.zeros((int(cell_h), int(cell_w), 3), dtype=np.uint8), 0))
    #     cells.append(row_cells)
    return cells

def load_type_imgs(dir_path):
    """
    加载指定目录下所有 png 文件，返回 TypeImg 列表。
    name 字段为文件名前缀（不含扩展名）。
    """
    type_imgs = []
    img_type = 1
    for fname in os.listdir(dir_path):
        if fname.lower().endswith('.png'):
            name = os.path.splitext(fname)[0]
            img_path = os.path.join(dir_path, fname)
            img = cv2.imread(img_path)
            print(f"load_type_imgs name: {name}, w: {img.shape[1]}, h: {img.shape[0]}")
            if img is not None:
                type_imgs.append(TypeImg(img, name, img_type))
                img_type += 1
            else:
                print(f"图片加载失败: {img_path}")
    print(f"共加载 {len(type_imgs)} 个类型图片")
    return type_imgs

def find_cell_type(cell, type_imgs, threshold=0.7):
    """
    对 cell.img 与每个 type_img.img 进行 ncc 匹配，返回最相似的 TypeImg 和分数。
    若所有分数都低于 threshold，则返回 (None, None)
    """
    best_score = -1
    best_type_img = None
    cell_img = cell.img
    # 转灰度
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
        # 尺寸归一化
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

def build_board_img(cells, type_imgs, cell_w=103, cell_h=107):
    """
    根据 cells 和 type_imgs 拼接出一个棋盘大图。
    每个 cell 按 cell_type - 1 取 type_imgs，resize 到 cell_w*cell_h。
    空节点填充全白。
    返回拼接后的大图。
    """
    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0
    board_img = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            cell = cells[i][j]
            if cell is None or cell.cell_type == -1 or cell.cell_type == 0:
                continue
            idx = cell.cell_type - 1
            if idx < 0 or idx >= len(type_imgs):
                continue
            tpl_img = type_imgs[idx].img
            tpl_img_resized = cv2.resize(tpl_img, (cell_w, cell_h))
            y0 = i * cell_h
            x0 = j * cell_w
            board_img[y0:y0+cell_h, x0:x0+cell_w] = tpl_img_resized
    cv2.imwrite("board.png", board_img)
    return board_img

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python linkup_solver.py <screenshot.png>")
        sys.exit(1)
    img_path = sys.argv[1]
    # img_path = "4layer.png"
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败")
        sys.exit(1)

    r = find_board_roi(img_path, top_n=10, select_idx=2)  # 返回第三大
    if r is None:
        print("未检测到棋盘")

    save_patch(img_path, r.x, r.y, r.w, r.h, "2.png")

    cells = split_board_cells_v2(r)

    type_imgs = load_type_imgs("type_imgs")
    bg_imgs = cv2.imread("bg.png")
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            if cells[i][j] is None:
                cells[i][j] = Cell(i, j, 0, 0, 103, 107, bg_imgs, 0)

    # cell = cells[0][0]
    # cell2 = cells[0][2]
    # # save_cell(cell, "3.png")

    # 遍历cells，判断cell的类型
    cell_type = 1
    for row in cells:
        for cell in row:
            if cell.cell_type == 0:
                continue
            type_img, score = find_cell_type(cell, type_imgs, threshold=0.0)
            if type_img is None:
                print(f"没有找到形同的棋子, {cell.row},{cell.col}")
                continue
            if score < 0.9:
                print(f"疑似匹配, {cell.row},{cell.col}, {type_img.name} score: {score}")
                cv2.imwrite(f"cell_{cell.row}_{cell.col}.png", cell.img)
                cv2.imwrite(f"type_{type_img.name}.png", type_img.img)
            cell.cell_type = type_img.img_type

    # 统计每个 cell_type 的棋子数量
    type_counter = Counter()
    for row in cells:
        for cell in row:
            type_counter[cell.cell_type] += 1

    print("每种棋子的数量：")
    for t, cnt in sorted(type_counter.items()):
        print(f"类型 {t}: {cnt} 个")

    new_img = build_board_img(cells, type_imgs)

    # # ---- 4. 求解 ----
    # game_board = [[cell.cell_type for cell in row] for row in cells]
    # # print(game_board)
    # show_game_board(game_board)
    # dfs(game_board)
    # new_result = []
    # for i in range(len(result)-1, -1, -1):
    #     new_result.append(result[i])
    # # print(new_result)

    # result_board = copy_board(game_board)
    # for i in range(len(new_result)):
    #     # 打印矩阵
    #     old_pos = new_result[i][0]
    #     move_pos = new_result[i][1]
    #     check_pos = new_result[i][2]
    #     print(f"step {i}: {old_pos} -> {move_pos} match {check_pos}")

    #     show_game_board_with_highlight(result_board, f"step {i}", old_pos, move_pos)
    #     tmp_board = copy_board(result_board)
    #     if old_pos[0] == move_pos[0] and old_pos[1] == move_pos[1]:
    #         tmp_board[old_pos[0]][old_pos[1]] = 0
    #         tmp_board[check_pos[0]][check_pos[1]] = 0
    #         # show_game_board(tmp_board, "after erase")
    #     else:
    #         move_cell_2(tmp_board, result_board, old_pos, move_pos)
    #         # show_game_board(tmp_board, "after move")
    #         tmp_board[move_pos[0]][move_pos[1]] = 0
    #         tmp_board[check_pos[0]][check_pos[1]] = 0
    #         # show_game_board(tmp_board, "after erase")
    #     result_board = copy_board(tmp_board)
