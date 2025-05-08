from utils import copy_board, hash_game_board
from collections import deque

def find_same_cell(game_board, i, j):
    result = []
    target_type = game_board[i][j]
    if target_type == 0:
        return result
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if (row != i or col != j) and game_board[row][col] == target_type:
                result.append((row, col))
    return result

def find_step_list(game_board, i, j):
    result = []
    rows = len(game_board)
    cols = len(game_board[0])
    for up in range(i-1, -1, -1):
        if game_board[up][j] == 0:
            max_step = 1
            for up2 in range(up-1, -1, -1):
                if game_board[up2][j] == 0:
                    max_step += 1
                else:
                    break
            for up_step in range(1, max_step+1):
                result.append((i - up_step, j))
            break
    for down in range(i+1, rows):
        if game_board[down][j] == 0:
            max_step = 1
            for down2 in range(down+1, rows):
                if game_board[down2][j] == 0:
                    max_step += 1
                else:
                    break
            for down_step in range(1, max_step+1):
                result.append((i + down_step, j))
            break
    for left in range(j-1, -1, -1):
        if game_board[i][left] == 0:
            max_step = 1
            for left2 in range(left-1, -1, -1):
                if game_board[i][left2] == 0:
                    max_step += 1
                else:
                    break
            for left_step in range(1, max_step+1):
                result.append((i, j - left_step))
            break
    for right in range(j+1, cols):
        if game_board[i][right] == 0:
            max_step = 1
            for right2 in range(right+1, cols):
                if game_board[i][right2] == 0:
                    max_step += 1
                else:
                    break
            for right_step in range(1, max_step+1):
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

def check_match(game_board, pos1, pos2):
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

def move_cell_2(game_board, old_board, pos1, pos2):
    i, j = pos1
    i2, j2 = pos2
    if i == i2:
        if j < j2:
            for x in range(j, j2):
                game_board[i][x] = 0
            fill_index = 0
            while True:
                if old_board[i][j+fill_index] != 0:
                    game_board[i][j2+fill_index] = old_board[i][j+fill_index]
                    fill_index += 1
                else:
                    break
        else:
            for x in range(j, j2, -1):
                game_board[i][x] = 0
            fill_index = 0
            while True:
                if old_board[i][j-fill_index] != 0:
                    game_board[i][j2-fill_index] = old_board[i][j-fill_index]
                    fill_index += 1
                else:
                    break
    elif j == j2:
        if i < i2:
            for x in range(i, i2):
                game_board[x][j] = 0
            fill_index = 0
            while True:
                if old_board[i+fill_index][j] != 0:
                    game_board[i2+fill_index][j] = old_board[i+fill_index][j]
                    fill_index += 1
                else:
                    break
        else:
            for x in range(i, i2, -1):
                game_board[x][j] = 0
            fill_index = 0
            while True:
                if old_board[i-fill_index][j] != 0:
                    game_board[i2-fill_index][j] = old_board[i-fill_index][j]
                    fill_index += 1
                else:
                    break
    return game_board

def solve_board(board):
    result = []
    hash_board_set = set()
    def dfs(game_board):
        hash_board = hash_game_board(game_board)
        if hash_board in hash_board_set:
            return False
        hash_board_set.add(hash_board)
        all_cleared = True
        first_piece = None
        for i in range(len(game_board)):
            for j in range(len(game_board[0])):
                if game_board[i][j] != 0:
                    all_cleared = False
                    first_piece = (i, j)
                    break
            if first_piece:
                break
        if all_cleared:
            return True
        for i in range(len(game_board)):
            for j in range(len(game_board[0])):
                if game_board[i][j] == 0:
                    continue
                same_cells = find_same_cell(game_board, i, j)
                step_list = find_step_list(game_board, i, j)
                for same_cell in same_cells:
                    i2, j2 = same_cell
                    if i == i2:
                        if check_match_line(game_board, (i, j), (i2, j2)):
                            new_board = copy_board(game_board)
                            new_board[i][j] = 0
                            new_board[i2][j2] = 0
                            if dfs(new_board):
                                result.append({
                                    'type': 'click',
                                    'from': [i, j],
                                    'to': [i, j],
                                    'match_cell': [i2, j2]
                                })
                                return True
                    elif j == j2:
                        if check_match_line(game_board, (i, j), (i2, j2)):
                            new_board = copy_board(game_board)
                            new_board[i][j] = 0
                            new_board[i2][j2] = 0
                            if dfs(new_board):
                                result.append({
                                    'type': 'click',
                                    'from': [i, j],
                                    'to': [i, j],
                                    'match_cell': [i2, j2]
                                })
                                return True
                    else:
                        for step in step_list:
                            if check_match(game_board, step, same_cell):
                                new_board = copy_board(game_board)
                                move_cell_2(new_board, game_board, (i, j), step)
                                new_board[step[0]][step[1]] = 0
                                new_board[same_cell[0]][same_cell[1]] = 0
                                if dfs(new_board):
                                    result.append({
                                        'type': 'move',
                                        'from': [i, j],
                                        'to': list(step),
                                        'match_cell': list(same_cell)
                                    })
                                    return True
        return False
    ok = dfs(board)
    if not ok:
        return []
    return result[::-1] 