from flask import Flask, request, jsonify
from board_parser import parse_board_image
from solver import solve_board, move_cell_2
from utils import copy_board
import os
import numpy as np
import cv2

app = Flask(__name__, static_folder='static')

@app.route('/api/parse_board_image', methods=['POST'])
def parse_board_image_api():
    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    board, type_imgs = parse_board_image(img)
    type_imgs_map = {str(idx+1): f"/static/type_imgs/{img['name']}.png" for idx, img in enumerate(type_imgs)}
    return jsonify({
        "board": board,
        "typeImgs": type_imgs_map
    })

@app.route('/api/solve_board', methods=['POST'])
def solve_board_api():
    data = request.get_json()
    board = data['board']
    actions = solve_board(board)
    board_list = []
    cur_board = copy_board(board)
    board_list.append({
        "board": copy_board(cur_board),
        "action": actions[0] if actions else {}
    })
    for idx, action in enumerate(actions):
        prev_board = copy_board(cur_board)
        if action['type'] == 'click':
            i1, j1 = action['from']
            i2, j2 = action['match_cell']
            cur_board[i1][j1] = 0
            cur_board[i2][j2] = 0
        elif action['type'] == 'move':
            i1, j1 = action['from']
            i2, j2 = action['to']
            i3, j3 = action['match_cell']
            move_cell_2(cur_board, prev_board, (i1, j1), (i2, j2))
            cur_board[i2][j2] = 0
            cur_board[i3][j3] = 0
        next_action = actions[idx+1] if idx+1 < len(actions) else {}
        board_list.append({
            "board": copy_board(cur_board),
            "action": next_action
        })
    return jsonify({
        "actions": actions,
        "board_list": board_list
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050) 