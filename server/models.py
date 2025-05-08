from dataclasses import dataclass

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
    cell_type: int # 棋子的类型

@dataclass
class TypeImg:
    img: any
    name: str
    img_type: int 