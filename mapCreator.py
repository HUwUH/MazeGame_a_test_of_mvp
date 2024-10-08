import tkinter as tk
import random
import numpy as np
#configs
SEED = 42   #随机种子
WIDTH = 71  #迷宫宽度
HEIGHT = 41 #迷宫高度
BEGPOS = 0  #起始顶点号(于vertices)
ENDPOS = -1 #终止顶点号(于vertices)
P_HUAN = 0.35 #出现环的概率


def initialize_maze(width, height):
    """初始化迷宫、prim的顶点集合"""
    maze = np.ones((width,height), dtype=int)
    #顶点用二元组表示(x,y)，其中x范围为range(0,width,2),y范围为range(0,height,2)
    vertices = [(x,y) for x in range(0,width,2) for y in range(0,height,2)]
    #将顶点标记-1
    for x,y in vertices:
        maze[x,y] = -1
    return maze,vertices

def add_walls(point,width,height,edges,maze):
    """将vertices的可选墙壁添加到墙壁列表中"""
    #检测该点上下左右的墙，如果墙为1，且墙对面的顶点未被访问过，则将墙加入到edges中
    x, y = point
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx,dy in directions:
        wall_x = x+dx
        wall_y = y+dy
        vect_x = x+2*dx
        vect_y = y+2*dy
        #邻接点坐标必须在迷宫的边界内
        if 0<=vect_x and vect_x<width and 0<=vect_y and vect_y<height:
            if maze[wall_x, wall_y]==1 and maze[vect_x,vect_y]==-1:  
                #如果 墙是未被删除 且 对面点未被访问
                #那么将墙加入到 edges 中
                edges.append((wall_x, wall_y))
    return None

def creat_maze(width,height,begpos,endpos,seed,p):
    #初始化地图，顶点
    #迷宫规定： -1表示未被访问的顶点，-2表示被访问过的顶点，
    #           0表示被删除的墙，1表示未被删除的墙。
    random.seed(seed)
    np.random.seed(seed)
    maze,vert = initialize_maze(width,height)
    begpoint,endpoint = vert[begpos],vert[endpos]#在prim算法中没啥用

    edges = []  #可选墙壁

    maze[begpoint] = -2  # 将起始点标记为已访问
    add_walls(begpoint, width, height, edges, maze)  # 添加初始墙壁

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while edges:
        # 随机选择一条边
        wall = random.choice(edges)
        edges.remove(wall)  #从edges中移除该边
        wall_x, wall_y = wall

        # 获取边（墙）四周两个顶点
        neighboring_points = []
        for dx, dy in directions:
            vect_x = wall_x + dx
            vect_y = wall_y + dy
            # 检查邻接点是否在边界内，并且是未被访问的点
            if 0 <= vect_x < width and 0 <= vect_y < height:
                if maze[vect_x, vect_y] == -1:
                    neighboring_points.append((vect_x, vect_y))
        
        # 如果标记正常，那么删除墙
        if len(neighboring_points) == 2:
            pass
        elif len(neighboring_points) == 1:
            maze[wall_x, wall_y] = 0
        elif len(neighboring_points) != 0:
            raise "错误"

        # 将两个非1的点使用add_walls，并标记为-2
        for point in neighboring_points:
            px, py = point
            maze[px, py] = -2
            add_walls(point, width, height, edges, maze)
    
    # 随机删除p次墙
    for _ in range(0,int(p*width*height/4)):
        x,y,z = random.randint(0,width-2),random.randint(0,height-2),random.random()
        if z<0.5:
            x = int(x/2)*2+1
            y = int(y/2)*2
        else:
            x = int(x/2)*2
            y = int(y/2)*2+1
        maze[x,y]=0

    # 设置终点
    maze[endpoint] = -2  # 将终点标记为已访问
    return maze,begpoint,endpoint

def norm_maze(maze, begpoint, endpoint, path):
    """清洗，并储存地图"""
    # 使用list存储字符: -1,-2,0均变为'0',1变为'1',begpoint,endpoint分别变成'@','$'
    maze = maze.T
    char_maze = []
    for row in maze:
        char_row = []
        for value in row:
            if value == -1 or value == -2 or value == 0:
                char_row.append('0')  # 未访问和已访问的点、通路
            elif value == 1:
                char_row.append('1')  # 墙
        char_maze.append(char_row)
    char_maze[begpoint[1]][begpoint[0]] = "@"
    char_maze[endpoint[1]][endpoint[0]] = "$"

    # 字符数组转换为字符串
    char_maze_str = '\n'.join([','.join(row) for row in char_maze])

    # 保存到指定路径
    with open(path, 'w') as file:
        file.write(char_maze_str)

def allCreatmap(width,height,begpos,endpos,seed,path,p_rotate):
    # 生成迷宫
    maze, begpoint, endpoint = creat_maze(width, height, begpos, endpos, seed, p_rotate)
    # 规范化并保存迷宫
    norm_maze(maze, begpoint, endpoint, path)

if __name__ == "__main__":
    #配置参数
    width = WIDTH  # 迷宫宽度
    height = HEIGHT  # 迷宫高度
    begpos = BEGPOS  # 起始顶点号
    endpos = ENDPOS  # 终止顶点号
    seed = SEED  # 随机种子
    path = "maze.txt"  # 保存迷宫的文件路径
    p = P_HUAN  # 出现环的概率
    
    # 生成迷宫
    maze, begpoint, endpoint = creat_maze(width, height, begpos, endpos, seed, p)
    
    # 打印生成的迷宫
    print("生成的迷宫：")
    for row in maze:
        print(" ".join(str(val) for val in row))

    # 规范化并保存迷宫
    norm_maze(maze, begpoint, endpoint, path)
