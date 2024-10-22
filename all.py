"""
作者：刘冰昱
2024/10/20 
重构代码结构，并使用该单文件完成全部任务1前三问。想要使用mvc结构，参考网页：https://zh.wikipedia.org/wiki/MVC
M类含有：
    实例：Map抽象类，MazeCreater实现它，主要封装函数
    实例：Agent类，主要封装函数
    地图，运动信息
略，在这里写文档太麻烦。
2024/10/21
我发现我的使用的结构好像是mvp，而不是mvc。但没有关系。
我发现mvp模式好方便，加功能也超级方便
"""
import tkinter as tk
from tkinter import ttk,messagebox
from collections import deque
import heapq
from abc import ABC, abstractmethod
import random
import numpy as np

#configs
MAZE_PATH = "./maze.txt" #地图储存路径
PHUAN = 0.05 #地图生成过程中，敲掉墙形成环的比例 （迷宫是prim算法加随机敲掉墙完成的）。需要小于1
DEFAULT_MAZE_WIDTH = 11 #menu界面默认的几个值
DEFAULT_MAZE_HEIGHT = 11 
DEFAULT_MAZE_SEED = 42  #minmax的agent2的初始位置也用的这个seed
MENU_WIDTH = 400  #menu界面大小
MENU_HEIGHT = 500
MOVE_DELAY = 100 #自动模式agent移动等待时间，毫秒
GENERAL_DELAY = 0 #界面跳转的一般延迟
MINMAX_MAX_STEP = 1000 #minmax允许的最多步数



class Map(ABC):
    """接口类，主要起个封装作用。在该程序中没啥用，其实。"""
    @abstractmethod
    def generate_and_save_map(self,width,height,seed,
                              beginpos=0,endpos=-1,phuan=PHUAN,path=MAZE_PATH):
        """按照参数生成地图到MAZE_PATH文件中"""
        pass
    @abstractmethod
    def load_map(self,path=MAZE_PATH):
        """load地图从MAZE_PATH文件中"""
        pass

class MapCreater(Map):
    """面向过程范式，因为初版就是面向过程的，改的时候懒得改了，反正能用"""
    def __init__(self):
        self.tempMaze = None

    def generate_and_save_map(self,width,height,seed,
                              beginpos=0,endpos=-1,phuan=PHUAN,path=MAZE_PATH):
        # 生成迷宫
        maze, begpoint, endpoint = self.creat_maze(width, height, beginpos, endpos, seed, phuan)
        # 规范化并保存迷宫
        self.norm_maze(maze, begpoint, endpoint, path)

    def load_map(self,path=MAZE_PATH)->list:
        """load地图从MAZE_PATH文件中"""
        with open(path, 'r') as f:
            maze = [line.strip().split(',') for line in f.readlines()]
        return maze
    
    def initialize_maze(self,width, height):
        """初始化迷宫、prim的顶点集合"""
        maze = np.ones((width,height), dtype=int)
        #顶点用二元组表示(x,y)，其中x范围为range(0,width,2),y范围为range(0,height,2)
        vertices = [(x,y) for x in range(0,width,2) for y in range(0,height,2)]
        #将顶点标记-1
        for x,y in vertices:
            maze[x,y] = -1
        return maze,vertices
    
    def add_walls(self,point,width,height,edges,maze):
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
    
    def creat_maze(self,width,height,begpos,endpos,seed,p):
        #初始化地图，顶点
        #迷宫规定： -1表示未被访问的顶点，-2表示被访问过的顶点，
        #           0表示被删除的墙，1表示未被删除的墙。
        random.seed(seed)
        np.random.seed(seed)
        maze,vert = self.initialize_maze(width,height)
        begpoint,endpoint = vert[begpos],vert[endpos]#在prim算法中没啥用

        edges = []  #可选墙壁

        maze[begpoint] = -2  # 将起始点标记为已访问
        self.add_walls(begpoint, width, height, edges, maze)  # 添加初始墙壁

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
                self.add_walls(point, width, height, edges, maze)
        
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

    def norm_maze(self, maze, begpoint, endpoint, path):
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


class Agent:
    def __init__(self):
        """
        该对象的意义是作为一些函数的封装，并没有成员变量
        默认搜索方向：右、下、左、上        
        """
        pass

    def __is_valid(self,x,y,grid):
        rows, cols = len(grid), len(grid[0])
        if 0 <= y < rows and 0 <= x < cols and grid[y][x] != '1':
            return True
        else:
            return False
        
    def __search_goal(self,grid):
        """返回目标的(x,y)，即grid[y][x]=="$" """
        rows, cols = len(grid), len(grid[0])
        goal = None
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == '$':
                    goal = (x,y)
                    return goal
        if not goal:
            raise ValueError("地图不存在终点")

    def use_BFS(self,grid:list,begin_pos=(0,0)):
        """使用广度优先搜索
        返回
        1.(路径长度，搜索一共访问的节点数)
        2.路径，用装有position元组的list表示
        或者，raise错误
        """
        rows, cols = len(grid), len(grid[0])
        start = begin_pos
        goal = None
        directions = [(1,0), (0,1), (-1,0), (0,-1)]  #Right,Down,Left,Up
        goal = self.__search_goal(grid)      # 寻找终点'$'

        # BFS使用的临时变量
        queue = deque([(start, [])])  # (current_position, path_taken)
        visited = set([start])
        nodes_visited = 0   # 访问节点数

        # BFS
        while queue:
            (current_pos, path) = queue.popleft()
            nodes_visited += 1

            # If we reach the goal, return the path and stats
            if current_pos == goal:
                return (len(path) + 1, nodes_visited), path + [current_pos]

            # Explore neighbors
            for direction in directions:
                new_x, new_y = current_pos[0] + direction[0], current_pos[1] + direction[1]
                if self.__is_valid(new_x,new_y,grid):  # Check for valid move
                    new_pos = (new_x, new_y)
                    if new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, path + [current_pos]))

        raise ValueError("终点不可达")
    
    def use_DFS(self, grid: list, begin_pos=(0, 0)):
        """使用深度优先搜索
        返回
        1.(路径长度，搜索一共访问的节点数)
        2.路径，用装有position元组的list表示
        或者，raise错误
        """
        rows, cols = len(grid), len(grid[0])
        start = begin_pos
        goal = None
        directions = [(1,0), (0,1), (-1,0), (0,-1)]  #Right,Down,Left,Up
        directions.reverse()
        goal = self.__search_goal(grid)      # 寻找终点'$'

        # dfs 使用
        stack = [(start, [])]  # 使用栈保存当前路径
        visited = set()  # 访问过的节点
        visited.add(start)
        nodes_visited = 0   # 访问节点数

        while stack:
            current_pos, path = stack.pop()
            nodes_visited += 1 

            # 检查当前节点是否为目标节点
            if current_pos == goal:
                return (len(path) + 1, nodes_visited), path + [current_pos]  # 返回路径长度和路径

            # 遍历所有方向
            for direction in directions:
                new_x = current_pos[0] + direction[0]
                new_y = current_pos[1] + direction[1]
                new_position = (new_x, new_y)
                # 检查新位置的有效性
                if self.__is_valid(new_x,new_y,grid) and new_position not in visited:
                    visited.add(new_position)
                    stack.append((new_position, path + [current_pos]))

        raise ValueError("终点不可达")
    
    def __heuristic(self, a, b):
        """ 计算曼哈顿距离作为启发式函数 """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def use_A_star(self, grid: list, begin_pos=(0, 0)):
        """使用A*搜索"""
        rows, cols = len(grid), len(grid[0])
        start = begin_pos
        goal = self.__search_goal(grid)  # 寻找终点'$'
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 右、下、左、上

        # 优先队列
        open_set = []
        heapq.heappush(open_set, (0 + self.__heuristic(start, goal), start, []))  # (f, position, path)
        visited = set()
        nodes_visited = 0  # 访问节点数

        while open_set:
            f, current_pos, path = heapq.heappop(open_set)
            # 如果当前节点已访问，继续处理下一个节点
            if current_pos in visited:
                continue
            nodes_visited += 1

            # 如果到达目标，返回结果
            if current_pos == goal:
                return (len(path) + 1, nodes_visited), path + [current_pos]

            # 考虑到A*算法中，可能会在加入openset计算过程中，找到到同一个节点的不同f，
            # 所以这里在访问结束后再将其加入到visited中，以允许同一节点以不同f加入优先队列 
            visited.add(current_pos)

            # 探索邻居
            for direction in directions:
                new_x = current_pos[0] + direction[0]
                new_y = current_pos[1] + direction[1]
                new_pos = (new_x, new_y)

                if self.__is_valid(new_x, new_y, grid) and new_pos not in visited:
                    new_cost = len(path) + 1
                    heapq.heappush(open_set, (new_cost + self.__heuristic(new_pos, goal), new_pos, path + [current_pos]))

        raise ValueError("终点不可达")
    
    def use_greedy(self, grid: list, begin_pos=(0, 0)):
        """使用贪心搜索"""
        rows, cols = len(grid), len(grid[0])
        start = begin_pos
        goal = self.__search_goal(grid)  # 寻找终点'$'
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 右、下、左、上

        open_set = []
        heapq.heappush(open_set, (self.__heuristic(start, goal), start, []))  # (f, position, path)
        visited = set()
        nodes_visited = 0  # 访问节点数

        while open_set:
            _, current_pos, path = heapq.heappop(open_set)
            # 如果当前节点已访问，继续处理下一个节点
            if current_pos in visited:
                continue
            nodes_visited += 1

            # 如果到达目标，返回结果
            if current_pos == goal:
                return (len(path) + 1, nodes_visited), path + [current_pos]

            visited.add(current_pos)

            # 探索邻居
            for direction in directions:
                new_x = current_pos[0] + direction[0]
                new_y = current_pos[1] + direction[1]
                new_pos = (new_x, new_y)

                if self.__is_valid(new_x, new_y, grid) and new_pos not in visited:
                    heapq.heappush(open_set, (self.__heuristic(new_pos, goal), new_pos, path + [current_pos]))

        raise ValueError("终点不可达")
    

        

class MazeModel:
    def __init__(self, mapcr:Map, agent:Agent):
        self.mapcr = mapcr
        self.agent = agent
        self.grid = None
        self.agent_position = (0, 0)
        self.way = None

    def init_position(self):
        """清除position和way"""
        self.agent_position = (0, 0)
        self.way = None

    def generate_map(self,width,height,seed):
        """自动生成迷宫的逻辑"""
        self.grid = None
        self.mapcr.generate_and_save_map(width,height,seed)
        self.grid = self.mapcr.load_map()
    
    def move_agent(self,dir:str)->bool:
        """agent移动逻辑"""
        if dir not in ["U","D","L","R"]:
            return False
        move_map = {"U":(0,-1),"D":(0,1),"L":(-1,0),"R":(1,0)}
        added = move_map[dir]
        newx,newy = self.agent_position[0]+added[0],self.agent_position[1]+added[1]
        if self.__is_valid_move(newx,newy):
            self.agent_position = (newx,newy)
            return True
        return False

    def __is_valid_move(self, x, y, grid = None):
        """移动逻辑辅助"""
        if grid:
            ggrid = grid
        else:
            ggrid = self.grid
        if 0 <= x < len(ggrid[0]) and 0 <= y < len(ggrid):
            return ggrid[y][x] in ['0', '$', '@']
        return False

    def get_agent_position(self):
        """获取agent_position"""
        return self.agent_position

    def get_grid(self):
        """获取地图"""
        return self.grid
    
    def get_agent_success(self):
        x,y = self.agent_position
        return self.grid[y][x]=='$'
    
    def get_search_path(self, algorithm:str):
        """根据所选算法搜索路径,返回
        1.grid
        2.(路径长度，搜索一共访问的节点数))
        3.agent_position的list
        或者，raise错误
        """
        if not self.grid:
            raise ValueError("没有加载的地图")
        if self.grid[self.agent_position[1]][self.agent_position[0]] != '@':
            raise ValueError("无效的起点位置")
        
        if algorithm == "BFS":
            try:
                result, path = self.agent.use_BFS(self.grid, self.agent_position)
                return self.grid, result, path
            except ValueError as e:
                raise ValueError(f"算法错误: {e}")
        elif algorithm == "DFS":
            try:
                result, path = self.agent.use_DFS(self.grid, self.agent_position)
                return self.grid, result, path
            except ValueError as e:
                raise ValueError(f"算法错误: {e}")
        elif algorithm == "A_star":
            try:
                result, path = self.agent.use_A_star(self.grid, self.agent_position)
                return self.grid, result, path
            except ValueError as e:
                raise ValueError(f"算法错误: {e}")
        elif algorithm == "greedy":
            try:
                result, path = self.agent.use_greedy(self.grid, self.agent_position)
                return self.grid, result, path
            except ValueError as e:
                raise ValueError(f"算法错误: {e}")
        else:
            raise ValueError("没有找到对应算法")

    def get_minmax_position(self):
        """返回 
        1.能够生成计算agent1、agent2的位置的生成器
        2.返回 生成器，agent1初始位置(0,0)，agent2初始位置
        """
        grid = self.grid
        rows = len(grid)
        cols = len(grid[0])

        # 随机生成agent2的初始position
        tt = 0
        random.seed(DEFAULT_MAZE_SEED)
        while True:
            y = random.randint(rows // 2, rows - 1)
            x = random.randint(cols // 2, cols - 1)
            tt +=1
            # 检查对应点是否为0
            if grid[y][x] == '0':
                agent2position = (x,y)
                break
            if tt == 1000:
                raise ValueError("agent2随机位置生成错误")

        def position_generator(thegrid,agent1_init,agent2_init,obj:MazeModel):
            grid = [row[:] for row in thegrid]
            dir_map = ((0,1),(1,0),(0,-1),(-1,0))
            agent1_pos = agent1_init
            agent2_pos = agent2_init

            while True:
                # agent2移动
                dir = random.randint(0,3)
                dir = dir_map[dir]
                x2,y2 = agent2_pos[0]+dir[0], agent2_pos[1]+dir[1]
                if obj.__is_valid_move(x2,y2,grid) and grid[y2][x2]!='$':
                    agent2_pos = (x2,y2)
                # agent2作为墙
                grid[agent2_pos[1]][agent2_pos[0]] = '1'
                # agent1移动
                try:
                    _,way1 = obj.agent.use_BFS(grid,agent1_pos)
                    agent1_pos = way1[1]
                except ValueError as e:
                    pass
                except IndexError:
                    return
                # 地图复原
                grid[agent2_pos[1]][agent2_pos[0]] = '0'
                yield agent1_pos,agent2_pos
        gen = position_generator(grid,(0,0),agent2position,self)
        return gen,(0,0),agent2position



class MazeView:
    def __init__(self, root):
        self.root = root    #tk.root()
        # menu界面用
        self.menu_frame = None
        # 在迷宫界面用到
        self.cell_size = None
        self.canvas = None
        self.agent_icon = None
        self.agent2_icon = None # agent2
        self.up_canvas_frame = None


    def show_message(self, title: str, message: str):
        """显示提示窗"""
        messagebox.showinfo(title, message)

    def __destory_frame_and_canvas(self):
        """清除所有界面"""
        if self.menu_frame:
            self.menu_frame.destroy()
            self.menu_frame = None
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None
            self.agent_icon = None
            self.agent2_icon = None
        if self.up_canvas_frame:
            # canvas_frame上层信息
            self.up_canvas_frame.destroy()
            self.up_canvas_frame = None

    def creat_and_show_menu(self)->tuple:
        """创建menu界面（每次显示menu都会创造一次），然后将控件返回给controller"""
        self.__destory_frame_and_canvas()
        
        # 创建新的菜单框架
        self.menu_frame = tk.Frame(self.root,width=MENU_WIDTH,height=MENU_HEIGHT)
        self.menu_frame.pack_propagate(False)
        self.menu_frame.pack()
        self.root.title("迷宫游戏菜单")

        # 控件部分
        title_label = tk.Label(self.menu_frame, text="作者：刘冰昱\n代码开头部分有config，有兴趣可以修改")
        # 1.两个按钮
        start_manual_btn = tk.Button(self.menu_frame, text="手动模式", width=20)
        start_auto_btn = tk.Button(self.menu_frame, text="自动模式", width=20)
        # 2.ai选项
        search_algorithms = ["深度优先", "广度优先", "一致代价", "贪心", "A*"]
        algorithm_dropdown = ttk.Combobox(self.menu_frame, values=search_algorithms, state="readonly")
        algorithm_dropdown.current(0)  # 默认选中第一个选项
        # 3.地图选项
        creat_map_btn = tk.Button(self.menu_frame, text="创建地图", width=20)
        seed_label = tk.Label(self.menu_frame, text="地图种子：") #seed
        seed_entry = tk.Entry(self.menu_frame)
        seed_entry.insert(0, str(DEFAULT_MAZE_SEED))
        width_label = tk.Label(self.menu_frame, text="地图宽度：") #宽
        width_entry = tk.Entry(self.menu_frame)
        width_entry.insert(0, str(DEFAULT_MAZE_WIDTH))  # 设置默认宽度值
        height_label = tk.Label(self.menu_frame, text="地图高度：") #高
        height_entry = tk.Entry(self.menu_frame)
        height_entry.insert(0, str(DEFAULT_MAZE_HEIGHT))  # 设置默认高度值
        # 4.minmax
        start_minmax_btn = tk.Button(self.menu_frame, text="MinMax", width=20)

        # 布局
        # title
        title_label.pack(pady=10)
        # 前几个按钮
        start_manual_btn.pack(pady=10)
        start_auto_btn.pack(pady=10)
        algorithm_dropdown.pack(pady=5)
        creat_map_btn.pack(pady=10)
        #地图
        seed_label.pack(pady=5)
        seed_entry.pack(pady=5)
        width_label.pack(pady=5)
        width_entry.pack(pady=5)
        height_label.pack(pady=5)
        height_entry.pack(pady=5)
        #minmax
        start_minmax_btn.pack(pady=10)

        return (start_manual_btn,start_auto_btn,creat_map_btn,start_minmax_btn,
                algorithm_dropdown, seed_entry, width_entry, height_entry)

    def init_maze(self, grid: list, init_agent_position: tuple, path_len=None, search_cost=None):
        """初始化迷宫界面"""
        self.__destory_frame_and_canvas()

        self.cell_size = 30 if max(len(grid[0]),len(grid)*1.3) < 50 else min(1300/len(grid[0]),700/len(grid))
        self.canvas = tk.Canvas(self.root, width=len(grid[0]) * self.cell_size, height=len(grid) * self.cell_size)
        self.canvas.pack()

        self.root.title("x键返回菜单")

        # 绘制迷宫
        color_map = {'1':'black','0':'white','@':'blue','$':'green'}
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                color = color_map[cell]
                self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                              (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                              fill=color, outline="gray")

        # 初始化智能体位置
        x,y = init_agent_position
        self.agent_icon = self.canvas.create_rectangle( x * self.cell_size,     y * self.cell_size,
                                                       (x+1) * self.cell_size, (y+1) * self.cell_size,
                                                       fill="red")
        
        #其他信息path_len和search_cost
        self.up_canvas_frame = tk.Frame(self.root)
        self.up_canvas_frame.pack(pady=5)
        path_len_label = tk.Label(self.up_canvas_frame, text=f"路径长度: {path_len if path_len is not None else 'N/A'}")
        path_len_label.pack(side=tk.LEFT, padx=10)
        search_cost_label = tk.Label(self.up_canvas_frame, text=f"搜索访问节点总数: {search_cost if search_cost is not None else 'N/A'}")
        search_cost_label.pack(side=tk.LEFT, padx=10)
    
    def update_agent_position(self, new_position: tuple):
        """更新agent在迷宫的位置"""
        # 更新智能体的图标位置
        x,y = new_position
        self.canvas.coords(self.agent_icon,
                           x * self.cell_size, y * self.cell_size,
                           (x+1)*self.cell_size, (y+1)*self.cell_size)

    def update_or_init_agent2_position(self, position: tuple):
        """更新agent2在迷宫的位置"""
        # 更新智能体的图标位置
        # init agent2
        if self.agent2_icon == None:
            x,y = position
            self.agent2_icon = self.canvas.create_rectangle( x * self.cell_size,     y * self.cell_size,
                                                       (x+1) * self.cell_size, (y+1) * self.cell_size,
                                                       fill="purple")
        else:
            x,y = position
            self.canvas.coords(self.agent2_icon,
                            x * self.cell_size, y * self.cell_size,
                            (x+1)*self.cell_size, (y+1)*self.cell_size)

class MazeController:
    def __init__(self, model:MazeModel, view:MazeView):
        self.model = model
        self.view = view
        self.root = view.root
        # menu信息绑定
        self.algorithm_dropdown = None
        self.seed_entry = None
        self.width_entry = None
        self.height_entry = None


    def BeginAll(self):
        """主函数将把此函数注册到初始的mainloop中"""
        # self.view.show_message("提示","代码最前面有config，有需要可以改  作者：刘冰昱")
        self.root.after(GENERAL_DELAY, self.menu_control)

    def menu_control(self):
        """调用view，来呈现界面，并将按钮绑定到具体的操作中"""
        self.algorithm_dropdown = None
        self.seed_entry = None
        self.width_entry = None
        self.height_entry = None

        manu_btn,auto_btn,map_btn,minmax_btn,\
            algorithm_dropdown, seed_entry, width_entry, height_entry = self.view.creat_and_show_menu()

        manu_btn.config(command=self.start_manual_mode)
        auto_btn.config(command=self.start_auto_mode)
        map_btn.config(command=self.creat_map)
        minmax_btn.config(command=self.start_minmax_mode)

        # 将下拉框和输入框存储在控制器中，以便后续访问
        self.algorithm_dropdown = algorithm_dropdown
        self.seed_entry = seed_entry
        self.width_entry = width_entry
        self.height_entry = height_entry

    def creat_map(self):
        """创建地图功能"""
        # 清除agent相关
        self.model.init_position()
        # 获取用户输入的地图种子、宽度和高度
        try:
            seed = int(self.seed_entry.get())
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
        except Exception:
            self.view.show_message("错误","请正确输入seed、width、height")
            return
        if seed==None or width==None or height==None:
            self.view.show_message("错误","请正确输入seed、width、height")
            return

        try:
            self.model.generate_map(width, height, seed)
        except Exception as e:
            self.view.show_message("错误","未知原因，无法创建地图，已停止程序，请看命令行错误提示")
            raise e
        else:
            self.view.show_message("成功",
                               f"创建地图：种子={seed}, 宽度={width}, 高度={height}")
            return

    def start_manual_mode(self):
        """手动模式控制"""
        # 没有地图则退出
        if self.model.get_grid() == None:
            self.view.show_message("失败","当前没有地图")
            return False
        
        # 定义需要的callback的闭包函数
        def handle_keypress(event):
            """处理键盘按下事件"""
            key_mapping = {'Up':'U', 'Down':'D', 'Left':'L', 'Right':'R', 'x':'x', 'X':'x'}
            direction = key_mapping.get(event.keysym)
            
            # 行走
            if direction and direction!='x':
                self.model.move_agent(direction)  # 移动智能体
                new_position = self.model.get_agent_position()  # 获取新的位置
                self.view.update_agent_position(new_position)  # 更新视图
                # 成功提示
                if self.model.get_agent_success():
                    self.view.show_message("成功","到达终点")
            # 退出
            elif direction and direction=='x':
                self.root.unbind("<KeyPress>")  # 解除键盘事件绑定
                self.root.after(GENERAL_DELAY,self.backto_menu)
        
        # 初始化
        self.model.init_position()
        inited_position = self.model.get_agent_position()
        grid = self.model.get_grid()
        self.view.init_maze(grid,inited_position)
        # 操作提示
        self.view.show_message("提示","上下左右控制移动，x键退出")
        # 绑定按键
        self.root.bind("<KeyPress>", handle_keypress)
        
    def start_auto_mode(self):
        """自动模式控制"""
        # 没有地图则退出
        if self.model.get_grid() == None:
            self.view.show_message("失败","当前没有地图")
            return False

        # 初始化，得到路径
        self.model.init_position()
        selected_option = self.algorithm_dropdown.get()
        name_map = {"广度优先":"BFS","深度优先":"DFS","A*":"A_star","一致代价":"BFS","贪心":"greedy"}
        selected_option_trans = name_map.get(selected_option)
        if selected_option_trans:
            try:
                grid, result, path = self.model.get_search_path(selected_option_trans)
            except Exception as e:
                self.view.show_message("失败",f"agent算法计算错误，信息:{e}")
                return False
        else:
            self.view.show_message("失败","agent算法选项错误")
            return False

        # 初始化界面
        self.model.init_position()
        inited_position = self.model.get_agent_position()
        self.view.init_maze(grid,inited_position,result[0],result[1])
        # 操作提示
        self.view.show_message("提示",f"现在运行{selected_option},x键退出")

        # 闭包函数完成主循环
        path_counter = 0
        early_termination = False # 提前终止标志，能被handle_keypress事件修改，从而停止loop
        def move_loop():
            nonlocal path_counter
            if path_counter>=len(path):
                return
            if early_termination:
                return

            # 行走，此处的行走与Modle的信息完全无关
            new_position = path[path_counter]
            self.view.update_agent_position(new_position)   # 更新视图

            #注册下一个循环
            self.root.after(MOVE_DELAY, move_loop)
            path_counter +=1

        # 定义需要按键触发的动作
        def handle_keypress(event):
            """处理键盘按下事件，只有退出
                幸好逻辑不复杂，否则不用await会导致回调地狱吧
            """
            key_mapping = {'x':'x', 'X':'x'}
            direction = key_mapping.get(event.keysym)
            # 退出
            if direction and direction=='x':
                self.root.unbind("<KeyPress>")  # 解除键盘事件绑定
                nonlocal early_termination  # 发出提前终止信号
                early_termination = True
                self.root.after(GENERAL_DELAY,self.backto_menu)

        # 绑定按键
        self.root.bind("<KeyPress>", handle_keypress)
        #移动loop触发
        self.root.after(1000, move_loop)

    def start_minmax_mode(self):
        """minmax对抗agent模式"""
         # 没有地图则退出
        if self.model.get_grid() == None:
            self.view.show_message("失败","当前没有地图")
            return False

        # 初始化，得到路径生成器，注意，不保证生成器不是无限循环
        try:
            agent12_generator,init_a1,init_a2 = self.model.get_minmax_position()
        except Exception as e:
            self.view.show_message("失败",f"minmax算法计算错误，信息:{e}")
            return False

        # 初始化界面
        grid = self.model.get_grid()
        self.view.init_maze(grid,init_a1)
        self.view.update_or_init_agent2_position(init_a2)
        # 操作提示
        self.view.show_message("提示",f"minmax实时计算，不保证程序能终止，按x键随时退出")

        # 闭包函数完成主循环
        path_counter = 0
        early_termination = False # 提前终止标志，能被handle_keypress事件修改，从而停止loop
        def move_loop():
            nonlocal path_counter
            nonlocal agent12_generator

            if path_counter>=MINMAX_MAX_STEP:
                return
            if early_termination:
                return
            path_counter +=1

            # 行走，此处的行走由迭代器实时计算
            try:
                agent1_pos, agent2_pos = next(agent12_generator)
            except StopIteration:
                self.view.show_message("提示",f"minmax结束，按x键退出")
                return
            # 更新视图
            self.view.update_agent_position(agent1_pos)   
            self.view.update_or_init_agent2_position(agent2_pos)

            #注册下一个循环
            self.root.after(MOVE_DELAY, move_loop)
            

        # 定义需要按键触发的动作
        def handle_keypress(event):
            """处理键盘按下事件，只有退出
                幸好逻辑不复杂，否则不用await会导致回调地狱吧
            """
            key_mapping = {'x':'x', 'X':'x'}
            direction = key_mapping.get(event.keysym)
            # 退出
            if direction and direction=='x':
                self.root.unbind("<KeyPress>")  # 解除键盘事件绑定
                nonlocal early_termination  # 发出提前终止信号
                early_termination = True
                self.root.after(GENERAL_DELAY,self.backto_menu)

        # 绑定按键
        self.root.bind("<KeyPress>", handle_keypress)
        #移动loop触发
        self.root.after(1000, move_loop)

    def backto_menu(self):
        """返回目录选项"""
        self.root.after(GENERAL_DELAY,self.menu_control)

if __name__ == "__main__":
    agenter = Agent()
    mapper = MapCreater()
    model = MazeModel(mapper,agenter)          # 创建模型

    root = tk.Tk()
    view = MazeView(root)   # 创建视图，传入root

    controller = MazeController(model, view)  # 创建控制器，连接模型和视图

    #迷宫，启动！
    root.after(GENERAL_DELAY, controller.BeginAll)
    root.mainloop()