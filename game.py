import tkinter as tk

# 读取迷宫文件并解析成二维列表
def load_maze(filename):
    with open(filename, 'r') as f:
        maze = [line.strip().split(',') for line in f.readlines()]
    return maze

#查找起点位置
def get_start(maze):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == '@':
                return [x, y]

#游戏界面
class MazeGui:
    def __init__(self, gameinfo):
        """绑定Tk对象和maze，用maze初始化界面"""
        master,maze,pos = gameinfo
        self.master = master
        self.maze = maze

        self.agentpos = pos
        self.rect = None
        
        self.cell_size = 30 if max(len(maze[0]),len(maze)*1.3) < 70 else 17
        self.canvas = tk.Canvas(master, width=len(maze[0]) * self.cell_size+10, 
                                height=len(maze) * self.cell_size+10)
        self.canvas.pack()
        
        self.__draw_maze_and_agent()

    #绘制迷宫和agent(只运行一次)
    def __draw_maze_and_agent(self):
        self.canvas.delete("all")
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == '1':
                    self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                                 (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                                 fill="black")
                elif cell == '0':
                    self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                                 (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                                 fill="white")
                elif cell == '@':
                    self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                                 (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                                 fill="blue")
                elif cell == '$':
                    self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                                 (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                                 fill="green")
        x, y = self.agentpos
        self.rect = self.canvas.create_rectangle(x*self.cell_size, y*self.cell_size,
                                                    (x+1)*self.cell_size, (y+1)*self.cell_size,
                                                    fill="red")

    def step(self):
        """更新agent"""
        x,y = self.agentpos
        self.canvas.coords(self.rect,
                           x*self.cell_size, y*self.cell_size,
                           (x+1)*self.cell_size, (y+1)*self.cell_size)

#游戏引擎  
class GameEngineer:
    def __init__(self,gui,gameinfo):
        self.gui = gui
        master, self.maze, self.agent_position = gameinfo

        #绑定按键
        master.bind("<Key>", self.key_handler)
    
    # 处理方向
    def process_dir(self, direction):
        x, y = self.agent_position
        if direction == 'Up':
            self.move_agent(x, y - 1)
        elif direction == 'Down':
            self.move_agent(x, y + 1)
        elif direction == 'Left':
            self.move_agent(x - 1, y)
        elif direction == 'Right':
            self.move_agent(x + 1, y)

    # 处理键盘输入
    def key_handler(self, event):
        self.process_dir(event.keysym)

    # 虚拟键盘输入：使用after来安排事件
    def virtual_keyboard(self, event, delay=500):
        direction_map = {'U': 'Up', 'D': 'Down', 'L': 'Left', 'R': 'Right'}
        if isinstance(event,list):
            for i,eventi in enumerate(event):
                eventi = direction_map[eventi]
                self.gui.master.after(delay*(i+1),self.process_dir,eventi)
        else:
            """下面这段估计会有bug，因为进不了事件循环"""
            pass
            # event = direction_map[event]
            # self.process_dir(event)
        
            

    # 移动 agent
    def move_agent(self, new_x, new_y):
        if self.is_valid_move(new_x, new_y):
            self.agent_position[0] = new_x
            self.agent_position[1] =new_y
            if self.maze[new_y][new_x] == '$':
                print("You reached the end!")
            self.gui.step()

    # 检查移动是否合法
    def is_valid_move(self, x, y):
        if 0 <= x < len(self.maze[0]) and 0 <= y < len(self.maze):
            return self.maze[y][x] in ['0', '$']
        return False

if __name__ == "__main__":
    root = tk.Tk()
    maze = load_maze("maze.txt")
    position = get_start(maze)
    gameinfo = (root,maze,position)

    gui = MazeGui(gameinfo)
    gc = GameEngineer(gui,gameinfo)
    gc.virtual_keyboard(list("DDRRRRRR"))
    root.mainloop()
    
    