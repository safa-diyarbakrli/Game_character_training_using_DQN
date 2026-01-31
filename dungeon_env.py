import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Optional, Tuple, List
import pygame
from os import path

MAP = [
    "+-------------+",
    "|M: : : : :B|F|",
    "| : : : : : : |",
    "|M: |-:-: : |B|",
    "|R:B| : : :-| |",
    "|-:-| : :M: :B|",
    "| : : :-:-: :-|",
    "|P: : : :R:M:L|",
    "+-------------+",
]

WINDOW_SIZE = (526, 350)

class Fireball:
    """Represents a fireball projectile, manages its own position, direction, and boundary checks."""
    def __init__(self, pos: Tuple[float, float], direction: Tuple[float, float], speed: float = 0.5):
        self.x, self.y = pos # Continuous position
        self.dx, self.dy = direction # Direction vector 
        self.speed = speed
        self.active = True
        
    def update(self):
        """Update fireball position,
         new_pos = old_pos + direction * speed  """
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        
    def get_grid_pos(self) -> Tuple[int, int]:
        """Convert float coordinates to the nearest integer grid cell."""
        return (int(round(self.y)), int(round(self.x)))
        
    def is_out_of_bounds(self, max_row: int, max_col: int) -> bool:
        """Check if fireball is out of bounds and destroyes the fireball if it leaves the map."""
        return not (0 <= self.x <= max_col and 0 <= self.y <= max_row)
    
class DungeonDQNEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }
    
    def __init__(self, render_mode: Optional[str] = None, difficulty: float = 1.0):
        super().__init__()
        
        # Map configuration
        self.desc = np.asarray(MAP, dtype="c")
        self.num_rows = 7
        self.num_columns = 7
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        
        # Game locations (static)
        self.start = (6, 0)
        self.restricted = {(3, 0), (6, 4)}
        self.finishLine = (0, 6)
        self.loot = (6, 6)
        self.boss_spawn_locs = [(0, 5), (2, 6), (3, 1), (4, 6)]
        self.mini_boss_spawn_locs = [(0, 0), (2, 0), (4, 4), (6, 5)]
        
        # CURRICULUM LEARNING
        self.difficulty = difficulty  # 0.0 to 1.0
        
        # Boss movement
        base_boss_move = 0.10
        base_mini_move = 0.15
        # Boss movement frequency increases with difficulty
        self.boss_move_frequency = base_boss_move + (difficulty * 0.10)
        self.mini_boss_move_frequency = base_mini_move + (difficulty * 0.15)
        
        # Fireball parameters
        base_spawn_rate = 0.05
        self.fireball_spawn_rate = base_spawn_rate + (difficulty * 0.10) #increases with difficulty 
        self.fireball_damage = 1 # Damage per fireball hit (1 HP)
        
        
        # Fireball limits (ensures the neural network input size is constant)
        self.HARD_MAX_FIREBALLS = 5 
        self.current_max_fireballs = max(2, int(3 + difficulty * 2)) #ranges from 2 (easy) to 5 (full difficulty)
        self.current_max_fireballs = min(self.current_max_fireballs, self.HARD_MAX_FIREBALLS) #ensures max fireballs does not exceed HARD_MAX_FIREBALLS for the NN
        
        # Game state variables
        self.player_row = None
        self.player_col = None
        self.boss_row = None
        self.boss_col = None
        self.mini_boss_row = None
        self.mini_boss_col = None
        self.has_loot = None
        self.has_mini = None
        self.boss_alive = None
        self.health = None
        self.fireballs: List[Fireball] = [] # List of active fireballs
        
        # Observation space 
        
        """ Calculate observation size:
        10 base features + (5 fireballs × 4 features each) = 30 dimensions """
        obs_size = 10 + (self.HARD_MAX_FIREBALLS * 4) 
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space (5 discrete actions: 0=South, 1=North, 2=East, 3=West, 4=Attack)
        self.action_space = spaces.Discrete(5)
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.num_columns,
            WINDOW_SIZE[1] / self.num_rows,
        )
        
        # Images
        self.player_img = None
        self.loot_image = None
        self.boss_img = None
        self.miniBoss_img = None
        self.finishLine_img = None
        self.restricted_img = None
        self.background_img = None
        self.walls_img = None
        self.fireball_img = None
        
    def _normalize_position(self, row: int, col: int) -> Tuple[float, float]:
        """Normalize position to [-1, 1] range"""
        norm_row = (row / self.max_row) * 2 - 1
        norm_col = (col / self.max_col) * 2 - 1
        return norm_row, norm_col
    
    def _get_observation(self) -> np.ndarray:
        """
        Constructs the state vector.
        Pads the fireball data with zeros if fewer than HARD_MAX_FIREBALLS exist.
        [0-1]:   Player position (normalized)
        [2-3]:   Main boss position (normalized)
        [4-5]:   Mini-boss position (normalized)
        [6]:     has_loot flag (0 or 1)
        [7]:     has_mini_power flag (0 or 1)
        [8]:     boss_alive flag (0 or 1)
        [9]:     health (normalized to [0, 1])
        [10-29]: Fireball data (5 fireballs × 4 features each)
        """
        
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # --- Player position (indices 0-1) ---
        obs[0], obs[1] = self._normalize_position(self.player_row, self.player_col)
        
         # --- Main boss position (indices 2-3) ---
        if self.boss_alive:
            obs[2], obs[3] = self._normalize_position(self.boss_row, self.boss_col)
        else:
            obs[2], obs[3] = 0.0, 0.0
            
        # --- Mini-boss position (indices 4-5) ---    
        if not self.has_mini:
            obs[4], obs[5] = self._normalize_position(self.mini_boss_row, self.mini_boss_col)
        else:
            obs[4], obs[5] = 0.0, 0.0
        
        # --- Binary flags (indices 6-8) ---
        obs[6] = float(self.has_loot)
        obs[7] = float(self.has_mini)
        obs[8] = float(self.boss_alive)
        
        # --- Health (index 9) ---
        obs[9] = self.health / 3.0
        
        # --- Fireball data (indices 10-29) ---
        for i, fireball in enumerate(self.fireballs[:self.HARD_MAX_FIREBALLS]):
            base_idx = 10 + (i * 4)
            norm_y, norm_x = self._normalize_position(int(fireball.y), int(fireball.x))
            obs[base_idx] = norm_x
            obs[base_idx + 1] = norm_y
            obs[base_idx + 2] = fireball.dx
            obs[base_idx + 3] = fireball.dy
            
        return obs
    
    def _can_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Check if movement is allowed"""
        if from_col == to_col:
            if to_row > from_row:
                return self.desc[1 + from_row + 1, 2 * from_col + 1] != b"-"
            else:
                return self.desc[1 + from_row, 2 * from_col + 1] != b"-"
        elif from_row == to_row:
            if to_col > from_col:
                return self.desc[1 + from_row, 2 * from_col + 2] == b":"
            else:
                return self.desc[1 + from_row, 2 * from_col] == b":"
        return True
    
    def _get_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get a list of all valid adjacent positions reachable from current position."""
        valid = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row <= self.max_row and 
                0 <= new_col <= self.max_col and
                (new_row, new_col) not in self.restricted and
                self._can_move(row, col, new_row, new_col)):
                valid.append((new_row, new_col))
                
        return valid
    
    def _move_toward_target(self, from_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Move one step toward target using greedy best-first search (used by the bosses to chase the agent)
            1. Get all valid adjacent cells
            2. For each, calculate Manhattan distance to target
            3. Choose the cell with minimum distance
            4. If no valid moves (surrounded), stay in place
            """
        from_row, from_col = from_pos
        target_row, target_col = target_pos
        
        valid_moves = self._get_valid_moves(from_row, from_col)
        if not valid_moves:
            return from_pos # Stay in place if trapped
        
        best_move = from_pos
        best_dist = abs(from_row - target_row) + abs(from_col - target_col) #Manhattan Distance
        
        for new_row, new_col in valid_moves:
            dist = abs(new_row - target_row) + abs(new_col - target_col)
            if dist < best_dist:
                best_dist = dist
                best_move = (new_row, new_col)
        
        return best_move
    
    def _update_boss_positions(self):
        """Update bosses positions"""
        player_pos = (self.player_row, self.player_col)
        
        if self.boss_alive and random.random() < self.boss_move_frequency:
            new_pos = self._move_toward_target((self.boss_row, self.boss_col), player_pos)
            self.boss_row, self.boss_col = new_pos
        
        if not self.has_mini and random.random() < self.mini_boss_move_frequency:
            new_pos = self._move_toward_target((self.mini_boss_row, self.mini_boss_col), player_pos)
            self.mini_boss_row, self.mini_boss_col = new_pos
    
    def _has_line_of_sight(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Check if there's a clear line of sight between two positions. (fireball mechanics)

            1. Determine if path is more horizontal (dx > dy) or vertical (dy > dx)
            2. Trace along the major axis, incrementing minor axis when error accumulates
            3. At each step, check if movement is blocked by wall or lava
            4. Return False immediately if any obstacle found

        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        dx = abs(to_col - from_col)
        dy = abs(to_row - from_row)
        
        x = from_col
        y = from_row
        
        x_inc = 1 if to_col > from_col else -1
        y_inc = 1 if to_row > from_row else -1
        
        # More horizontal movement (dx > dy)
        if dx > dy:
            error = dx / 2
            while x != to_col:
                next_x = x + x_inc
                if not self._can_move(y, x, y, next_x):
                    return False
                if (y, next_x) in self.restricted:
                    return False
                
                error -= dy
                if error < 0:
                    next_y = y + y_inc
                    if not self._can_move(y, x, next_y, x):
                        return False
                    if (next_y, x) in self.restricted:
                        return False
                    y = next_y
                    error += dx
                x = next_x
        else: # More vertical movement (dy >= dx)
            error = dy / 2
            while y != to_row:
                next_y = y + y_inc
                if not self._can_move(y, x, next_y, x):
                    return False
                if (next_y, x) in self.restricted:
                    return False
                
                error -= dx
                if error < 0:
                    next_x = x + x_inc
                    if not self._can_move(y, x, y, next_x):
                        return False
                    if (y, next_x) in self.restricted:
                        return False
                    x = next_x
                    error += dy
                y = next_y
        
        return True
    

    def _spawn_fireball(self):
        """
        Attempt to spawn a fireball from the main boss toward the player.
          CONSTRAINTS (to ensure fair gameplay):
                1. Range limit: Player must be within 5 cells (Manhattan distance)
                2. Line of sight: No walls blocking between boss and player
                3. 8-direction movement: Quantize direction to cardinal/diagonal only
                4. Speed limit: 0.35 cells/step (slower than player for dodgeability)
                5. Max fireballs: Respect current_max_fireballs limit
          If any constraint fails, no fireball is spawned.
        """
        
        # CONSTRAINT 5: Check if we've reached maximum fireballs
        if len(self.fireballs) >= self.current_max_fireballs:
            return
        
        # CONSTRAINT 1: Range check (Manhattan distance)
        manhattan_dist = abs(self.player_row - self.boss_row) + abs(self.player_col - self.boss_col)
        MAX_RANGE = 5
        
        if manhattan_dist > MAX_RANGE:
            return  # Player too far away
        
       # CONSTRAINT 2: Check if boss has line of sight to player
        if not self._has_line_of_sight(
            (self.boss_row, self.boss_col),
            (self.player_row, self.player_col)
        ):
            return  # Wall blocking, can't shoot
        
        # Calculate direction
        dx = self.player_col - self.boss_col
        dy = self.player_row - self.boss_row
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            # Normalize to unit vector
            norm_dx = dx / distance
            norm_dy = dy / distance
            # CONSTRAINT 3: Snap to nearest of 8 cardinal/diagonal directions
            angle = np.arctan2(norm_dy, norm_dx)
            direction_index = int(np.round(angle / (np.pi / 4))) % 8
            
            # 8 directions: E, NE, N, NW, W, SW, S, SE
            directions = [
                (1.0, 0.0),         # East
                (0.707, -0.707),    # NorthEast  
                (0.0, -1.0),        # North
                (-0.707, -0.707),   # NorthWest
                (-1.0, 0.0),        # West
                (-0.707, 0.707),    # SouthWest
                (0.0, 1.0),         # South
                (0.707, 0.707)      # SouthEast
            ]
            
            final_dx, final_dy = directions[direction_index]
            
            # CONSTRAINT 4: Slower speed for dodging
            fireball = Fireball(
                pos=(float(self.boss_col), float(self.boss_row)),
                direction=(final_dx, final_dy),
                speed=0.35  # Slower than before (was 0.5)
            )
            self.fireballs.append(fireball)
    
    def _update_fireballs(self):
        """Update all fireballs and check collisions"""
        for fireball in self.fireballs[:]:
            old_x, old_y = fireball.x, fireball.y
            old_grid = (int(round(old_y)), int(round(old_x)))
            
            fireball.update()
            new_grid = fireball.get_grid_pos()
            
            if fireball.is_out_of_bounds(self.max_row, self.max_col):
                self.fireballs.remove(fireball)
                continue
            
            # Check wall collision
            if old_grid != new_grid:
                if not self._can_move(old_grid[0], old_grid[1], new_grid[0], new_grid[1]):
                    self.fireballs.remove(fireball)
                    continue
            
            if new_grid in self.restricted:
                self.fireballs.remove(fireball)
                continue
    
    def _check_fireball_collision(self) -> bool:
        """Check if any fireball hit the player"""
        player_pos = (self.player_row, self.player_col)
        
        for fireball in self.fireballs[:]:
            grid_pos = fireball.get_grid_pos()
            if grid_pos == player_pos:
                self.fireballs.remove(fireball)
                return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.player_row, self.player_col = self.start
        
        boss_idx = random.randint(0, len(self.boss_spawn_locs) - 1)
        mini_idx = random.randint(0, len(self.mini_boss_spawn_locs) - 1)
        
        self.boss_row, self.boss_col = self.boss_spawn_locs[boss_idx]
        self.mini_boss_row, self.mini_boss_col = self.mini_boss_spawn_locs[mini_idx]
        
        self.has_loot = 0
        self.has_mini = 0
        self.boss_alive = 1
        self.health = 3
        self.fireballs = []
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """Execute one step - reward structure"""
        row, col = self.player_row, self.player_col
        reward = -0.05
        terminated = False
        
        new_row, new_col = row, col
        
        # MOVEMENT
        if action == 0:
            new_row = min(row + 1, self.max_row)
        elif action == 1:
            new_row = max(row - 1, 0)
        elif action == 2:
            new_col = min(col + 1, self.max_col)
        elif action == 3:
            new_col = max(col - 1, 0)
        
        # Check walls
        if (new_row, new_col) in self.restricted or not self._can_move(row, col, new_row, new_col):
            new_row, new_col = row, col
            reward -= 0.3
        
        self.player_row, self.player_col = new_row, new_col
        player_loc = (new_row, new_col)
        
        # LOOT COLLECTION
        if player_loc == self.loot and self.has_loot == 0:
            self.has_loot = 1
            reward += 25
        
        # ATTACK ACTION
        if action == 4:
            attacked_something = False
            
            if player_loc == (self.mini_boss_row, self.mini_boss_col) and self.has_mini == 0:
                attacked_something = True
                if self.has_loot == 1:
                    self.has_mini = 1
                    reward += 35
                else:
                    self.health -= 1
                    reward -= 15
            
            elif player_loc == (self.boss_row, self.boss_col) and self.boss_alive == 1:
                attacked_something = True
                if self.has_loot == 1 and self.has_mini == 1:
                    self.boss_alive = 0
                    reward += 50
                else:
                    self.health -= 1
                    reward -= 15
            
            if not attacked_something:
                reward -= 0.5
        
        # Update environment
        self._update_boss_positions()
        
        if self.boss_alive and random.random() < self.fireball_spawn_rate:
            self._spawn_fireball()
        
        self._update_fireballs()
        
        # FIREBALL COLLISION
        if self._check_fireball_collision():
            self.health -= self.fireball_damage
            reward -= 8
        
        # FIREBALL PROXIMITY WARNING
        min_fireball_dist = float('inf')
        for fireball in self.fireballs:
            fb_pos = fireball.get_grid_pos()
            dist = abs(self.player_row - fb_pos[0]) + abs(self.player_col - fb_pos[1])
            min_fireball_dist = min(min_fireball_dist, dist)
        
        if min_fireball_dist <= 1:
            reward -= 1.5
        elif min_fireball_dist == 2:
            reward -= 0.5
        
        # Death
        if self.health <= 0:
            terminated = True
            reward -= 25
            self.health = 0
        
        # FINISH LINE
        if player_loc == self.finishLine:
            if self.boss_alive == 0:
                reward += 75
                terminated = True
            else:
                reward -= 20
                self.player_row, self.player_col = row, col
        
        # REWARD SHAPING
        if not terminated:
            target = None
            if self.has_loot == 0:
                target = self.loot
            elif self.has_mini == 0:
                target = (self.mini_boss_row, self.mini_boss_col)
            elif self.boss_alive == 1:
                target = (self.boss_row, self.boss_col)
            else:
                target = self.finishLine
            
            if target:
                old_dist = abs(row - target[0]) + abs(col - target[1])
                new_dist = abs(new_row - target[0]) + abs(new_col - target[1])
                if new_dist < old_dist:
                    reward += 0.5
                elif new_dist > old_dist:
                    reward -= 0.3
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), reward, terminated, False, {}
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return None
        return self._render_gui(self.render_mode)
    
    def _render_gui(self, mode: str):
        """Render using pygame"""
        try:
            if self.window is None:
                pygame.init()
                pygame.display.set_caption("Deep RL Dungeon Crawler ")
                if mode == "human":
                    self.window = pygame.display.set_mode(WINDOW_SIZE)
                elif mode == "rgb_array":
                    self.window = pygame.Surface(WINDOW_SIZE)
            
            if self.cell_size is None or self.cell_size[0] < 10:
                self.cell_size = (
                    WINDOW_SIZE[0] / self.num_columns,
                    WINDOW_SIZE[1] / self.num_rows,
                )
            
            self._load_images()
            
            wall_thickness = 8
            vertical_wall = pygame.transform.scale(self.walls_img, (wall_thickness, int(self.cell_size[1])))
            horizontal_wall = pygame.transform.scale(self.walls_img, (int(self.cell_size[0]), wall_thickness))
            
            def get_pos(r, c):
                return (c * self.cell_size[0], r * self.cell_size[1])
            
            # Draw layers
            for r in range(self.num_rows):
                for c in range(self.num_columns):
                    pos = get_pos(r, c)
                    self.window.blit(self.background_img, pos)
            
            for r, c in self.restricted:
                pos = get_pos(r, c)
                self.window.blit(self.restricted_img, pos)
            
            desc = self.desc
            for r in range(self.num_rows):
                for c in range(self.num_columns):
                    pos = get_pos(r, c)
                    ascii_row = r + 1
                    ascii_col_idx = 2 * c + 1
                    
                    if c < self.max_col:
                        if desc[ascii_row][ascii_col_idx + 1] == b"|":
                            wall_pos = (pos[0] + self.cell_size[0] - (wall_thickness/2), pos[1])
                            self.window.blit(vertical_wall, wall_pos)
                    
                    if r < self.max_row:
                        if desc[ascii_row + 1][ascii_col_idx] == b"-":
                            wall_pos = (pos[0], pos[1] + self.cell_size[1] - (wall_thickness/2))
                            self.window.blit(horizontal_wall, wall_pos)
            
            self.window.blit(self.finishLine_img, get_pos(*self.finishLine))
            
            if self.has_loot == 0:
                self.window.blit(self.loot_image, get_pos(*self.loot))
            
            if self.has_mini == 0:
                self.window.blit(self.miniBoss_img, get_pos(self.mini_boss_row, self.mini_boss_col))
            
            if self.boss_alive == 1:
                self.window.blit(self.boss_img, get_pos(self.boss_row, self.boss_col))
            
            for fireball in self.fireballs:
                grid_pos = fireball.get_grid_pos()
                if 0 <= grid_pos[0] <= self.max_row and 0 <= grid_pos[1] <= self.max_col:
                    self.window.blit(self.fireball_img, get_pos(*grid_pos))
            
            self.window.blit(self.player_img, get_pos(self.player_row, self.player_col))
            
            self._draw_ui()
            
            if mode == "human":
                pygame.event.pump()
                pygame.display.update()
            elif mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
                )
        except Exception as e:
            print(f"Rendering error: {e}")
            return None
    
    def _load_images(self):
        """Load all game images"""
        if self.player_img is None:
            try:
                self.player_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/mario.png")),
                    self.cell_size
                )
            except:
                self.player_img = pygame.Surface(self.cell_size)
                self.player_img.fill((0, 0, 255))
        
        if self.loot_image is None:
            try:
                self.loot_image = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/loot.png")),
                    self.cell_size
                )
            except:
                self.loot_image = pygame.Surface(self.cell_size)
                self.loot_image.fill((255, 215, 0))
        
        if self.boss_img is None:
            try:
                self.boss_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/main-boss.png")),
                    self.cell_size
                )
            except:
                self.boss_img = pygame.Surface(self.cell_size)
                self.boss_img.fill((255, 0, 0))
        
        if self.miniBoss_img is None:
            try:
                self.miniBoss_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/mini-boss.png")),
                    self.cell_size
                )
            except:
                self.miniBoss_img = pygame.Surface(self.cell_size)
                self.miniBoss_img.fill((255, 165, 0))
        
        if self.finishLine_img is None:
            try:
                img = pygame.image.load(path.join(".", "img/princessPeach.png"))
                self.finishLine_img = pygame.transform.scale(img, self.cell_size)
            except:
                self.finishLine_img = pygame.Surface(self.cell_size)
                self.finishLine_img.fill((255, 192, 203))
        
        if self.background_img is None:
            try:
                self.background_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/background2.png")),
                    self.cell_size
                )
            except:
                self.background_img = pygame.Surface(self.cell_size)
                self.background_img.fill((200, 200, 200))
        
        if self.restricted_img is None:
            try:
                self.restricted_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/lava.png")),
                    self.cell_size
                )
            except:
                self.restricted_img = pygame.Surface(self.cell_size)
                self.restricted_img.fill((255, 69, 0))
        
        if self.walls_img is None:
            try:
                self.walls_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/mario_wall.png")),
                    self.cell_size
                )
            except:
                self.walls_img = pygame.Surface(self.cell_size)
                self.walls_img.fill((139, 69, 19))
        
        if self.fireball_img is None:
            try:
                self.fireball_img = pygame.transform.scale(
                    pygame.image.load(path.join(".", "img/fireball2.png")),
                    self.cell_size
                )
            except:
                self.fireball_img = pygame.Surface(self.cell_size, pygame.SRCALPHA)
                center = (int(self.cell_size[0] / 2), int(self.cell_size[1] / 2))
                radius = int(min(self.cell_size) / 4)
                pygame.draw.circle(self.fireball_img, (255, 100, 0), center, radius)
                pygame.draw.circle(self.fireball_img, (255, 200, 0), center, radius // 2)
    
    def _draw_ui(self):
        """Draw UI elements"""
        bar_width = 100
        bar_height = 20
        bar_x = 10
        bar_y = 10
        
        pygame.draw.rect(self.window, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        health_width = int((self.health / 3.0) * bar_width)
        if self.health == 3:
            color = (0, 255, 0)
        elif self.health == 2:
            color = (255, 255, 0)
        elif self.health == 1:
            color = (255, 128, 0)
        else:
            color = (255, 0, 0)
        
        if health_width > 0:
            pygame.draw.rect(self.window, color, (bar_x, bar_y, health_width, bar_height))
        
        pygame.draw.rect(self.window, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
        
        try:
            font = pygame.font.Font(None, 24)
            health_text = font.render(f"HP: {self.health}/3", True, (255, 255, 255))
            self.window.blit(health_text, (bar_x + bar_width + 10, bar_y))
            
            y_offset = bar_y + 30
            if self.has_loot:
                loot_text = font.render("✓ Loot", True, (0, 255, 0))
                self.window.blit(loot_text, (bar_x, y_offset))
            
            if self.has_mini:
                mini_text = font.render("✓ Power", True, (0, 255, 0))
                self.window.blit(mini_text, (bar_x + 80, y_offset))
            
            if not self.boss_alive:
                boss_text = font.render("✓ Boss", True, (0, 255, 0))
                self.window.blit(boss_text, (bar_x + 170, y_offset))
            
            if len(self.fireballs) > 0:
                fireball_text = font.render(f"Fireballs: {len(self.fireballs)}", True, (255, 100, 0))
                self.window.blit(fireball_text, (bar_x, y_offset + 25))
        except:
            pass
    
    def close(self):
        """Close the environment"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None