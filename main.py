import io
import os
import json
import random
import re
import time
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import inch

from crew_ai import generate_words, generate_clues


from fastapi.middleware.cors import CORSMiddleware

# create app ONCE
app = FastAPI(title="AI Puzzle Factory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React (CRA)
        "http://localhost:5173",  # Vite
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def is_valid_word(w: str) -> bool:
    return w.isalpha() and len(w) >= 3
# ============================
# CLUE + WORD CACHE
# ============================

# ===============================================================
# BOOK-LEVEL WORD DE-DUPLICATION
# ===============================================================
BOOK_USED_WORDS = set()


WORD_CACHE_FILE = "word_cache.json"
CLUE_CACHE_FILE = "clue_cache.json"

WORD_CACHE = {}
CLUE_CACHE = {}

def load_cache():
    global WORD_CACHE, CLUE_CACHE

    if os.path.exists(WORD_CACHE_FILE):
        with open(WORD_CACHE_FILE, "r") as f:
            WORD_CACHE = json.load(f)

    if os.path.exists(CLUE_CACHE_FILE):
        with open(CLUE_CACHE_FILE, "r") as f:
            CLUE_CACHE = json.load(f)

    with open(CLUE_CACHE_FILE, "w") as f:
        json.dump(CLUE_CACHE, f, indent=2)


# ===============================================================
# ENV + CONFIG
# ===============================================================
load_dotenv()

FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "")

PAGE_W = 6 * inch
PAGE_H = 9 * inch
MARGIN = 0.5 * inch
GRID_MAX = 5.5 * inch

LAST_REPLICATE_TIME = 0
REPLICATE_COOLDOWN = 15


# ===============================================================
# WORD CACHE (FAST MODE)
# ===============================================================
WORDS_CACHE: List[str] = []
CACHE_SIZE = 200


def load_word_cache():
    global WORDS_CACHE

    if WORDS_CACHE:
        return

    print("🔄 Loading word cache...")

    try:
        words = generate_words(CACHE_SIZE)
    except:
        words = []

    # FIX: real-word fallback (instead of random letters)
    if not words or len(words) < 50:
        words = [
            "APPLE", "HOUSE", "PLANT", "RIVER", "TABLE", "STONE",
            "MOUSE", "TRAIN", "LIGHT", "CLOUD", "BRIDGE", "FLOWER",
            "WINDOW", "GARDEN", "PUZZLE", "MARKET", "FRIEND",
            "MIRROR", "BOTTLE", "CAMERA"
        ]

    WORDS_CACHE = list(set(words))
    print(f"✅ Cached {len(WORDS_CACHE)} words")

    # ============================
# THEME WORD SUPPORT
# ============================

THEME_WORDS = {
    "animals": ["TIGER", "LION", "HORSE", "SNAKE", "EAGLE", "WHALE", "ZEBRA"],
    "nature": ["RIVER", "STONE", "CLOUD", "FOREST", "OCEAN", "PLANT"],
    "kids": ["APPLE", "BALL", "TRAIN", "HOUSE", "SMILE", "CANDY"],
    "general": []
}

def get_theme_words(theme):
    theme = theme.lower()
    if theme in THEME_WORDS and THEME_WORDS[theme]:
        return THEME_WORDS[theme]

    return WORDS_CACHE

def detect_theme(words):
    scores = {}

    for theme, keywords in THEME_WORDS.items():
        if not keywords:
            continue
        scores[theme] = sum(1 for w in words if w in keywords)

    if not scores:
        return "general"

    best_theme = max(scores, key=scores.get)
    return best_theme if scores[best_theme] > 0 else "general"

    
# ============================
# CLUE CACHE + SMART CLUES
# ============================

CLUE_CACHE = {}


def save_cache():
    with open("clue_cache.json", "w") as f:
        json.dump(CLUE_CACHE, f, indent=2)


def get_clues(words, difficulty="easy", theme="general"):
    load_cache()

    final_clues = {}
    words = [w for w in words if w.isalpha() and len(w) >= 3]

    missing = [w for w in words if w not in CLUE_CACHE]

    if missing:
        try:
            raw = generate_clues(missing)
            for w, clue in raw.items():
                CLUE_CACHE[w] = clue.strip()
            save_cache()
        except:
            pass

    # ✅ SAFE fallback
    for w in words:
        final_clues[w] = CLUE_CACHE.get(w, f"{w.capitalize()} (definition)")

    return final_clues



# ===============================================================
# JSON HELPER
# ===============================================================
def extract_json_any(text: str):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


# ===============================================================
# LLM HELPERS
# ===============================================================
def featherless_completion(prompt: str) -> str:
    if not FEATHERLESS_API_KEY:
        return ""

    try:
        r = requests.post(
            "https://api.featherless.ai/v1/completions",
            headers={
                "Authorization": f"Bearer {FEATHERLESS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "Sao10K/Fimbulvetr-11B-v2",
                "prompt": prompt,
                "max_tokens": 700,
                "temperature": 0.7,
            },
            timeout=60,
        )
        return r.json().get("completion", "")
    except:
        return ""


def replicate_completion(prompt: str) -> str:
    global LAST_REPLICATE_TIME
    if not REPLICATE_API_KEY:
        return ""

    if time.time() - LAST_REPLICATE_TIME < REPLICATE_COOLDOWN:
        return ""

    try:
        import replicate
        client = replicate.Client(api_token=REPLICATE_API_KEY)
        LAST_REPLICATE_TIME = time.time()
        out = client.run(
            "meta/meta-llama-3-70b-instruct",
            input={"prompt": prompt, "max_tokens": 700},
        )
        return "".join(out)
    except:
        return ""


def call_llm(prompt: str) -> str:
    return featherless_completion(prompt) or replicate_completion(prompt)


# ===============================================================
# SUDOKU ENGINE
# ===============================================================
def su_find(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def su_valid(board, num, pos):
    r, c = pos

    if num in board[r]:
        return False
    if num in [board[i][c] for i in range(9)]:
        return False

    br, bc = r // 3, c // 3
    for rr in range(br * 3, br * 3 + 3):
        for cc in range(bc * 3, bc * 3 + 3):
            if board[rr][cc] == num:
                return False
    return True


def su_solve(board):
    pos = su_find(board)
    if not pos:
        return True

    r, c = pos
    nums = list(range(1, 10))
    random.shuffle(nums)

    for n in nums:
        if su_valid(board, n, (r, c)):
            board[r][c] = n
            if su_solve(board):
                return True
            board[r][c] = 0
    return False


def generate_sudoku(diff):
    board = [[0] * 9 for _ in range(9)]
    su_solve(board)

    solution = [row[:] for row in board]

    holes = {"easy": 34, "medium": 44, "hard": 54}[diff]
    removed = set()

    # FIX: prevent removing same cell repeatedly
    while len(removed) < holes:
        r, c = random.randint(0, 8), random.randint(0, 8)
        if (r, c) not in removed:
            board[r][c] = 0
            removed.add((r, c))

    return {
        "type": "sudoku",
        "puzzle": board,
        "solution": solution
    }


# ===============================================================
# MAZE ENGINE
# ===============================================================
def maze_size(diff):
    return (15, 15) if diff == "easy" else (21, 21) if diff == "medium" else (31, 31)


def carve(w, h):
    grid = [[1] * w for _ in range(h)]

    def dfs(r, c):
        grid[r][c] = 0
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        random.shuffle(dirs)

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < h and 0 < nc < w and grid[nr][nc] == 1:
                grid[r + dr // 2][c + dc // 2] = 0
                dfs(nr, nc)

    dfs(1, 1)
    return grid


def solve_maze(grid):
    h, w = len(grid), len(grid[0])
    start, end = (1, 1), (h - 2, w - 2)

    stack = [(start, [start])]
    visited = {start}

    while stack:
        (r, c), path = stack.pop()
        if (r, c) == end:
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append(((nr, nc), path + [(nr, nc)]))
    return []


def generate_maze(diff):
    rows, cols = maze_size(diff)
    grid = carve(cols, rows)
    path = solve_maze(grid) or [(1, 1)]  # FIX: never empty

    return {
        "type": "maze",
        "grid": grid,
        "solution_path": path
    }
# ===============================================================
# WORDSEARCH ENGINE
# ===============================================================

WORD_DIRS = [(0, 1), (1, 0), (1, 1), (-1, 1)]


def generate_wordsearch(diff):
    load_word_cache()

    size = {
        "easy": 12,
        "medium": 16,
        "hard": 20
    }[diff]

    grid = [[""] * size for _ in range(size)]
    available = [w for w in WORDS_CACHE if w not in BOOK_USED_WORDS]
    words = random.sample(available, min(10, len(available)))
    BOOK_USED_WORDS.update(words)

    placements = []

    for word in words:
        placed = False
        for _ in range(300):
            dr, dc = random.choice(WORD_DIRS)
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)

            ok = True
            for i, ch in enumerate(word):
                rr, cc = r + dr * i, c + dc * i
                if not (0 <= rr < size and 0 <= cc < size):
                    ok = False
                    break
                if grid[rr][cc] not in ("", ch):
                    ok = False
                    break

            if ok:
                for i, ch in enumerate(word):
                    grid[r + dr * i][c + dc * i] = ch
                placements.append({
                    "word": word,
                    "start": (r, c),
                    "direction": (dr, dc)
                })
                placed = True
                break

    # Fill empty cells
    for r in range(size):
        for c in range(size):
            if grid[r][c] == "":
                grid[r][c] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    return {
        "type": "wordsearch",
        "grid": grid,
        "words": [p["word"] for p in placements],
        "placements": placements
    }


# ===============================================================
# CROSSWORD ENGINE — FIXED & STABLE
# ===============================================================

# ===============================================================
# MULTIPLE CROSSWORD TEMPLATES (ROTATED PER PUZZLE)
# ===============================================================

CROSSWORD_TEMPLATES = [
    [
        "##..#..#.......",
        "..#..#..#..#...",
        ".....#.....#..",
        ".....#.....#..",
        ".....#..#.....",
        ".....#.....#..",
        "..#...........#",
        "...............",
        "#..............",
        ".....#.....#..",
        ".....#..#.....",
        ".....#.....#..",
        "..#.....#.....",
        "...#..#..#....",
        ".......#..####"
    ],
    [
        ".....#.....#..",
        "..#..#..#..#...",
        ".....#........",
        ".....#.....#..",
        "#..#..#..#..#.",
        "...............",
        "..#.....#.....",
        ".....#..#.....",
        ".....#.....#..",
        "...............",
        ".#..#..#..#..#",
        "..#.....#.....",
        "........#.....",
        "...#..#..#..#.",
        "..#.....#....."
    ],
    [
        "..#.....#.....",
        ".....#.....#..",
        "...............",
        ".#..#..#..#..#",
        ".....#..#.....",
        "..#...........#",
        ".....#.....#..",
        "...............",
        "#...........#..",
        ".....#..#.....",
        ".#..#..#..#..#",
        "...............",
        "..#.....#.....",
        ".....#.....#..",
        ".....#.....#.."
    ],
    [
        "...............",
        ".#..#..#..#..#",
        ".....#.....#..",
        "..#.....#.....",
        "...............",
        ".....#..#.....",
        "#..............",
        "...............",
        "......#.......",
        ".....#..#.....",
        "...............",
        "..#.....#.....",
        ".....#.....#..",
        ".#..#..#..#..#",
        "..............."
    ],
    [
        ".....#.....#..",
        "...............",
        "..#.....#.....",
        ".#..#..#..#..#",
        ".....#..#.....",
        "...............",
        "#..............",
        "...............",
        "......#.......",
        "...............",
        ".....#..#.....",
        ".#..#..#..#..#",
        "..#.....#.....",
        "...............",
        "..#.....#....."
    ]
]


def generate_crossword(diff="easy", used_words=None):
    load_word_cache()

    # -----------------------
    # GRID SETUP
    # -----------------------
    template = random.choice(CROSSWORD_TEMPLATES)
    max_cols = max(len(row) for row in template)
    grid = [list(row.ljust(max_cols, "#")) for row in template]

    rows, cols = len(grid), len(grid[0])

    filled = [row[:] for row in grid]

    # -----------------------
    # FIND ALL VALID SLOTS
    # -----------------------
    slots = []

    for r in range(rows):
        for c in range(cols):
            if filled[r][c] != ".":
                continue

            # ACROSS
            if c == 0 or filled[r][c - 1] == "#":
                ln = 0
                while c + ln < cols and filled[r][c + ln] == ".":
                    ln += 1
                if ln >= 3:
                    slots.append(("across", r, c, ln))

            # DOWN
            if r == 0 or filled[r - 1][c] == "#":
                ln = 0
                while r + ln < rows and filled[r + ln][c] == ".":
                    ln += 1
                if ln >= 3:
                    slots.append(("down", r, c, ln))

    # -----------------------
    # WORD POOL BY LENGTH
    # -----------------------
    pool = {}
    for w in WORDS_CACHE:
        if is_valid_word(w):
            pool.setdefault(len(w), []).append(w)

    used = used_words if used_words is not None else set()


    # -----------------------
    # PLACE WORDS SLOT-BY-SLOT
    # -----------------------
    for direction, r, c, length in slots:
        candidates = pool.get(length, [])
        random.shuffle(candidates)

        placed = False
        for word in candidates:
            if word in used:
                continue

            ok = True
            for i in range(length):
                rr = r + (i if direction == "down" else 0)
                cc = c + (i if direction == "across" else 0)
                if filled[rr][cc] not in (".", word[i]):
                    ok = False
                    break

            if ok:
                for i in range(length):
                    rr = r + (i if direction == "down" else 0)
                    cc = c + (i if direction == "across" else 0)
                    filled[rr][cc] = word[i]
                used.add(word)
                placed = True
                break


    # -----------------------
    # CLEAN REMAINING DOTS
    # -----------------------
    for r in range(rows):
        for c in range(cols):
            if filled[r][c] == ".":
                filled[r][c] = "#"


    # -----------------------
    # CLEAN REMAINING DOTS
    # -----------------------
    for r in range(rows):
        for c in range(cols):
            if filled[r][c] == ".":
                filled[r][c] = "#"

    # -----------------------
    # NUMBER GRID (ONCE)
    # -----------------------
    numbers = number_grid(filled)

    # -----------------------
    # COLLECT WORDS WITH POSITIONS
    # -----------------------
    across = []
    down = []

    for r in range(rows):
        for c in range(cols):
            n = numbers[r][c]
            if n == 0:
                continue

            # ACROSS
            if c == 0 or filled[r][c - 1] == "#":
                w = ""
                i = c
                while i < cols and filled[r][i] != "#":
                    w += filled[r][i]
                    i += 1
                if is_valid_word(w):
                    across.append((n, w))

            # DOWN
            if r == 0 or filled[r - 1][c] == "#":
                w = ""
                i = r
                while i < rows and filled[i][c] != "#":
                    w += filled[i][c]
                    i += 1
                if is_valid_word(w):
                    down.append((n, w))

    # -----------------------
    # CLUES (WORD → CLUE SAFE)
    # -----------------------
    all_words = {w for _, w in across + down}
    clue_map = get_clues(list(all_words))

    clues_across = [(n, clue_map[w]) for n, w in across]
    clues_down = [(n, clue_map[w]) for n, w in down]

    return {
        "type": "crossword",
        "puzzle_grid": [["" if x != "#" else "#" for x in row] for row in filled],
        "solution_grid": filled,
        "numbers": numbers,
        "clues_across": clues_across,
        "clues_down": clues_down
    }


def number_grid(grid):
    rows, cols = len(grid), len(grid[0])
    numbers = [[0] * cols for _ in range(rows)]
    num = 1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                continue

            start_across = (
                (c == 0 or grid[r][c - 1] == "#") and
                (c + 1 < cols and grid[r][c + 1] != "#")
            )

            start_down = (
                (r == 0 or grid[r - 1][c] == "#") and
                (r + 1 < rows and grid[r + 1][c] != "#")
            )

            if start_across or start_down:
                numbers[r][c] = num
                num += 1

    return numbers

clues_across = []
clues_down = []



  
# ===============================================================
# PDF DRAWING HELPERS
# ===============================================================

def draw_metadata_page(c, title, puzzle_type, difficulty, count):
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 120, title)

    c.setFont("Helvetica", 14)
    y = PAGE_H - 220

    lines = [
        f"Puzzle Type: {puzzle_type.capitalize()}",
        f"Difficulty: {difficulty.capitalize()}",
        f"Total Puzzles: {count}",
        "",
        "Instructions:",
        "• Use pencil for best results",
        "• Answers are included at the back",
        "• Enjoy and challenge yourself!"
    ]

    for line in lines:
        c.drawCentredString(PAGE_W / 2, y, line)
        y -= 26

    c.showPage()

def draw_grid(c, top, left, rows, cols, getter):
    cell = min(GRID_MAX / cols, GRID_MAX / rows)
    for r in range(rows):
        for cc in range(cols):
            x = left + cc * cell
            y = top - (r + 1) * cell
            c.rect(x, y, cell, cell)
            val = getter(r, cc)
            if val:
                c.setFont("Helvetica", 9)
                c.drawCentredString(x + cell / 2, y + cell / 2 - 3, str(val))


# ===============================================================
# SUDOKU PDF
# ===============================================================

def draw_sudoku_pdf(c, p):
    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2
    draw_grid(c, top, left, 9, 9, lambda r, c: p["puzzle"][r][c] or "")


def draw_sudoku_solution_pdf(c, p):
    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2
    draw_grid(c, top, left, 9, 9, lambda r, c: p["solution"][r][c])


# ===============================================================
# MAZE PDF
# ===============================================================

def draw_maze_pdf(c, p):
    g = p["grid"]
    rows, cols = len(g), len(g[0])
    cell = min(GRID_MAX / cols, GRID_MAX / rows)

    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2

    for r in range(rows):
        for c0 in range(cols):
            x = left + c0 * cell
            y = top - (r + 1) * cell
            c.rect(x, y, cell, cell, fill=1 if g[r][c0] else 0)


def draw_maze_solution_pdf(c, p):
    draw_maze_pdf(c, p)

    path = p["solution_path"]
    rows, cols = len(p["grid"]), len(p["grid"][0])
    cell = min(GRID_MAX / cols, GRID_MAX / rows)

    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2

    c.setLineWidth(2)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        c.line(
            left + c1 * cell + cell / 2,
            top - (r1 + 1) * cell + cell / 2,
            left + c2 * cell + cell / 2,
            top - (r2 + 1) * cell + cell / 2
        )
    c.setLineWidth(1)


# ===============================================================
# WORDSEARCH PDF
# ===============================================================

def draw_wordsearch_pdf(c, p):
    g = p["grid"]
    rows, cols = len(g), len(g[0])

    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2

    draw_grid(c, top, left, rows, cols, lambda r, c: g[r][c])

    y = MARGIN + 1.2 * inch
    c.setFont("Helvetica", 10)

    for i in range(0, len(p["words"]), 3):
        c.drawString(MARGIN, y, ", ".join(p["words"][i:i + 3]))
        y -= 14


def draw_wordsearch_solution_pdf(c, p):
    g = p["grid"]
    rows, cols = len(g), len(g[0])

    top = PAGE_H - MARGIN - 0.5 * inch
    left = (PAGE_W - GRID_MAX) / 2
    cell = min(GRID_MAX / cols, GRID_MAX / rows)

    highlights = set()
    for it in p["placements"]:
        r, c0 = it["start"]
        dr, dc = it["direction"]
        for i in range(len(it["word"])):
            highlights.add((r + dr * i, c0 + dc * i))

    for r in range(rows):
        for cc in range(cols):
            x = left + cc * cell
            y = top - (r + 1) * cell
            c.rect(x, y, cell, cell)
            if (r, cc) in highlights:
                c.circle(x + cell / 2, y + cell / 2, cell * 0.4)
            c.drawCentredString(x + cell / 2, y + cell / 2 - 3, g[r][cc])


# ===============================================================
# CROSSWORD PDF
# ===============================================================

def draw_crossword_pdf(c, p):
    grid = p["puzzle_grid"]
    numbers = p["numbers"]


    rows, cols = len(grid), len(grid[0])
    cell = 24

    start_x = (PAGE_W - cols * cell) / 2
    start_y = PAGE_H - 80

    for r in range(rows):
        for c0 in range(cols):
            x = start_x + c0 * cell
            y = start_y - r * cell

            if grid[r][c0] == "#":
                c.setFillColorRGB(0, 0, 0)
                c.rect(x, y, cell, cell, fill=1)
            else:
                # White cell
                c.setFillColorRGB(1, 1, 1)
                c.rect(x, y, cell, cell, fill=1)

                # Number (top-left)
                if numbers[r][c0] > 0:
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica", 7)
                    c.drawString(x + 2, y + cell - 10, str(numbers[r][c0]))

    # ---- CLUES ----
    left_x = MARGIN
    right_x = PAGE_W / 2 + 20
    y = start_y - rows * cell - 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_x, y, "Across")
    c.drawString(right_x, y, "Down")
    y -= 18

    c.setFont("Helvetica", 10)
    max_lines = max(len(p["clues_across"]), len(p["clues_down"]))

    for i in range(max_lines):
        if i < len(p["clues_across"]):
            num, clue = p["clues_across"][i]
            c.drawString(left_x, y, f"{num}. {clue}")

        if i < len(p["clues_down"]):
            num, clue = p["clues_down"][i]
            c.drawString(right_x, y, f"{num}. {clue}")

        y -= 14


def draw_crossword_solution_pdf(c, p):
    grid = p["solution_grid"]
    numbers = p["numbers"]


    rows, cols = len(grid), len(grid[0])
    cell = 24

    start_x = (PAGE_W - cols * cell) / 2
    start_y = PAGE_H - 80

    for r in range(rows):
        for c0 in range(cols):
            x = start_x + c0 * cell
            y = start_y - r * cell

            if grid[r][c0] == "#":
                c.setFillColorRGB(0, 0, 0)
                c.rect(x, y, cell, cell, fill=1)
            else:
                # Cell
                c.setFillColorRGB(1, 1, 1)
                c.rect(x, y, cell, cell, fill=1)

                # Number
                if numbers[r][c0] > 0:
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica", 7)
                    c.drawString(x + 2, y + cell - 10, str(numbers[r][c0]))

                # Letter (blue like image)
                c.setFillColorRGB(0.1, 0.3, 0.9)
                c.setFont("Helvetica-Bold", 14)
                c.drawCentredString(
                    x + cell / 2,
                    y + cell / 2 - 5,
                    grid[r][c0]
                )

    c.setFillColorRGB(0, 0, 0)

# ===============================================================
# PDF BUILDER
# ===============================================================

def build_pdf(puzzles, title, difficulty):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))

    draw_metadata_page(
        c,
        title,
        puzzles[0]["type"],
        difficulty,
        len(puzzles)
    )



    # -------- PUZZLES --------
    for i, p in enumerate(puzzles, 1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, PAGE_H - 30, f"Puzzle {i}")

        if p["type"] == "sudoku":
            draw_sudoku_pdf(c, p)
        elif p["type"] == "maze":
            draw_maze_pdf(c, p)
        elif p["type"] == "wordsearch":
            draw_wordsearch_pdf(c, p)
        elif p["type"] == "crossword":
            draw_crossword_pdf(c, p)

        c.showPage()

    # -------- ANSWERS --------
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(PAGE_W / 2, PAGE_H / 2, "Answers")
    c.showPage()

    for i, p in enumerate(puzzles, 1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN, PAGE_H - 30, f"Answer {i}")

        if p["type"] == "sudoku":
            draw_sudoku_solution_pdf(c, p)
        elif p["type"] == "maze":
            draw_maze_solution_pdf(c, p)
        elif p["type"] == "wordsearch":
            draw_wordsearch_solution_pdf(c, p)
        elif p["type"] == "crossword":
            draw_crossword_solution_pdf(c, p)

        c.showPage()

    c.save()
    buf.seek(0)
    return buf


# ===============================================================
# FASTAPI APP
# ===============================================================

class GenerateRequest(BaseModel):
    puzzle_type: str
    difficulty: str = "easy"
    count: int = 1
    title: str = "Puzzle Book"


@app.post("/generate")
async def generate(req: GenerateRequest):
    BOOK_USED_WORDS.clear()

    puzzle_type = req.puzzle_type.lower().strip()
    difficulty = req.difficulty.lower().strip()

    if puzzle_type not in ["sudoku", "maze", "wordsearch", "crossword"]:
        return {
            "error": "Invalid puzzle_type",
            "allowed": ["sudoku", "maze", "wordsearch", "crossword"]
        }

    puzzles = []

    for _ in range(req.count):
        if puzzle_type == "sudoku":
            puzzles.append(generate_sudoku(difficulty))
        elif puzzle_type == "maze":
            puzzles.append(generate_maze(difficulty))
        elif puzzle_type == "wordsearch":
            puzzles.append(generate_wordsearch(difficulty))
        elif puzzle_type == "crossword":
            puzzles.append(generate_crossword(difficulty, BOOK_USED_WORDS))

    pdf = build_pdf(puzzles, req.title, difficulty)

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=puzzles.pdf"}
    )
