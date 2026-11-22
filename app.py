
import streamlit as st
import time

st.set_page_config(
    page_title="CodeQuest",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CodeQuest")
st.subheader("Learn programming by guiding a robot through a maze!")

st.write(
    "Write commands to navigate the robot from START to GOAL."
)

# Maze: 0 = empty, 1 = wall
MAZE = [
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

START = (6, 0)
GOAL = (0, 6)

DIRECTIONS = [
    (-1, 0),  # Up
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1)   # Left
]

ARROWS = ["⬆️", "➡️", "⬇️", "⬅️"]

# Initialize game
# Challenge timer and performance tracking

if "challenge_started_at" not in st.session_state:
    st.session_state.challenge_started_at = None

if "completion_time" not in st.session_state:
    st.session_state.completion_time = None

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

if "errors" not in st.session_state:
    st.session_state.errors = 0

if "challenge_active" not in st.session_state:
    st.session_state.challenge_active = False
    
if "robot_position" not in st.session_state:
    st.session_state.robot_position = START

if "robot_direction" not in st.session_state:
    st.session_state.robot_direction = 0

if "game_message" not in st.session_state:
    st.session_state.game_message = "Your robot is ready!"

if "moves" not in st.session_state:
    st.session_state.moves = 0

if "game_won" not in st.session_state:
    st.session_state.game_won = False


# Display maze
def display_maze():

    robot_row, robot_col = st.session_state.robot_position

    html = """
    <style>
    .maze {
        display: grid;
        grid-template-columns: repeat(7, 48px);
        gap: 4px;
        width: fit-content;
        margin: 15px auto;
    }

    .cell {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        border-radius: 8px;
        background-color: #e8edf3;
    }

    .wall {
        background-color: #444444;
    }

    .robot {
        background-color: #80cbc4;
    }

    .goal {
        background-color: #ffd54f;
    }

    .start {
        background-color: #a5d6a7;
    }
    </style>

    <div class="maze">
    """

    for row in range(7):

        for col in range(7):

            if (row, col) == st.session_state.robot_position:
                cell = ARROWS[st.session_state.robot_direction]
                html += f'<div class="cell robot">{cell}</div>'

            elif (row, col) == GOAL:
                html += '<div class="cell goal">🏁</div>'

            elif (row, col) == START:
                html += '<div class="cell start">🚩</div>'

            elif MAZE[row][col] == 1:
                html += '<div class="cell wall">🧱</div>'

            else:
                html += '<div class="cell">⬜</div>'

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


# Robot movement
def move_forward():

    row, col = st.session_state.robot_position

    dr, dc = DIRECTIONS[st.session_state.robot_direction]

    new_row = row + dr
    new_col = col + dc

    if (
        0 <= new_row < 7
        and 0 <= new_col < 7
        and MAZE[new_row][new_col] == 0
    ):

        st.session_state.robot_position = (new_row, new_col)
        st.session_state.moves += 1

        if st.session_state.robot_position == GOAL:
        
            st.session_state.game_won = True
        
            st.session_state.completion_time = (
                time.time() - st.session_state.challenge_started_at
            )
        
            st.session_state.challenge_active = False
        
            st.session_state.game_message = (
                "🎉 Congratulations! You reached the goal!"
            )
        else:
        
            st.session_state.errors += 1
        
            st.session_state.game_message = (
                "🧱 You hit a wall! Try another direction."
            )


def turn_right():

    st.session_state.robot_direction = (
        st.session_state.robot_direction + 1
    ) % 4


def turn_left():

    st.session_state.robot_direction = (
        st.session_state.robot_direction - 1
    ) % 4


# Safe command interpreter
def run_student_code(code):

    commands = {
        "move_forward()": move_forward,
        "turn_right()": turn_right,
        "turn_left()": turn_left,
    }

    lines = code.split("\n")

    executed = 0

    for line in lines:

        line = line.split("#")[0].strip()

        if not line:
            continue

        if line not in commands:
        
            st.session_state.errors += 1
        
            st.session_state.game_message = (
                f"❌ Invalid command: {line}"
            )
        
            return

        commands[line]()

        executed += 1

        if st.session_state.game_won:
            break

    if executed > 0 and not st.session_state.game_won:

        st.session_state.game_message = (
            f"✅ Executed {executed} commands!"
        )


# =========================
# CHALLENGE SYSTEM
# =========================

st.sidebar.title("🎮 Game Levels")

st.sidebar.markdown("### Level 1")
st.sidebar.success("🟢 Sequence Programming")

st.sidebar.markdown("""
### 🎯 Your Mission

Guide the robot 🤖 from the starting
position 🚩 to the goal 🏁.

### 📚 Learning Objective

Learn how the order of instructions
(sequence) affects program execution.

### 🧠 Rules

- Commands execute from top to bottom.
- The robot moves in the direction it faces.
- Walls cannot be crossed.
- Reach the goal to complete the level.

### 🛠️ Commands

`move_forward()`

`turn_right()`

`turn_left()`
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🏆 Progress")

if st.session_state.game_won:
    st.sidebar.success("Level 1 Completed! 🎉")
else:
    st.sidebar.info("Level 1: In Progress")

st.sidebar.markdown(
    f"**Moves:** {st.session_state.moves}"
)

# =========================
# LIVE CHALLENGE TIMER
# =========================

@st.fragment(run_every="1s")
def show_timer():

    if st.session_state.challenge_active:

        elapsed = (
            time.time() -
            st.session_state.challenge_started_at
        )

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        st.metric(
            "⏱️ Time Elapsed",
            f"{minutes:02d}:{seconds:02d}"
        )

    elif st.session_state.completion_time is not None:

        elapsed = st.session_state.completion_time

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        st.metric(
            "🏁 Completion Time",
            f"{minutes:02d}:{seconds:02d}"
        )

    else:

        st.metric("⏱️ Time Elapsed", "00:00")
# =========================
# MAIN GAME LAYOUT
# =========================

left, right = st.columns([1.2, 1])

with left:

    st.markdown("## 🗺️ Robot Maze")
    show_timer()

    display_maze()

    st.info(st.session_state.game_message)

    st.metric("Number of Moves", st.session_state.moves)
    st.markdown("### 📊 Challenge Statistics")

    st.metric("🔄 Attempts", st.session_state.attempts)
    
    st.metric("❌ Errors", st.session_state.errors)
    
    st.metric("🚶 Total Moves", st.session_state.moves)


with right:

    st.markdown("## 💻 Programming Area")

    st.markdown(
        "Write your program using the available commands."
    )

    st.code(
        """move_forward()
turn_right()
turn_left()""",
        language="python"
    )

    student_code = st.text_area(
        "Write your program here:",
        height=250,
        placeholder="""move_forward()
turn_right()
move_forward()"""
    )

    # Start challenge
if st.button("🚀 Start Challenge", type="primary"):

    st.session_state.robot_position = START
    st.session_state.robot_direction = 0

    st.session_state.moves = 0
    st.session_state.errors = 0
    st.session_state.attempts = 0

    st.session_state.game_won = False
    st.session_state.completion_time = None

    st.session_state.challenge_started_at = time.time()
    st.session_state.challenge_active = True

    st.session_state.game_message = "Challenge started! Good luck!"

    st.rerun()


    col1, col2 = st.columns(2)
    
    with col1:
    
        if st.button("▶ Run Program", type="primary"):
    
            if not st.session_state.challenge_active:
    
                st.warning("Please click Start Challenge first!")
    
            else:
    
                st.session_state.attempts += 1
    
                run_student_code(student_code)
    
                st.rerun()
    
    
    with col2:
    
        if st.button("🔄 Reset Game"):
    
            st.session_state.robot_position = START
            st.session_state.robot_direction = 0
    
            st.session_state.game_message = "Game reset! Try again."
    
            st.session_state.moves = 0
            st.session_state.errors = 0
            st.session_state.attempts = 0
    
            st.session_state.game_won = False
            st.session_state.completion_time = None
    
            st.session_state.challenge_started_at = None
            st.session_state.challenge_active = False
    
            st.rerun()


# =========================
# LEVEL COMPLETION
# =========================

if st.session_state.game_won:

    st.balloons()

    st.success(
        "🏆 Level 1 Completed! "
        "You successfully programmed the robot!"
    )

    st.markdown("### 🎉 Achievement Unlocked")

    st.markdown("""
    🥇 **Sequence Master**

    You successfully used sequential programming
    instructions to navigate the maze.
    """)
    st.markdown("### 📈 Your Performance")

    st.write(f"⏱️ Time: {st.session_state.completion_time:.2f} seconds")

    st.write(f"🔄 Attempts: {st.session_state.attempts}")

    st.write(f"❌ Errors: {st.session_state.errors}")

    st.write(f"🚶 Moves: {st.session_state.moves}")
