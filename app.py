
import streamlit as st

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
            st.session_state.game_message = "🎉 Congratulations! You reached the goal!"

    else:

        st.session_state.game_message = "🧱 You hit a wall! Try another direction."


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


# Layout
left, right = st.columns([1.2, 1])

with left:

    st.markdown("### 🗺️ Robot Maze")

    display_maze()

    st.info(st.session_state.game_message)

    st.metric("Number of Moves", st.session_state.moves)

with right:

    st.markdown("### 💻 Programming Area")

    st.markdown("**Available commands:**")

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
move_forward()
turn_right()
move_forward()"""
    )

    if st.button("▶ Run Program", type="primary"):

        st.session_state.game_message = "Program running..."

        run_student_code(student_code)

        st.rerun()

    if st.button("🔄 Reset Game"):

        st.session_state.robot_position = START
        st.session_state.robot_direction = 0
        st.session_state.game_message = "Game reset! Try again."
        st.session_state.moves = 0
        st.session_state.game_won = False

        st.rerun()

if st.session_state.game_won:

    st.balloons()

    st.success(
        "🏆 Level Completed! You successfully programmed the robot!"
    )
