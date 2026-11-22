
import streamlit as st
import time
import csv
import io

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
    
if "challenge_history" not in st.session_state:
    st.session_state.challenge_history = []

if "completion_recorded" not in st.session_state:
    st.session_state.completion_recorded = False


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

# =========================
# SAVE CHALLENGE RESULTS
# =========================

def save_challenge_result():

    if st.session_state.completion_recorded:
        return

    record = {
        "Challenge": "Level 1 - Sequence",
        "Topic": "Sequential Programming",
        "Success": True,
        "Completion Time (sec)": round(
            st.session_state.completion_time, 2
        ),
        "Attempts": st.session_state.attempts,
        "Moves": st.session_state.moves,
        "Errors": st.session_state.errors,
        "Completed At": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    st.session_state.challenge_history.append(record)

    st.session_state.completion_recorded = True
    
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
            save_challenge_result()
            
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
    errors_before = st.session_state.errors

    for line in lines:

        line = line.split("#")[0].strip()

        if not line:
            continue

        if line not in commands:

            st.session_state.errors += 1

            st.session_state.game_message = (
                f"Invalid command: {line}"
            )

            return

        commands[line]()

        executed += 1

        if st.session_state.game_won:
            break

    errors_during_run = (
        st.session_state.errors - errors_before
    )

    if st.session_state.game_won:

        st.session_state.game_message = (
            "Congratulations! You reached the goal!"
        )

    elif errors_during_run > 0:

        st.session_state.game_message = (
            f"Program executed {executed} commands, "
            f"but encountered {errors_during_run} error(s). "
            "Check your robot's path."
        )

    elif executed > 0:

        st.session_state.game_message = (
            f"Program executed {executed} commands successfully."
        )

    else:

        st.session_state.game_message = (
            "No commands were entered."
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
            " ⏱️ Time",
            f"{minutes:02d}:{seconds:02d}"
        )

    else:

        st.metric("⏱️ Time ", "00:00")

# =========================
# MAIN GAME LAYOUT
# =========================

st.markdown("## Level 1: Sequential Programming")

st.markdown(
    "Guide the robot from the starting point to the goal "
    "using sequential commands."
)

left, right = st.columns([1.15, 1])

# =========================
# LEFT SIDE: MAZE + STATS
# =========================

with left:

    st.markdown("### Robot Maze")

    # Compact horizontal statistics
    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        show_timer()

    with stat2:
        st.metric("Moves", st.session_state.moves)

    with stat3:
        st.metric("Attempts", st.session_state.attempts)

    with stat4:
        st.metric("Errors", st.session_state.errors)

    display_maze()

    st.info(st.session_state.game_message)

    if st.session_state.game_won:

        st.success("Challenge completed successfully!")

        st.markdown("### Performance")

        perf1, perf2, perf3 = st.columns(3)

        with perf1:
            st.metric(
                "Completion Time",
                f"{st.session_state.completion_time:.1f} sec"
            )

        with perf2:
            st.metric(
                "Attempts",
                st.session_state.attempts
            )

        with perf3:
            st.metric(
                "Errors",
                st.session_state.errors
            )

# =========================
# FULL-WIDTH LEARNING REFLECTION
# =========================

if st.session_state.game_won:

    st.markdown("---")

    st.markdown("# Concept Learned: Sequential Programming")

    st.markdown(
        "You completed the maze by giving the robot "
        "instructions in the correct order."
    )

    # Two-column learning section
    concept_col, example_col = st.columns(2)

    with concept_col:

        with st.container(border=True):

            st.markdown("### What is Sequential Programming?")

            st.write(
                "Sequential programming is a programming "
                "approach in which instructions are executed "
                "one after another in a specific order."
            )

            st.write(
                "The computer follows the instructions in "
                "the order they are written. Changing the "
                "order can change the program's output."
            )

    with example_col:

        with st.container(border=True):

            st.markdown("### Example")

            st.code(
                """move_forward()
turn_right()
move_forward()""",
                language="python"
            )

            st.write(
                "First, the robot moves forward. "
                "Next, it turns right. Finally, it moves "
                "forward again."
            )

    # Full-width real-world analogy
    st.markdown("### Real-World Connection")

    with st.container(border=True):

        st.write(
            "Consider preparing a cup of tea. You first boil "
            "water, then add tea powder, and then add milk. "
            "These steps follow a particular sequence."
        )

        st.write(
            "Similarly, in programming, the order of "
            "instructions determines how a task is performed."
        )

    # Full-width takeaway
    st.markdown("### Key Takeaway")

    st.info(
        "A sequence is an ordered set of instructions "
        "executed one after another. The order of "
        "instructions matters."
    )

    st.markdown("### Reflection")

    st.write(
        "Think about how changing the order of the "
        "robot's commands would change its path."
    )


# =========================
# RIGHT SIDE: CODE EDITOR
# =========================

with right:

    st.markdown("### Programming Editor")

    st.markdown("Available commands")

    st.code(
        """move_forward()
turn_right()
turn_left()""",
        language="python"
    )

    student_code = st.text_area(
        "Write your program",
        height=300,
        placeholder="""move_forward()
move_forward()
turn_right()
move_forward()""",
        key="student_code"
    )

    # Buttons arranged horizontally
    button1, button2, button3 = st.columns(3)

    with button1:

        if st.button(
            "Start Challenge",
            use_container_width=True
        ):

            st.session_state.robot_position = START
            st.session_state.robot_direction = 0

            st.session_state.moves = 0
            st.session_state.errors = 0
            st.session_state.attempts = 0

            st.session_state.game_won = False
            st.session_state.completion_recorded = False
            st.session_state.completion_time = None

            st.session_state.challenge_started_at = time.time()
            st.session_state.challenge_active = True

            st.session_state.game_message = (
                "Challenge started. Begin programming."
            )

            st.rerun()

    with button2:

        if st.button(
            "Run Program",
            type="primary",
            use_container_width=True
        ):

            if not st.session_state.challenge_active:

                st.warning(
                    "Start the challenge before running your program."
                )

            else:

                st.session_state.attempts += 1

                run_student_code(student_code)

                st.rerun()

    with button3:

        if st.button(
            "Reset",
            use_container_width=True
        ):

            st.session_state.robot_position = START
            st.session_state.robot_direction = 0

            st.session_state.moves = 0
            st.session_state.errors = 0
            st.session_state.attempts = 0

            st.session_state.game_won = False
            st.session_state.completion_time = None

            st.session_state.challenge_started_at = None
            st.session_state.challenge_active = False

            st.session_state.game_message = (
                "Game reset. Start a new challenge."
            )

            st.rerun()
# =========================
# LEARNING HISTORY
# =========================

st.markdown("---")

st.markdown("## Challenge History")

if st.session_state.challenge_history:

    st.dataframe(
        st.session_state.challenge_history,
        use_container_width=True,
        hide_index=True
    )

    # Create CSV download
    csv_buffer = io.StringIO()

    fieldnames = st.session_state.challenge_history[0].keys()

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        st.session_state.challenge_history
    )

    st.download_button(
        label="Download Learning History (CSV)",
        data=csv_buffer.getvalue(),
        file_name="codequest_learning_history.csv",
        mime="text/csv"
    )

else:

    st.write(
        "Your completed challenges will appear here."
    )
