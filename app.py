import streamlit as st
import time
import csv
import io
import random
import math
import pandas as pd

st.set_page_config(page_title="CodeQuest", page_icon="🤖", layout="wide")
st.title("CodeQuest")
st.subheader("Learn programming by guiding a robot through a maze")
st.write("Write a program to navigate the robot from START to GOAL.")

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
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

LEVELS = {
    "Level 1 — Sequence": {
        "topic": "Sequential Programming",
        "concept": "sequence",
        "description": "Execute commands in the correct order, one after another.",
        "missions": {
            "Mission 1 — First Checkpoint": {"goal": (4, 6), "difficulty": "Easy", "brief": "Reach the first checkpoint using ordered commands."},
            "Mission 2 — Intermediate Route": {"goal": (2, 6), "difficulty": "Medium", "brief": "Navigate farther through the maze using a sequence."},
            "Mission 3 — Final Destination": {"goal": (0, 6), "difficulty": "Hard", "brief": "Reach the final destination through a longer route."},
        },
        "commands": "move_forward()\nturn_right()\nturn_left()",
        "syntax_help": "Commands run from top to bottom. The order matters.",
    },
    "Level 2 — Loops": {
        "topic": "Loops / Repetition",
        "concept": "loops",
        "description": "Use loops to repeat instructions instead of writing the same command many times.",
        "missions": {
            "Mission 1 — Repeat a Move": {"goal": (4, 6), "difficulty": "Easy", "brief": "Use a loop to repeat a command. Try `for i in range(3):` with an indented command."},
            "Mission 2 — Repeated Path": {"goal": (2, 6), "difficulty": "Medium", "brief": "Reduce repeated instructions by using loops where appropriate."},
            "Mission 3 — Efficient Route": {"goal": (0, 6), "difficulty": "Hard", "brief": "Use loops to make your program shorter and easier to read."},
        },
        "commands": "move_forward()\nturn_right()\nturn_left()\n\nfor i in range(3):\n    move_forward()",
        "syntax_help": "Supported loop: for i in range(1–10): followed by indented commands. Nested loops are not supported yet.",
    },
    "Level 3 — Conditionals": {
        "topic": "Conditional Statements",
        "concept": "conditionals",
        "description": "Use conditions to make the robot choose what to do based on the space ahead.",
        "missions": {
            "Mission 1 — Detect a Wall": {"goal": (4, 6), "difficulty": "Easy", "brief": "Use `if wall_ahead():` to make a decision when a wall is in front."},
            "Mission 2 — Choose a Path": {"goal": (2, 6), "difficulty": "Medium", "brief": "Use `if path_clear():` and optionally `else:` to react to the maze."},
            "Mission 3 — Smart Navigation": {"goal": (0, 6), "difficulty": "Hard", "brief": "Combine conditions with commands to navigate more flexibly."},
        },
        "commands": "move_forward()\nturn_right()\nturn_left()\nwall_ahead()\npath_clear()\n\nif wall_ahead():\n    turn_right()\nelse:\n    move_forward()",
        "syntax_help": "Supported conditions: if wall_ahead(): / if path_clear():, with optional else:. Put commands inside each block using 4 spaces.",
    },
}

level_name = st.sidebar.selectbox("Learning Level", list(LEVELS.keys()))
level = LEVELS[level_name]
mission_name = st.sidebar.selectbox("Mission", list(level["missions"].keys()))
mission = level["missions"][mission_name]
GOAL = mission["goal"]
challenge_key = f"{level_name} | {mission_name}"

# Session state defaults
DEFAULTS = {
    "robot_position": START, "robot_direction": 0, "moves": 0, "errors": 0,
    "attempts": 0, "game_won": False, "completion_time": None,
    "challenge_started_at": None, "challenge_active": False,
    "completion_recorded": False, "program_running": False,
    "game_message": "Select Start Challenge when you are ready.",
    "challenge_history": [], "previous_challenge": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- Reinforcement Learning Tutor ----------------
RL_DEFAULTS = {
    "q_table": {}, "rl_log": [], "rl_epsilon": 0.15, "rl_alpha": 0.30, "rl_gamma": 0.85,
    "rl_last_state": None, "rl_last_action": None, "rl_last_recommendation": None,
    "rl_last_reward": None, "rl_last_q_before": None, "rl_last_q_after": None,
    "rl_pending_adaptation": False, "rl_simulation_results": [], "rl_training_episodes": 0,
    "rl_mode": "Q-learning adaptive", "rl_selected_page": "Game",
}
for key, value in RL_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

RL_ACTIONS = ["Easier challenge", "Same difficulty", "Harder challenge", "Revision", "Advance concept"]

def learner_state(history=None, current_level=None):
    history = history if history is not None else st.session_state.challenge_history
    current_level = current_level or level_name
    recent = history[-5:] if history else []
    successes = sum(1 for x in recent if x.get("Success"))
    success_rate = successes / len(recent) if recent else 0.0
    errors = sum(int(x.get("Errors", 0)) for x in recent)
    avg_time = sum(float(x.get("Completion Time (sec)", 0) or 0) for x in recent) / max(1, len(recent))
    mastery = "High" if len(recent) >= 3 and success_rate >= .8 and errors <= 2 else ("Medium" if recent and success_rate >= .5 else "Low")
    success_band = "Good" if success_rate >= .8 else ("Moderate" if success_rate >= .5 else "Poor")
    error_band = "High" if errors >= 3 else "Low"
    time_band = "Fast" if avg_time and avg_time < 45 else ("Slow" if avg_time >= 90 else "Normal")
    concept = LEVELS[current_level]["concept"]
    return (concept, mastery, success_band, error_band, time_band)

def q_values(state):
    return st.session_state.q_table.setdefault(str(state), {a: 0.0 for a in RL_ACTIONS})

def choose_rl_action(state, explore=True):
    q = q_values(state)
    if explore and random.random() < st.session_state.rl_epsilon:
        return random.choice(RL_ACTIONS), "exploration (epsilon-greedy)"
    best = max(q.values())
    candidates = [a for a, v in q.items() if v == best]
    return random.choice(candidates), "exploitation (highest Q-value)"

def action_to_challenge(action, current_level, current_mission):
    level_idx = list(LEVELS).index(current_level)
    missions = list(LEVELS[current_level]["missions"])
    mission_idx = missions.index(current_mission)
    if action == "Easier challenge":
        mission_idx = max(0, mission_idx - 1)
    elif action == "Harder challenge":
        mission_idx = min(len(missions) - 1, mission_idx + 1)
    elif action == "Revision":
        mission_idx = max(0, mission_idx)
    elif action == "Advance concept":
        level_idx = min(len(LEVELS) - 1, level_idx + 1)
        mission_idx = 0
    target_level = list(LEVELS)[level_idx]
    target_missions = list(LEVELS[target_level]["missions"])
    return target_level, target_missions[min(mission_idx, len(target_missions) - 1)]

def calculate_reward(success, errors, attempts, elapsed, action, previous_state, next_state):
    reward = 0.0
    if success:
        reward += 5.0
        if errors == 0: reward += 2.0
        if attempts <= 1: reward += 1.5
        reward += 1.0 if elapsed < 90 else 0.0
    else:
        reward -= 3.0
    reward -= min(errors, 5) * 0.5
    if action == "Advance concept" and not success: reward -= 2.0
    if action == "Easier challenge" and success and previous_state[1] == "High": reward -= 0.5
    if action == "Harder challenge" and success: reward += 0.5
    return reward

def update_q(state, action, reward, next_state):
    q = q_values(state)
    old = q[action]
    next_max = max(q_values(next_state).values())
    alpha, gamma = st.session_state.rl_alpha, st.session_state.rl_gamma
    new = old + alpha * (reward + gamma * next_max - old)
    q[action] = new
    return old, new

def record_rl_transition(success, errors, attempts, elapsed):
    if not st.session_state.rl_pending_adaptation or st.session_state.rl_last_state is None:
        return
    s = st.session_state.rl_last_state
    a = st.session_state.rl_last_action
    ns = learner_state()
    reward = calculate_reward(success, errors, attempts, elapsed, a, s, ns)
    old, new = update_q(s, a, reward, ns)
    st.session_state.rl_last_reward = reward
    st.session_state.rl_last_q_before = old
    st.session_state.rl_last_q_after = new
    st.session_state.rl_log.append({"State": str(s), "Action": a, "Reward": reward, "Next state": str(ns), "Q before": round(old, 4), "Q after": round(new, 4), "Success": success, "Errors": errors, "Attempts": attempts, "Time (sec)": round(elapsed or 0, 2)})
    st.session_state.rl_pending_adaptation = False

def adaptive_recommendation():
    state = learner_state()
    action, decision_type = choose_rl_action(state)
    next_level, next_mission = action_to_challenge(action, level_name, mission_name)
    st.session_state.rl_last_state = state
    st.session_state.rl_last_action = action
    st.session_state.rl_last_recommendation = (next_level, next_mission)
    st.session_state.rl_pending_adaptation = True
    return state, action, decision_type, next_level, next_mission

def run_simulated_training(episodes=300):
    profiles = {
        "Beginner": {"mastery": "Low", "success": "Poor", "errors": "High", "time": "Slow", "ideal": "Easier challenge"},
        "Fast learner": {"mastery": "High", "success": "Good", "errors": "Low", "time": "Fast", "ideal": "Harder challenge"},
        "Inconsistent": {"mastery": "Medium", "success": "Moderate", "errors": "High", "time": "Normal", "ideal": "Revision"},
        "Persistent": {"mastery": "Medium", "success": "Moderate", "errors": "Low", "time": "Normal", "ideal": "Same difficulty"},
    }
    rewards=[]
    for ep in range(episodes):
        profile=random.choice(list(profiles.values()))
        state=(random.choice(["sequence","loops","conditionals"]),profile["mastery"],profile["success"],profile["errors"],profile["time"])
        action,_=choose_rl_action(state, explore=True)
        reward=(4.0 if action==profile["ideal"] else -1.0) + random.uniform(-0.4,0.4)
        next_state=state
        update_q(state,action,reward,next_state)
        rewards.append(reward)
    st.session_state.rl_training_episodes += episodes
    st.session_state.rl_simulation_results = rewards
    return rewards

if st.session_state.previous_challenge != challenge_key:
    st.session_state.previous_challenge = challenge_key
    st.session_state.robot_position = START
    st.session_state.robot_direction = 0
    st.session_state.moves = 0
    st.session_state.errors = 0
    st.session_state.attempts = 0
    st.session_state.game_won = False
    st.session_state.completion_time = None
    st.session_state.challenge_started_at = None
    st.session_state.challenge_active = False
    st.session_state.completion_recorded = False
    st.session_state.program_running = False
    st.session_state.game_message = "Select Start Challenge when you are ready."


def cell_ahead():
    row, col = st.session_state.robot_position
    dr, dc = DIRECTIONS[st.session_state.robot_direction]
    nr, nc = row + dr, col + dc
    return 0 <= nr < 7 and 0 <= nc < 7 and MAZE[nr][nc] == 0


def wall_ahead():
    return not cell_ahead()


def move_forward():
    if not st.session_state.challenge_active or st.session_state.game_won:
        return
    if cell_ahead():
        row, col = st.session_state.robot_position
        dr, dc = DIRECTIONS[st.session_state.robot_direction]
        st.session_state.robot_position = (row + dr, col + dc)
        st.session_state.moves += 1
        if st.session_state.robot_position == GOAL:
            st.session_state.game_won = True
            st.session_state.completion_time = time.time() - st.session_state.challenge_started_at
            st.session_state.challenge_active = False
            st.session_state.game_message = "Challenge completed!"
            save_challenge_result()
    else:
        st.session_state.errors += 1
        st.session_state.game_message = "Blocked: the robot cannot move into a wall or outside the maze."


def turn_right():
    if st.session_state.challenge_active and not st.session_state.game_won:
        st.session_state.robot_direction = (st.session_state.robot_direction + 1) % 4


def turn_left():
    if st.session_state.challenge_active and not st.session_state.game_won:
        st.session_state.robot_direction = (st.session_state.robot_direction - 1) % 4


def save_challenge_result():
    if st.session_state.completion_recorded:
        return
    st.session_state.challenge_history.append({
        "Level": level_name,
        "Topic": level["topic"],
        "Mission": mission_name,
        "Success": True,
        "Completion Time (sec)": round(st.session_state.completion_time, 2),
        "Attempts": st.session_state.attempts,
        "Moves": st.session_state.moves,
        "Errors": st.session_state.errors,
        "Completed At": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    st.session_state.completion_recorded = True
    record_rl_transition(True, st.session_state.errors, st.session_state.attempts, st.session_state.completion_time)


def display_maze(container=None):
    rpos = st.session_state.robot_position
    direction_degrees = st.session_state.robot_direction * 90
    walking_class = "walking" if st.session_state.program_running else ""
    html = f"""
    <style>
    .cq-maze {{ display:grid; grid-template-columns:repeat(7,48px); gap:4px; width:fit-content; margin:12px auto; }}
    .cq-cell {{ width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:#e8edf3;color:#222;font-size:22px;box-sizing:border-box; }}
    .cq-wall {{ background:#454545; }} .cq-start {{ background:#a5d6a7; }} .cq-goal {{ background:#ffd54f; }} .cq-robot {{ background:#80cbc4; }}
    .cq-robot-icon {{ display:inline-block;font-size:29px;transform:rotate({direction_degrees}deg);transition:transform .2s ease; }}
    .cq-robot-icon.walking {{ animation:cq-walk .35s ease-in-out infinite; }}
    @keyframes cq-walk {{ 0%,100%{{translate:0 0;}} 50%{{translate:0 -4px;}} }}
    </style><div class="cq-maze">
    """
    for row in range(7):
        for col in range(7):
            pos = (row, col)
            if pos == rpos:
                html += f'<div class="cq-cell cq-robot"><span class="cq-robot-icon {walking_class}">🤖</span></div>'
            elif pos == GOAL:
                html += '<div class="cq-cell cq-goal">GOAL</div>'
            elif pos == START:
                html += '<div class="cq-cell cq-start">START</div>'
            elif MAZE[row][col] == 1:
                html += '<div class="cq-cell cq-wall">■</div>'
            else:
                html += '<div class="cq-cell"></div>'
    html += '</div>'
    if container is not None:
        container.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# Safe interpreter for the supported educational syntax. It does not use exec/eval.
MAX_COMMANDS_PER_RUN = 100


def parse_and_run_program(source, maze_display):
    lines = source.expandtabs(4).splitlines()
    command_count = 0
    error_count_before = st.session_state.errors

    def execute_command(command):
        nonlocal command_count
        command = command.strip()
        if command not in {"move_forward()", "turn_right()", "turn_left()"}:
            raise ValueError(f"Unsupported command: {command}")
        if command_count >= MAX_COMMANDS_PER_RUN:
            raise ValueError("Program exceeded 100 commands. Reduce repetitions.")
        command_count += 1
        if command == "move_forward()":
            move_forward()
        elif command == "turn_right()":
            turn_right()
        else:
            turn_left()
        display_maze(maze_display)
        time.sleep(0.25)

    def get_block(start_index, parent_indent):
        block = []
        i = start_index
        while i < len(lines):
            raw = lines[i]
            if not raw.strip() or raw.lstrip().startswith("#"):
                i += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent <= parent_indent:
                break
            block.append((i, indent, raw.strip()))
            i += 1
        return block, i

    def run_block(index, expected_indent=0):
        nonlocal command_count
        while index < len(lines):
            raw = lines[index]
            if not raw.strip() or raw.lstrip().startswith("#"):
                index += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent < expected_indent:
                return index
            if indent > expected_indent:
                raise ValueError(f"Unexpected indentation on line {index + 1}.")
            statement = raw.strip()

            # Repeat loop: for i in range(n):
            if statement.startswith("for "):
                if level["concept"] != "loops":
                    raise ValueError("Loops are available in Level 2.")
                import re
                match = re.fullmatch(r"for\s+i\s+in\s+range\((\d+)\):", statement)
                if not match:
                    raise ValueError("Use the supported loop format: for i in range(n):")
                count = int(match.group(1))
                if not 1 <= count <= 10:
                    raise ValueError("Loop count must be between 1 and 10.")
                body_start = index + 1
                body, next_index = get_block(body_start, indent)
                if not body:
                    raise ValueError("A loop must contain an indented command.")
                body_indent = body[0][1]
                for _ in range(count):
                    run_block(body_start, body_indent)
                    if st.session_state.game_won:
                        break
                index = next_index
                continue

            # Conditional blocks (Level 3)
            if statement in ("if wall_ahead():", "if path_clear():"):
                if level["concept"] != "conditionals":
                    raise ValueError("Conditionals are available in Level 3.")
                condition = wall_ahead() if statement == "if wall_ahead():" else cell_ahead()
                body_start = index + 1
                body, after_if = get_block(body_start, indent)
                if not body:
                    raise ValueError("An if statement must contain an indented command.")
                body_indent = body[0][1]
                if condition:
                    run_block(body_start, body_indent)
                index = after_if
                # Optional else at same indent
                if index < len(lines) and lines[index].strip() == "else:" and (len(lines[index]) - len(lines[index].lstrip(" "))) == indent:
                    else_start = index + 1
                    else_body, after_else = get_block(else_start, indent)
                    if not else_body:
                        raise ValueError("An else block must contain an indented command.")
                    if not condition:
                        run_block(else_start, else_body[0][1])
                    index = after_else
                continue

            if statement == "else:":
                raise ValueError("else must follow an if statement.")
            if statement.endswith(":"):
                raise ValueError("Unsupported block. Use the Level 2 for-loop or Level 3 if-condition syntax.")
            execute_command(statement)
            index += 1
            if st.session_state.game_won:
                return index
        return index

    try:
        run_block(0, 0)
        if st.session_state.game_won:
            st.session_state.game_message = "Challenge completed!"
        elif st.session_state.errors > error_count_before:
            st.session_state.game_message = "Program finished with blocked moves. Review your route."
        elif command_count:
            st.session_state.game_message = f"Executed {command_count} commands. Robot is at {st.session_state.robot_position}."
        else:
            st.session_state.game_message = "No commands were executed."
    except ValueError as exc:
        st.session_state.errors += 1
        st.session_state.game_message = str(exc)
    finally:
        st.session_state.program_running = False
        display_maze(maze_display)


# Module navigation
st.sidebar.divider()
st.sidebar.markdown("### CodeQuest Modules")
selected_page = st.sidebar.radio("Open module", ["Game", "Learner Profile", "Q-Learning Dashboard", "Training & Experiments"], key="rl_selected_page")

if selected_page == "Game":
    # Sidebar guide
    st.sidebar.title("CodeQuest Learning Path")
    st.sidebar.markdown("**Level 1 — Sequence**: ordered instructions")
    st.sidebar.markdown("**Level 2 — Loops**: repeat commands")
    st.sidebar.markdown("**Level 3 — Conditionals**: make decisions")
    st.sidebar.divider()
    st.sidebar.markdown("### Current mission")
    st.sidebar.write(mission_name)
    st.sidebar.caption(f"{level['topic']} · {mission['difficulty']}")
    st.sidebar.write(mission["brief"])
    st.sidebar.markdown("### Supported commands / syntax")
    st.sidebar.code(level["commands"], language="python")
    st.sidebar.caption(level["syntax_help"])
    st.sidebar.markdown("### Progress")
    if st.session_state.game_won:
        st.sidebar.success("Mission completed")
    else:
        st.sidebar.info("Mission in progress")
    st.sidebar.write(f"Moves: {st.session_state.moves} | Attempts: {st.session_state.attempts} | Errors: {st.session_state.errors}")

    @st.fragment(run_every="1s")
    def show_timer():
        if st.session_state.challenge_active:
            elapsed = time.time() - st.session_state.challenge_started_at
        elif st.session_state.completion_time is not None:
            elapsed = st.session_state.completion_time
        else:
            elapsed = 0
        st.metric("Time", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

    st.markdown(f"## {level_name}")
    st.caption(f"{mission_name} · Difficulty: {mission['difficulty']}")
    st.write(mission["brief"])
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Robot Maze")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_timer()
        with c2:
            st.metric("Moves", st.session_state.moves)
        with c3:
            st.metric("Attempts", st.session_state.attempts)
        with c4:
            st.metric("Errors", st.session_state.errors)
        maze_display = st.empty()
        display_maze(maze_display)
        st.info(st.session_state.game_message)
        if st.session_state.game_won:
            st.success("Mission completed successfully!")
            p1, p2, p3 = st.columns(3)
            p1.metric("Completion time", f"{st.session_state.completion_time:.1f} sec")
            p2.metric("Attempts", st.session_state.attempts)
            p3.metric("Errors", st.session_state.errors)

    with right:
        st.markdown("### Programming Editor")
        st.write("Available commands and syntax")
        st.code(level["commands"], language="python")
        student_code = st.text_area("Write your program", height=300, placeholder="Write your commands here", key="student_code")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Start Challenge", use_container_width=True):
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
                st.session_state.program_running = False
                st.session_state.game_message = "Challenge started. Write and run your program."
                st.rerun()
        with b2:
            if st.button("Run Program", type="primary", use_container_width=True):
                if not st.session_state.challenge_active:
                    st.warning("Start the challenge before running your program.")
                else:
                    st.session_state.attempts += 1
                    st.session_state.program_running = True
                    parse_and_run_program(student_code, maze_display)
                    st.rerun()
        with b3:
            if st.button("Reset", use_container_width=True):
                st.session_state.robot_position = START
                st.session_state.robot_direction = 0
                st.session_state.moves = 0
                st.session_state.errors = 0
                st.session_state.attempts = 0
                st.session_state.game_won = False
                st.session_state.completion_time = None
                st.session_state.challenge_started_at = None
                st.session_state.challenge_active = False
                st.session_state.completion_recorded = False
                st.session_state.program_running = False
                st.session_state.game_message = "Game reset. Start a new challenge."
                st.rerun()

    # Learning concept panel - only one copy, full width
    if st.session_state.game_won:
        st.markdown("---")
        st.markdown(f"# Concept Learned: {level['topic']}")
        if level["concept"] == "sequence":
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### What is Sequential Programming?")
                    st.write("Instructions execute one after another in the order written. Changing their order can change the result.")
            with col2:
                with st.container(border=True):
                    st.markdown("### Example")
                    st.code("move_forward()\nturn_right()\nmove_forward()", language="python")
                    st.write("The robot first moves forward, turns right, and then moves forward again.")
            st.markdown("### Real-world connection")
            st.write("Making tea involves steps in a sequence: boil water, add tea, then add milk. The order of steps matters.")
            st.info("Key takeaway: a sequence is an ordered set of instructions executed one after another.")
            st.write("Reflection: What would change if you rearranged the robot's commands?")
        elif level["concept"] == "loops":
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### What is a Loop?")
                    st.write("A loop repeats a set of instructions. It avoids writing the same command many times and makes programs shorter and easier to maintain.")
            with col2:
                with st.container(border=True):
                    st.markdown("### Example")
                    st.code("for i in range(3):\n    move_forward()", language="python")
                    st.write("The indented command runs three times. `range(3)` means three repetitions in this game.")
            st.markdown("### Real-world connection")
            st.write("If you need to take ten steps, you can say ‘repeat taking one step ten times’ instead of listing ten separate steps.")
            st.info("Key takeaway: loops express repetition compactly. The number of repetitions controls how many times the block runs.")
            st.write("Reflection: How many separate move_forward() lines would you need without a loop?")
        else:
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("### What is a Conditional?")
                    st.write("A conditional lets a program choose which instructions to execute based on whether a condition is true or false.")
            with col2:
                with st.container(border=True):
                    st.markdown("### Example")
                    st.code("if wall_ahead():\n    turn_right()\nelse:\n    move_forward()", language="python")
                    st.write("If there is a wall ahead, the robot turns. Otherwise, it moves forward.")
            st.markdown("### Real-world connection")
            st.write("A pedestrian might say: if the traffic light is red, wait; otherwise, cross when safe. The action depends on the condition.")
            st.info("Key takeaway: conditionals let a program respond differently to different situations.")
            st.write("Reflection: How would the robot behave if the condition were reversed?")

    st.markdown("---")
    st.markdown("## Challenge History")
    if st.session_state.challenge_history:
        st.dataframe(st.session_state.challenge_history, use_container_width=True, hide_index=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=st.session_state.challenge_history[0].keys())
        writer.writeheader()
        writer.writerows(st.session_state.challenge_history)
        st.download_button("Download Learning History (CSV)", data=csv_buffer.getvalue(), file_name="codequest_learning_history.csv", mime="text/csv")
    else:
        st.write("Completed missions will appear here.")


# ---------------- RL / learner analytics panels ----------------

if selected_page == "Learner Profile":
    st.markdown("# Learner Profile")
    hist = st.session_state.challenge_history
    state = learner_state()
    st.write("The learner profile is estimated from the most recent five completed challenges.")
    cols = st.columns(5)
    labels = ["Concept", "Mastery", "Success", "Errors", "Time"]
    for col, label, val in zip(cols, labels, state):
        col.metric(label, str(val))
    if hist:
        df = pd.DataFrame(hist)
        st.markdown("### Performance summary")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Completed challenges", len(hist))
        c2.metric("Success rate", f"{100*sum(x.get('Success',False) for x in hist)/len(hist):.0f}%")
        c3.metric("Average errors", f"{df['Errors'].mean():.2f}")
        c4.metric("Average time", f"{df['Completion Time (sec)'].mean():.1f} sec")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Complete challenges to build your learner profile.")

elif selected_page == "Q-Learning Dashboard":
    st.markdown("# Q-Learning Adaptive Tutor Dashboard")
    st.caption("The RL agent selects the next educational challenge—not the robot's movement.")
    st.markdown("### Current learner state")
    state=learner_state()
    st.code(str(state), language="text")
    q=q_values(state)
    qdf=pd.DataFrame([{"Action":a,"Q-value":v} for a,v in q.items()]).sort_values("Q-value",ascending=False)
    st.markdown("### Q-values for available actions")
    st.bar_chart(qdf.set_index("Action"))
    st.dataframe(qdf, hide_index=True, use_container_width=True)
    st.markdown("### Q-learning update")
    st.latex(r"Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]")
    st.write(f"Learning rate α = {st.session_state.rl_alpha:.2f} · Discount γ = {st.session_state.rl_gamma:.2f} · Exploration ε = {st.session_state.rl_epsilon:.2f}")
    with st.expander("Configure Q-learning"):
        st.session_state.rl_alpha=st.slider("Learning rate (α)",0.01,1.0,float(st.session_state.rl_alpha),0.01)
        st.session_state.rl_gamma=st.slider("Discount factor (γ)",0.0,0.99,float(st.session_state.rl_gamma),0.01)
        st.session_state.rl_epsilon=st.slider("Exploration rate (ε)",0.0,1.0,float(st.session_state.rl_epsilon),0.01)
    if st.session_state.rl_last_action:
        st.markdown("### Latest adaptive decision")
        st.write(f"**Chosen action:** {st.session_state.rl_last_action}")
        st.write(f"**Recommended challenge:** {st.session_state.rl_last_recommendation}")
        if st.session_state.rl_last_reward is not None:
            st.metric("Reward", st.session_state.rl_last_reward)
            st.write(f"Q-value: {st.session_state.rl_last_q_before:.4f} → {st.session_state.rl_last_q_after:.4f}")
    if st.session_state.rl_log:
        st.markdown("### Transition log")
        st.dataframe(pd.DataFrame(st.session_state.rl_log), use_container_width=True, hide_index=True)
        st.download_button("Download RL log CSV",pd.DataFrame(st.session_state.rl_log).to_csv(index=False),"codequest_rl_log.csv","text/csv")
    else:
        st.info("RL transition logs will appear after completing challenges that were selected through adaptive mode.")

elif selected_page == "Training & Experiments":
    st.markdown("# Training & Experiments")
    st.write("Train the Q-table using simulated learner profiles before conducting learner studies.")
    st.markdown("**Simulated profiles:** Beginner/struggling, fast learner, inconsistent, and persistent learner.")
    episodes=st.number_input("Training episodes",min_value=50,max_value=5000,value=300,step=50)
    if st.button("Train simulated learners",type="primary"):
        rewards=run_simulated_training(int(episodes))
        st.success(f"Completed {int(episodes)} simulation episodes. Total simulated training: {st.session_state.rl_training_episodes} episodes.")
    if st.session_state.rl_simulation_results:
        results=st.session_state.rl_simulation_results
        window=min(25,len(results))
        moving=[sum(results[max(0,i-window+1):i+1])/len(results[max(0,i-window+1):i+1]) for i in range(len(results))]
        st.markdown("### Reward curve")
        st.line_chart(pd.DataFrame({"Episode reward":results,"Moving average":moving}))
        st.metric("Mean reward (last 25 episodes)",f"{sum(results[-25:])/len(results[-25:]):.2f}")
    st.markdown("### Baseline comparison framework")
    st.table(pd.DataFrame([
        {"Method":"Fixed progression","Decision":"Predefined order","Learner adaptation":"None"},
        {"Method":"Rule-based","Decision":"Performance thresholds","Learner adaptation":"Manual rules"},
        {"Method":"Q-learning","Decision":"Learned action values","Learner adaptation":"Reward-driven"},
    ]))
    st.info("This simulation is a preliminary software test. It is not evidence of educational effectiveness; later evaluation should use consented learner data and defined study protocols.")

# Adaptive challenge controls shown on Game page
if selected_page == "Game":
    st.markdown("---")
    st.markdown("## Adaptive Tutor")
    st.caption("Q-learning adapts the next challenge based on the learner profile. It does not change the robot's commands or control the robot.")
    if st.button("Get RL recommendation for my next challenge"):
        state,action,decision,next_level,next_mission=adaptive_recommendation()
        st.success(f"RL action: **{action}** · Recommended: **{next_level} / {next_mission}**")
        st.caption(f"State: {state} · Decision: {decision}")
        st.session_state.rl_recommendation_display=(next_level,next_mission)
    if st.session_state.get("rl_recommendation_display"):
        rec_level,rec_mission=st.session_state.rl_recommendation_display
        st.write(f"Recommended challenge: **{rec_level} — {rec_mission}**")
        if st.button("Apply recommended challenge"):
            st.session_state.previous_challenge=None
            st.session_state.rl_recommendation_display=None
            st.session_state.rl_pending_adaptation=True
            st.session_state.rl_last_state=learner_state()
            st.session_state.rl_last_action=next((a for a in RL_ACTIONS if a==st.session_state.rl_last_action),"Same difficulty")
            st.info("Use the Learning Level and Mission selectors in the sidebar to select the recommended challenge, then start it.")

