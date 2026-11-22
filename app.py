import streamlit as st
import time
import csv
import io
import random
import math
import pandas as pd
import numpy as np

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

# ---------------- DQN Adaptive Learning Tutor ----------------
DQN_ACTIONS = [
    "Visual explanation", "Worked example", "Guided practice", "Debugging exercise",
    "Standard challenge", "Advanced challenge", "Revision", "Advance concept"
]
DQN_FEATURES = 12  # concept one-hot(3) + mastery/accuracy/speed/errors/hints/attempts + skill scores(3)
DQN_DEFAULTS = {
    "dqn_memory": [], "dqn_log": [], "dqn_epsilon": 0.25, "dqn_gamma": 0.90,
    "dqn_learning_rate": 0.01, "dqn_training_steps": 0, "dqn_episode_rewards": [],
    "dqn_last_state": None, "dqn_last_action": None, "dqn_last_action_idx": None,
    "dqn_last_q_values": None, "dqn_last_reward": None, "dqn_last_target": None,
    "dqn_loss_history": [], "dqn_synthetic_history": [], "dqn_action_counts": {}, "dqn_epsilon_history": [],
    "dqn_pending": False, "dqn_recommendation": None, "dqn_experience_count": 0,
    "diagnostic_scores": {}, "posttest_scores": {}, "skill_scores": {},
    "active_teaching_action": None, "teaching_completed": False,
    "quiz_attempts": [], "dqn_simulated_profiles": [], "dqn_online": None, "dqn_target": None,
}
for key, value in DQN_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Simple NumPy DQN: 12-input -> 64-ReLU -> 64-ReLU -> 8-action Q-values.
def init_network(seed=None):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, np.sqrt(2/DQN_FEATURES), (DQN_FEATURES,64)), "b1": np.zeros(64),
            "W2": rng.normal(0, np.sqrt(2/64), (64,64)), "b2": np.zeros(64),
            "W3": rng.normal(0, np.sqrt(2/64), (64,len(DQN_ACTIONS))), "b3": np.zeros(len(DQN_ACTIONS))}

def copy_network(net):
    return {k:v.copy() for k,v in net.items()}

if st.session_state.dqn_online is None:
    st.session_state.dqn_online = init_network(42)
    st.session_state.dqn_target = copy_network(st.session_state.dqn_online)

def dqn_forward(net, x, cache=False):
    x=np.asarray(x,dtype=float).reshape(-1,DQN_FEATURES)
    z1=x@net['W1']+net['b1']; h1=np.maximum(0,z1)
    z2=h1@net['W2']+net['b2']; h2=np.maximum(0,z2)
    q=h2@net['W3']+net['b3']
    return (q,(x,z1,h1,z2,h2)) if cache else q

def learner_vector(concept=None):
    concept=concept or level['concept']
    hist=st.session_state.challenge_history[-5:]
    succ=sum(bool(x.get('Success')) for x in hist)/max(1,len(hist))
    avg_time=np.mean([float(x.get('Completion Time (sec)',0) or 0) for x in hist]) if hist else 0
    avg_errors=np.mean([float(x.get('Errors',0) or 0) for x in hist]) if hist else 0
    avg_attempts=np.mean([float(x.get('Attempts',1) or 1) for x in hist]) if hist else 1
    hints=float(st.session_state.get('hint_count',0))
    skills=st.session_state.skill_scores.get(concept,{})
    mastery=float(st.session_state.diagnostic_scores.get(concept,0.5))
    acc=float(skills.get('accuracy',succ))
    speed=1.0-min(avg_time/180.0,1.0)
    err=min(avg_errors/5.0,1.0); hint=min(hints/5.0,1.0); attempt=min(avg_attempts/5.0,1.0)
    cidx={"sequence":0,"loops":1,"conditionals":2}.get(concept,0)
    onehot=[float(cidx==i) for i in range(3)]
    skill_values=[float(skills.get(k,0.5)) for k in ['ordering','direction','debugging']]
    return np.array(onehot+[mastery,acc,speed,err,hint,attempt]+skill_values,dtype=float)

def dqn_choose_action(state, explore=True):
    q=dqn_forward(st.session_state.dqn_online,state)[0]
    if explore and random.random()<float(st.session_state.dqn_epsilon):
        idx=random.randrange(len(DQN_ACTIONS)); reason='Exploration (epsilon-greedy)'
    else:
        idx=int(np.argmax(q)); reason='Exploitation (highest predicted Q-value)'
    return idx,reason,q

def dqn_train_batch(batch_size=32):
    memory=st.session_state.dqn_memory
    if len(memory)<max(8,min(batch_size,16)): return None
    batch=random.sample(memory,min(batch_size,len(memory)))
    net=st.session_state.dqn_online; target=st.session_state.dqn_target
    lr=float(st.session_state.dqn_learning_rate)
    losses=[]
    for state,action,reward,next_state,done in batch:
        q,cache=dqn_forward(net,state,cache=True); q=q[0]
        tq=dqn_forward(target,next_state)[0]
        y=float(reward) if done else float(reward)+float(st.session_state.dqn_gamma)*float(np.max(tq))
        err=float(q[action]-y); losses.append(err*err)
        # Gradient for selected action only; clip gradients for stability.
        grad=np.zeros_like(q); grad[action]=np.clip(2*err,-5,5)
        x,z1,h1,z2,h2=cache
        gW3=h2.T@grad.reshape(1,-1); gb3=grad
        gh2=grad@net['W3'].T; gz2=gh2*(z2>0)
        gW2=h1.T@gz2; gb2=gz2.sum(axis=0)
        gh1=gz2@net['W2'].T; gz1=gh1*(z1>0)
        gW1=x.T@gz1; gb1=gz1.sum(axis=0)
        for k,g in [('W1',gW1),('b1',gb1),('W2',gW2),('b2',gb2),('W3',gW3),('b3',gb3)]:
            net[k]-=lr*np.clip(g,-5,5)
    st.session_state.dqn_training_steps+=1
    if st.session_state.dqn_training_steps%10==0:
        st.session_state.dqn_target=copy_network(net)
    return float(np.mean(losses))

def add_dqn_experience(state, action_idx, reward, next_state, done=False, label='Learner interaction'):
    st.session_state.dqn_memory.append((np.asarray(state,dtype=float),int(action_idx),float(reward),np.asarray(next_state,dtype=float),bool(done)))
    if len(st.session_state.dqn_memory)>5000: st.session_state.dqn_memory.pop(0)
    loss=dqn_train_batch()
    if loss is not None:
        st.session_state.dqn_loss_history.append({"Update": st.session_state.dqn_training_steps, "Loss": float(loss), "Source": label})
    st.session_state.dqn_log.append({'Experience':len(st.session_state.dqn_memory),'Action':DQN_ACTIONS[action_idx],
        'Reward':round(float(reward),3),'Loss':None if loss is None else round(loss,5),'Source':label,
        'Epsilon':round(float(st.session_state.dqn_epsilon),3)})
    return loss

LESSON_CONTENT={
"Visual explanation":("Visual / step-by-step explanation","Think of a program as a route made of small instructions. The computer follows one instruction at a time, in order.","move_forward()\nturn_right()\nmove_forward()"),
"Worked example":("Worked example","Watch the example, trace each command, and predict the robot's position after every instruction.","move_forward()\nmove_forward()\nturn_right()\nmove_forward()"),
"Guided practice":("Guided practice","Complete the route with a scaffold. First plan the commands, then run them and compare your prediction.","# Plan each step before running\nmove_forward()\nturn_right()"),
"Debugging exercise":("Debugging practice","Inspect the command sequence. Find which command or ordering causes the robot to take the wrong route, then correct it.","move_forward()\nturn_left()\nmove_forward()\n# Identify and fix the mistake"),
"Standard challenge":("Practice challenge","Try a new maze task independently. Focus on correctness before optimizing.","# Write your own solution"),
"Advanced challenge":("Challenge extension","Plan a longer route and try to reduce unnecessary commands while keeping the solution correct.","# Solve efficiently and explain your strategy"),
"Revision":("Targeted revision","Review the concept and your recent mistakes. Explain why the corrected program works before trying another task.","# Review: instruction order matters"),
"Advance concept":("Ready to advance","Your evidence suggests you may be ready for the next programming concept. Complete the mastery check first.","# Next concept unlocked after mastery check")}

def concept_skill_labels(concept):
    if concept=='sequence': return ['ordering','direction','debugging']
    if concept=='loops': return ['repetition','loop tracing','debugging']
    return ['condition reasoning','branch tracing','debugging']

QUIZZES_A={
'sequence':[
('What happens when commands are executed?', ['Randomly','Top to bottom in order','All at once','Only the last command'],1,'ordering'),
('Which command changes the robot direction?', ['move_forward()','turn_right()','start()','repeat()'],1,'direction'),
('If the robot needs to turn before moving, which order is correct?', ['move then turn','turn then move','either always works','no commands'],1,'direction')],
'loops':[
('What does for i in range(3) do?', ['Runs once','Runs twice','Repeats 3 times','Repeats forever'],2,'repetition'),
('Why use a loop?', ['To repeat commands','To stop the program','To remove variables','To turn only'],0,'loop tracing'),
('How many times does the indented block run in range(4)?', ['2','3','4','5'],2,'repetition')],
'conditionals':[
('A conditional is used to…', ['repeat always','make a decision based on a condition','store a number','end a program'],1,'condition reasoning'),
('What does else represent?', ['Another loop','The alternative when if condition is false','A comment','A command'],1,'branch tracing'),
('If wall_ahead() is true, which branch executes?', ['if branch','else branch','both','neither'],0,'condition reasoning')]
}
QUIZZES_B={
'sequence':[
('A program contains turn_right() then move_forward(). What is first?', ['Moving','Turning right','Stopping','Repeating'],1,'ordering'),
('Which pair changes direction and then advances?', ['move_forward(), move_forward()','turn_left(), move_forward()','start(), move_forward()','repeat(), turn_left()'],1,'direction'),
('What happens if move_forward() and turn_right() are swapped?', ['Turns before moving','Moves before turning','No commands execute','Repeats twice'],1,'ordering')],
'loops':[
('How many times does move_forward() execute in range(5)?', ['4','5','6','Once'],1,'repetition'),
('Which is the main benefit of a loop?', ['It makes a decision','It avoids repeated code','It changes direction automatically','It ends a program'],1,'loop tracing'),
('A block inside range(2) is executed…', ['0 times','1 time','2 times','3 times'],2,'repetition')],
'conditionals':[
('When is the else block executed?', ['When if is true','When if condition is false','Always','Only in loops'],1,'branch tracing'),
('What does if path_clear() test?', ['Whether a path ahead is clear','How many commands exist','Whether a loop repeats','Robot speed'],0,'condition reasoning'),
('If wall_ahead() is false, which block runs?', ['if','else','both','neither'],1,'branch tracing')]
}

def record_game_transition(success, errors, attempts, elapsed):
    if not st.session_state.dqn_pending or st.session_state.dqn_last_state is None: return
    state=st.session_state.dqn_last_state; action=st.session_state.dqn_last_action_idx
    next_state=learner_vector()
    reward=(3 if success else -2)+ (1.5 if success and errors==0 else 0) - min(errors,5)*0.35 - max(attempts-1,0)*0.25
    add_dqn_experience(state,action,reward,next_state,done=bool(success),label='Maze outcome')
    st.session_state.dqn_last_reward=reward
    st.session_state.dqn_pending=False
    st.session_state.dqn_log[-1]['Maze success']=bool(success)

# Compatibility name used by the existing maze completion function.
record_rl_transition=record_game_transition

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
module_options = ["Game", "Learn", "Assess", "Learner Profile", "DQN Dashboard", "Training & Experiments"]
if st.session_state.get("rl_selected_page") not in module_options:
    st.session_state.rl_selected_page = "Game"
selected_page = st.sidebar.radio("Open module", module_options, key="rl_selected_page")

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

# ---------------- Learning / Assessment / DQN navigation ----------------

elif selected_page == "Learn":
    st.markdown("# Personalized Learning Tutor")
    st.caption("DQN selects an instructional action. Each choice opens an actual lesson or practice activity.")
    concept=level['concept']; state=learner_vector(concept)
    if concept not in st.session_state.diagnostic_scores:
        st.warning("Start with Assessment A (diagnostic) to establish your learner profile.")
    if st.button("Ask DQN to select my next learning activity",type="primary",disabled=concept not in st.session_state.diagnostic_scores):
        idx,reason,q=dqn_choose_action(state,explore=True)
        st.session_state.dqn_last_state=state.copy(); st.session_state.dqn_last_action_idx=idx
        st.session_state.dqn_last_action=DQN_ACTIONS[idx]; st.session_state.dqn_last_q_values=q.tolist()
        st.session_state.dqn_recommendation=DQN_ACTIONS[idx]; st.session_state.dqn_pending=True
        st.session_state.active_teaching_action=DQN_ACTIONS[idx]; st.session_state.teaching_completed=False
        st.success(f"DQN selected: **{DQN_ACTIONS[idx]}**")
        st.caption(reason)
    action=st.session_state.active_teaching_action
    if action and concept in st.session_state.diagnostic_scores:
        title,desc,example=LESSON_CONTENT[action]
        st.markdown(f"## {title}")
        st.write(desc)
        st.markdown("### Mini-lesson")
        if concept=="sequence":
            st.markdown("**Sequence:** instructions run from the first line to the last. Changing their order can change the robot's route.")
            st.code("move_forward()\nturn_right()\nmove_forward()",language="python")
            st.markdown("1. Move one cell.\n2. Turn right (position stays the same).\n3. Move in the new direction.")
        elif concept=="loops":
            st.markdown("**Loop:** repeat a block rather than writing the same command several times. Indentation shows what belongs to the loop.")
            st.code("for i in range(3):\n    move_forward()",language="python")
            st.markdown("The indented command runs **3 times**. `range(3)` means three iterations in this game.")
        else:
            st.markdown("**Conditional:** test a condition and execute the matching branch.")
            st.code("if wall_ahead():\n    turn_right()\nelse:\n    move_forward()",language="python")
            st.markdown("If a wall is ahead, turn right; otherwise, move forward. Only one branch executes.")
        with st.expander("Worked example: trace each step"):
            st.code(example,language="python")
            if concept=="sequence":
                st.table(pd.DataFrame([{"Step":1,"Instruction":"move_forward()","Effect":"Move one cell forward"},{"Step":2,"Instruction":"turn_right()","Effect":"Change facing direction"},{"Step":3,"Instruction":"move_forward()","Effect":"Move in the new direction"}]))
            elif concept=="loops":
                st.table(pd.DataFrame([{"Iteration":i,"Command":"move_forward()","Effect":f"Move {i} of 3"} for i in range(1,4)]))
            else:
                st.table(pd.DataFrame([{"Condition":"wall_ahead() = True","Executed branch":"if","Action":"turn_right()"},{"Condition":"wall_ahead() = False","Executed branch":"else","Action":"move_forward()"}]))
        st.markdown("### Interactive guided practice")
        if concept=="sequence":
            prompt="A robot must turn left first and then move. What is the first command?"
            opts=["move_forward()","turn_left()","turn_right()","for i in range(3):"]; correct=1
        elif concept=="loops":
            prompt="You need to move forward exactly 4 times. Which loop is correct?"
            opts=["for i in range(3): move_forward()","for i in range(4): move_forward()","for i in range(5): move_forward()","move_forward() once"]; correct=1
        else:
            prompt="If there is no wall ahead, which branch executes?"
            opts=["if wall_ahead():","else:","Both branches","Neither branch"]; correct=1
        with st.form("guided_practice_form"):
            response=st.radio(prompt,opts,key=f"guided_{concept}_{action}")
            submitted=st.form_submit_button("Check my answer")
        if submitted:
            is_correct=opts.index(response)==correct
            st.session_state.guided_practice_result={"concept":concept,"correct":is_correct}
            if is_correct: st.success("Correct! You applied the concept.")
            else:
                st.error("Not quite. Review the mini-lesson and try again.")
                st.info("Hint: " + ("command order matters." if concept=="sequence" else "the number in range(n) is the repetition count." if concept=="loops" else "else runs when the if condition is false."))
        result=st.session_state.get("guided_practice_result",{})
        if result.get("concept")==concept: st.caption("Latest guided practice: "+("Correct" if result.get("correct") else "Needs another try"))
        st.markdown("### Apply it")
        st.write("Try a maze challenge independently, then complete Assessment B (different questions) to measure learning.")
        if st.button("Mark teaching activity completed"):
            st.session_state.teaching_completed=True
            st.success("Activity recorded. Next: Game challenge, then Assessment B.")
    elif concept not in st.session_state.diagnostic_scores:
        st.info("Open **Assess → Diagnostic (Assessment A)** first.")
    if st.session_state.dqn_last_q_values:
        qdf=pd.DataFrame({'Teaching action':DQN_ACTIONS,'Predicted Q-value':st.session_state.dqn_last_q_values})
        st.markdown("### DQN action values for this learner")
        st.bar_chart(qdf.set_index('Teaching action'))


elif selected_page == "Assess":
    st.markdown("# Diagnostic & Post-Learning Assessment")
    st.caption("Assessment A and B use different questions targeting the same skills to reduce direct question repetition.")
    concept=level['concept']
    mode=st.radio("Assessment type",["Diagnostic (before teaching)","Post-test (after teaching)"],horizontal=True)
    is_post=mode.startswith('Post')
    quiz=(QUIZZES_B if is_post else QUIZZES_A)[concept]
    if is_post and not st.session_state.teaching_completed:
        st.warning("Complete a personalized teaching activity first. You can still use this page for a diagnostic.")
    with st.form('assessment_form'):
        st.write(f"Assessment for **{level['topic']}** · 3 questions")
        answers=[]
        for i,(question,options,correct,skill) in enumerate(quiz):
            answers.append(st.radio(f"{i+1}. {question}",options,key=f"quiz_{concept}_{mode}_{i}"))
        submitted=st.form_submit_button("Submit assessment",type='primary')
    if submitted:
        score=sum(1 for ans,(_,opts,correct,skill) in zip(answers,quiz) if opts.index(ans)==correct)
        pct=score/len(quiz)
        skill_result={}
        for ans,(_,opts,correct,skill) in zip(answers,quiz):
            skill_result[skill]=1.0 if opts.index(ans)==correct else 0.0
        if is_post:
            before=float(st.session_state.diagnostic_scores.get(concept,pct))
            st.session_state.posttest_scores[concept]=pct
            st.session_state.skill_scores[concept]={'accuracy':pct,'debugging':skill_result.get('debugging',pct),**skill_result}
            prev_state=st.session_state.dqn_last_state if st.session_state.dqn_pending else learner_vector(concept)
            next_state=learner_vector(concept)
            action_idx=st.session_state.dqn_last_action_idx if st.session_state.dqn_last_action_idx is not None else DQN_ACTIONS.index('Guided practice')
            reward=5*(pct-before)+3*pct+ (1.5 if pct>=.67 else 0) + (1 if pct>=.67 else -1)
            add_dqn_experience(prev_state,action_idx,reward,next_state,done=True,label='Pre/post assessment learning gain')
            st.session_state.dqn_last_reward=reward; st.session_state.dqn_pending=False
            st.session_state.teaching_completed=False
            st.success(f"Post-test score: **{score}/{len(quiz)} ({pct:.0%})** · DQN reward: **{reward:.2f}**")
            st.write(f"Learning gain from diagnostic: **{pct-before:+.0%}**")
        else:
            st.session_state.diagnostic_scores[concept]=pct
            st.session_state.skill_scores[concept]=skill_result
            st.success(f"Diagnostic score: **{score}/{len(quiz)} ({pct:.0%})**")
            weak=[k for k,v in skill_result.items() if v<1]
            st.write("Areas to focus:",", ".join(weak) if weak else "No errors on this diagnostic. Try a more challenging activity.")
        st.session_state.quiz_attempts.append({'Concept':concept,'Type':'Post-test' if is_post else 'Diagnostic','Score':pct,'Completed':time.strftime('%Y-%m-%d %H:%M:%S')})

elif selected_page == "Learner Profile":
    st.markdown("# Learner Profile")
    hist=st.session_state.challenge_history
    st.write("Profile combines diagnostic/post-test skill scores and recent game performance. Scores are provisional until enough data is collected.")
    concepts=['sequence','loops','conditionals']
    cols=st.columns(3)
    for col,c in zip(cols,concepts):
        score=st.session_state.posttest_scores.get(c,st.session_state.diagnostic_scores.get(c))
        col.metric(c.title(),"Not assessed" if score is None else f"{score:.0%}",help="Post-test score shown when available; otherwise diagnostic score.")
    if st.session_state.skill_scores:
        st.markdown("### Skill-level breakdown")
        rows=[]
        for c,skills in st.session_state.skill_scores.items():
            for skill,val in skills.items(): rows.append({'Concept':c.title(),'Skill':skill,'Score':val})
        if rows:
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("### Diagnostic vs post-test learning gain")
    gain_rows=[]
    for c in concepts:
        pre=st.session_state.diagnostic_scores.get(c); post=st.session_state.posttest_scores.get(c)
        gain_rows.append({"Concept":c.title(),"Assessment A":None if pre is None else f"{pre:.0%}","Assessment B":None if post is None else f"{post:.0%}","Gain":None if pre is None or post is None else f"{post-pre:+.0%}"})
    st.dataframe(pd.DataFrame(gain_rows),use_container_width=True,hide_index=True)
    if hist:
        df=pd.DataFrame(hist); st.markdown("### Game performance")
        a,b,c,d=st.columns(4); a.metric('Challenges',len(hist)); b.metric('Success rate',f"{100*sum(x.get('Success',False) for x in hist)/len(hist):.0f}%")
        c.metric('Avg errors',f"{df['Errors'].mean():.1f}"); d.metric('Avg time',f"{df['Completion Time (sec)'].mean():.1f}s")
        st.dataframe(df,use_container_width=True,hide_index=True)
    if st.session_state.quiz_attempts: st.markdown('### Assessment history'); st.dataframe(pd.DataFrame(st.session_state.quiz_attempts),use_container_width=True,hide_index=True)

elif selected_page == "DQN Dashboard":
    st.markdown("# Deep Q-Network (DQN) Dashboard")
    st.caption("Inspectable view of the DQN that chooses teaching strategies. It does not issue movement commands to the robot.")
    state=learner_vector(); q=dqn_forward(st.session_state.dqn_online,state)[0]
    c1,c2,c3,c4=st.columns(4); c1.metric('Replay experiences',len(st.session_state.dqn_memory)); c2.metric('Training updates',st.session_state.dqn_training_steps); c3.metric('Exploration ε',f"{st.session_state.dqn_epsilon:.3f}"); c4.metric('Latest reward',"—" if st.session_state.dqn_last_reward is None else f"{st.session_state.dqn_last_reward:.2f}")
    st.markdown('### Current learner state (network input)')
    st.dataframe(pd.DataFrame({'Feature index':range(len(state)),'Value':state}),hide_index=True,use_container_width=True)
    qdf=pd.DataFrame({'Action':DQN_ACTIONS,'Q-value':q}).sort_values('Q-value',ascending=False)
    st.markdown('### Predicted Q-values by instructional action')
    st.bar_chart(qdf.set_index('Action')); st.dataframe(qdf,hide_index=True,use_container_width=True)
    st.markdown('### DQN architecture')
    st.code('Learner state (12 features) → Dense(64, ReLU) → Dense(64, ReLU) → 8 Q-values\nOnline network + Target network\nExperience replay buffer → mini-batch gradient updates',language='text')
    st.markdown('### DQN equations')
    st.latex(r"y=r+\gamma\max_{a'}Q_{target}(s',a')")
    st.latex(r"L=(y-Q_{online}(s,a))^2")
    st.write(f"Learning rate: {st.session_state.dqn_learning_rate:.4f} · Discount γ: {st.session_state.dqn_gamma:.2f} · ε: {st.session_state.dqn_epsilon:.2f} · Target update every 10 training steps")
    with st.expander('Configure DQN'):
        st.session_state.dqn_learning_rate=st.slider('Learning rate',0.0005,0.05,float(st.session_state.dqn_learning_rate),0.0005)
        st.session_state.dqn_gamma=st.slider('Discount factor γ',0.0,0.99,float(st.session_state.dqn_gamma),0.01)
        st.session_state.dqn_epsilon=st.slider('Exploration ε',0.0,1.0,float(st.session_state.dqn_epsilon),0.01)
    if st.session_state.dqn_last_action is not None:
        st.markdown('### Latest decision')
        st.write(f"Action: **{st.session_state.dqn_last_action}**")
        st.write('Latest reward:',st.session_state.dqn_last_reward)
        st.write('Latest state/action Q-values:',st.session_state.dqn_last_q_values)
    if st.session_state.dqn_log:
        st.markdown('### Learning / transition log'); st.dataframe(pd.DataFrame(st.session_state.dqn_log),use_container_width=True,hide_index=True)
        st.download_button('Download DQN log CSV',pd.DataFrame(st.session_state.dqn_log).to_csv(index=False),'codequest_dqn_log.csv','text/csv')
    if st.session_state.dqn_memory:
        st.markdown('### Replay buffer preview'); st.write(f"Stored transitions: {len(st.session_state.dqn_memory)} (maximum 5,000)")

elif selected_page == "Training & Experiments":
    st.markdown('# DQN Training & Experiments')
    st.warning('Synthetic training is a software-development simulation using heuristic preferred actions. It demonstrates the training pipeline, not educational efficacy or real learner performance.')

    col1, col2, col3 = st.columns(3)
    with col1:
        episodes = st.number_input('Episodes in this run', min_value=50, max_value=3000, value=300, step=50)
    with col2:
        epsilon_start = st.slider('Starting exploration ε', 0.0, 1.0, float(st.session_state.dqn_epsilon), 0.05)
    with col3:
        epsilon_min = st.slider('Minimum ε', 0.0, 1.0, 0.05, 0.01)

    epsilon_decay = st.slider('Exploration decay per episode', 0.95, 1.0, 0.995, 0.001,
                              help='ε is reduced after each synthetic episode, down to the minimum. This schedule is used in this simulation run.')

    if st.button('Train DQN with simulated learner profiles', type='primary'):
        profiles = [('struggling',0.25,0.3,0.8),('fast',0.9,0.9,0.1),
                    ('inconsistent',0.5,0.55,0.5),('hint-dependent',0.45,0.5,0.7)]
        st.session_state.dqn_epsilon = float(epsilon_start)
        run_rewards, run_actions, run_qmeans, run_eps, run_losses = [], [], [], [], []
        for ep in range(int(episodes)):
            name, mastery, acc, err = random.choice(profiles)
            concept = random.choice(['sequence','loops','conditionals'])
            state = np.array([concept=='sequence',concept=='loops',concept=='conditionals',
                mastery,acc,random.random(),err,random.random(),random.random(),mastery,acc,1-err],dtype=float)
            ideal = {'struggling':3 if err>0.5 else 2,'fast':5,'inconsistent':6,'hint-dependent':2}[name]
            idx, reason, q = dqn_choose_action(state, explore=True)
            reward = (3.0 if idx==ideal else -0.5) + random.uniform(-0.5,0.5)
            next_state = state.copy()
            next_state[3] = min(1, mastery+random.uniform(0,0.08))
            epsilon_used = float(st.session_state.dqn_epsilon)
            loss = add_dqn_experience(state,idx,reward,next_state,False,'Synthetic profile')
            run_rewards.append(float(reward)); run_actions.append(DQN_ACTIONS[idx])
            run_qmeans.append(float(np.mean(q))); run_eps.append(epsilon_used)
            if loss is not None: run_losses.append(float(loss))
            st.session_state.dqn_epsilon = max(float(epsilon_min), st.session_state.dqn_epsilon*float(epsilon_decay))
            st.session_state.dqn_epsilon_history.append({"Episode":len(st.session_state.dqn_episode_rewards)+ep+1,"Epsilon":epsilon_used})

        run_start = len(st.session_state.dqn_episode_rewards)
        st.session_state.dqn_episode_rewards.extend(run_rewards)
        for action in run_actions:
            st.session_state.dqn_action_counts[action] = st.session_state.dqn_action_counts.get(action,0)+1
        for j, (reward, action, qmean, eps) in enumerate(zip(run_rewards,run_actions,run_qmeans,run_eps)):
            st.session_state.dqn_synthetic_history.append({
                "Episode":run_start+j+1,"Reward":reward,"Action":action,
                "Mean Q-value":qmean,"Epsilon":eps,"Profile source":"Synthetic"
            })
        st.success(f"Completed {int(episodes)} simulated episodes. Replay buffer: {len(st.session_state.dqn_memory)} experiences · Training updates: {st.session_state.dqn_training_steps}.")

    if st.session_state.dqn_episode_rewards:
        rewards = st.session_state.dqn_episode_rewards
        window = min(25,len(rewards))
        moving = [float(np.mean(rewards[max(0,i-window+1):i+1])) for i in range(len(rewards))]
        synth_df = pd.DataFrame(st.session_state.dqn_synthetic_history)
        loss_df = pd.DataFrame(st.session_state.dqn_loss_history)

        st.markdown('## Experiment summary')
        a,b,c,d,e = st.columns(5)
        a.metric('Episodes recorded',len(rewards))
        b.metric('Replay experiences',len(st.session_state.dqn_memory))
        c.metric('Training updates',st.session_state.dqn_training_steps)
        d.metric('Target-network syncs',st.session_state.dqn_training_steps//10)
        e.metric('Current ε',f"{st.session_state.dqn_epsilon:.3f}")
        st.metric('Recent mean reward (last 25 episodes)',f"{np.mean(rewards[-window:]):.2f}")

        st.markdown('## 1. Reward per episode and moving average')
        st.line_chart(pd.DataFrame({"Episode":range(1,len(rewards)+1),"Reward":rewards,"Moving average (25)":moving}).set_index("Episode"))

        if not loss_df.empty:
            st.markdown('## 2. Training loss per update')
            st.line_chart(loss_df.set_index("Update")[["Loss"]])
            st.caption("Loss is the squared TD error from minibatch training updates; lower loss is not by itself proof of better teaching outcomes.")

        if not synth_df.empty:
            st.markdown('## 3. Instructional-action selection distribution')
            counts = synth_df["Action"].value_counts().reindex(DQN_ACTIONS,fill_value=0)
            st.bar_chart(counts.rename("Selections"))

            st.markdown('## 4. Mean predicted Q-value by episode')
            st.line_chart(synth_df.set_index("Episode")[["Mean Q-value"]])

            st.markdown('## 5. Exploration (ε) over episodes')
            st.line_chart(synth_df.set_index("Episode")[["Epsilon"]])

            st.markdown('## Episode-level experiment log')
            st.dataframe(synth_df,use_container_width=True,hide_index=True)
            st.download_button("Download synthetic experiment CSV",synth_df.to_csv(index=False),
                "codequest_synthetic_experiment.csv","text/csv")

    st.markdown('### Experimental comparison plan')
    st.table(pd.DataFrame([
        {'Approach':'Fixed instruction','Adaptation':'Predefined sequence'},
        {'Approach':'Rule-based tutor','Adaptation':'Manual learner thresholds'},
        {'Approach':'DQN tutor','Adaptation':'Neural network learns action values from reward'}]))

    st.markdown('### Reset DQN / experiment')
    if st.button('Reset network, replay memory and experiment graphs',type='secondary'):
        st.session_state.dqn_online=init_network(); st.session_state.dqn_target=copy_network(st.session_state.dqn_online)
        st.session_state.dqn_memory=[]; st.session_state.dqn_log=[]; st.session_state.dqn_training_steps=0
        st.session_state.dqn_episode_rewards=[]; st.session_state.dqn_loss_history=[]
        st.session_state.dqn_synthetic_history=[]; st.session_state.dqn_action_counts={}
        st.session_state.dqn_epsilon_history=[]
        st.session_state.dqn_epsilon=0.25
        st.success('DQN and experiment history reset.'); st.rerun()

# Adaptive recommendation summary on Game page
if selected_page == "Game":
    st.markdown('---'); st.markdown('## DQN Tutor')
    st.caption('DQN recommends teaching strategies based on learner state. Use Learn and Assess to complete the adaptive instructional loop.')
    st.write('Current learner assessment:',st.session_state.diagnostic_scores.get(level['concept'],'Not assessed'))
    st.info('To begin personalized learning: open **Assess** → complete diagnostic → open **Learn** → request DQN strategy → complete activity → take post-test.')
