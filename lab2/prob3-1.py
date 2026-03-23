"""
task 3.1 - advanced problems
minimum 20 actions
over 50 states

warehouse robot - sorting packages with placement logic
4 different locations, 2 packages, robot state
"""
import time

from stripsProblem import STRIPS_domain, Strips, Planning_problem
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP


# 5*6*6*2 = 360 states
# robot may be one of the four locations - in, regal A, regal B, out
# same for packages but they may be also taken by the robot
# robot may be full or empty
feature_domain = {
    "robot_location": {"in", "regA", "regB", "regC", "out"},
    'p1_location': {'in', 'regA', 'regB', "regC", 'out', 'robot'},
    'p2_location': {'in', 'regA', 'regB', "regC", 'out', 'robot'},
    'robot': {'full', 'empty'}
}

# move - robot moves from one location to another, sequential order preserved, 8 actions
# pick - robot picks given package in a given location, 10 actions
# drop - robot drops given package in a given location, 10 actions
# 28 actions in total
actions_domain = {
    Strips('move_in_A', {'robot_location':'in'}, {'robot_location':'regA'}),
    Strips('move_A_in', {'robot_location': 'regA'}, {'robot_location': 'in'}),
    Strips('move_A_B', {'robot_location': 'regA'}, {'robot_location': 'regB'}),
    Strips('move_B_A', {'robot_location': 'regB'}, {'robot_location': 'regA'}),
    Strips('move_B_C', {'robot_location': 'regB'}, {'robot_location': 'regC'}),
    Strips('move_C_B', {'robot_location': 'regC'}, {'robot_location': 'regB'}),
    Strips('move_C_out', {'robot_location': 'regC'}, {'robot_location': 'out'}),
    Strips('move_out_C', {'robot_location': 'out'}, {'robot_location': 'regC'}),

    Strips('pick_p1_in', {'robot_location': 'in', 'p1_location': 'in', 'robot': 'empty'}, {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_A', {'robot_location': 'regA', 'p1_location': 'regA', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_B', {'robot_location': 'regB', 'p1_location': 'regB', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_C', {'robot_location': 'regC', 'p1_location': 'regC', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_out', {'robot_location': 'out', 'p1_location': 'out', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_in', {'robot_location': 'in', 'p2_location': 'in', 'robot': 'empty'}, {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_A', {'robot_location': 'regA', 'p2_location': 'regA', 'robot': 'empty'},
           {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_B', {'robot_location': 'regB', 'p2_location': 'regB', 'robot': 'empty'},
           {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_C', {'robot_location': 'regC', 'p2_location': 'regC', 'robot': 'empty'},
           {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_out', {'robot_location': 'out', 'p2_location': 'out', 'robot': 'empty'},
           {'p2_location': 'robot', "robot": 'full'}),

    Strips('drop_p1_in', {'robot_location': 'in', 'p1_location': 'robot'}, {'p1_location': 'in', 'robot': 'empty'}),
    Strips('drop_p1_A', {'robot_location': 'regA', 'p1_location': 'robot'}, {'p1_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p1_B', {'robot_location': 'regB', 'p1_location': 'robot'}, {'p1_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p1_C', {'robot_location': 'regC', 'p1_location': 'robot'}, {'p1_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p1_out', {'robot_location': 'out', 'p1_location': 'robot'}, {'p1_location': 'out', 'robot': 'empty'}),
    Strips('drop_p2_in', {'robot_location': 'in', 'p2_location': 'robot'}, {'p2_location': 'in', 'robot': 'empty'}),
    Strips('drop_p2_A', {'robot_location': 'regA', 'p2_location': 'robot'}, {'p2_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p2_B', {'robot_location': 'regB', 'p2_location': 'robot'}, {'p2_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p2_C', {'robot_location': 'regC', 'p2_location': 'robot'}, {'p2_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p2_out', {'robot_location': 'out', 'p2_location': 'robot'}, {'p2_location': 'out', 'robot': 'empty'}),
}

# creating delivery domain out of features and actions
delivery_domain = STRIPS_domain(
    feature_domain,
    actions_domain
)

# creating a problem for robot - move 2 packages from certain locations to different ones
problem = Planning_problem(delivery_domain,
                           {'robot_location': 'out', 'p1_location': 'in', 'p2_location': 'in', 'robot': 'empty'},
                           {'p1_location': 'regC', 'p2_location': 'out', 'robot': 'empty', 'robot_location': 'in'}
)

# A* search
start = time.time()
SearcherMPP(Forward_STRIPS(problem)).search()
end = time.time()
print("Searching took %.6f seconds" % (end - start))