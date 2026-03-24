"""
STRIPS domain - advanced problem
minimum 20 actions
over 50 states

warehouse robot - sorting packages with placement logic
7 different locations, 4 packages, robot state
"""
import time

from stripsProblem import STRIPS_domain, Strips, Planning_problem
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP
from searchGeneric import AStarSearcher
from searchBranchAndBound import DF_branch_and_bound


# 7*8*8*8*8*2 = 57 344 states
# robot may be one of the 7 locations - in, regal A, regal B, regal C, regal D, regal E, out (7 states)
# same for packages but they may be also taken by the robot (8 states for each robot)
# robot may be full or empty (2 states)
feature_domain = {
    "robot_location": {"in", "regA", "regB", "regC", "regD", "regE", "out"},
    'p1_location': {'in', 'regA', 'regB', "regC", "regD", "regE", 'out', 'robot'},
    'p2_location': {'in', 'regA', 'regB', "regC", "regD", "regE", 'out', 'robot'},
    'p3_location': {'in', 'regA', 'regB', "regC", "regD", "regE", 'out', 'robot'},
    'p4_location': {'in', 'regA', 'regB', "regC", "regD", "regE", 'out', 'robot'},
    'robot': {'full', 'empty'}
}

# move - robot moves from one location to another, sequential order preserved, 12 actions
# pick - robot picks given package in a given location, 28 actions
# drop - robot drops given package in a given location, 28 actions
# 68 actions in total
actions_domain = {
    Strips('move_in_A', {'robot_location':'in'}, {'robot_location':'regA'}),
    Strips('move_A_in', {'robot_location': 'regA'}, {'robot_location': 'in'}),
    Strips('move_A_B', {'robot_location': 'regA'}, {'robot_location': 'regB'}),
    Strips('move_B_A', {'robot_location': 'regB'}, {'robot_location': 'regA'}),
    Strips('move_B_C', {'robot_location': 'regB'}, {'robot_location': 'regC'}),
    Strips('move_C_B', {'robot_location': 'regC'}, {'robot_location': 'regB'}),
    Strips('move_C_D', {'robot_location': 'regC'}, {'robot_location': 'regD'}),
    Strips('move_D_C', {'robot_location': 'regD'}, {'robot_location': 'regC'}),
    Strips('move_D_E', {'robot_location': 'regD'}, {'robot_location': 'regE'}),
    Strips('move_E_D', {'robot_location': 'regE'}, {'robot_location': 'regD'}),
    Strips('move_E_out', {'robot_location': 'regE'}, {'robot_location': 'out'}),
    Strips('move_out_E', {'robot_location': 'out'}, {'robot_location': 'regE'}),

    Strips('pick_p1_in', {'robot_location': 'in', 'p1_location': 'in', 'robot': 'empty'}, {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_A', {'robot_location': 'regA', 'p1_location': 'regA', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_B', {'robot_location': 'regB', 'p1_location': 'regB', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_C', {'robot_location': 'regC', 'p1_location': 'regC', 'robot': 'empty'},
           {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_D', {'robot_location': 'regD', 'p1_location': 'regD', 'robot': 'empty'},
               {'p1_location': 'robot', "robot": 'full'}),
    Strips('pick_p1_E', {'robot_location': 'regE', 'p1_location': 'regE', 'robot': 'empty'},
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
    Strips('pick_p2_D', {'robot_location': 'regD', 'p2_location': 'regD', 'robot': 'empty'},
               {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_E', {'robot_location': 'regE', 'p2_location': 'regE', 'robot': 'empty'},
               {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p2_out', {'robot_location': 'out', 'p2_location': 'out', 'robot': 'empty'},
           {'p2_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_in', {'robot_location': 'in', 'p3_location': 'in', 'robot': 'empty'}, {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_A', {'robot_location': 'regA', 'p3_location': 'regA', 'robot': 'empty'},
           {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_B', {'robot_location': 'regB', 'p3_location': 'regB', 'robot': 'empty'},
           {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_C', {'robot_location': 'regC', 'p3_location': 'regC', 'robot': 'empty'},
           {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_D', {'robot_location': 'regD', 'p3_location': 'regD', 'robot': 'empty'},
               {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_E', {'robot_location': 'regE', 'p3_location': 'regE', 'robot': 'empty'},
               {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p3_out', {'robot_location': 'out', 'p3_location': 'out', 'robot': 'empty'},
           {'p3_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_in', {'robot_location': 'in', 'p4_location': 'in', 'robot': 'empty'}, {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_A', {'robot_location': 'regA', 'p4_location': 'regA', 'robot': 'empty'},
           {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_B', {'robot_location': 'regB', 'p4_location': 'regB', 'robot': 'empty'},
           {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_C', {'robot_location': 'regC', 'p4_location': 'regC', 'robot': 'empty'},
           {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_D', {'robot_location': 'regD', 'p4_location': 'regD', 'robot': 'empty'},
               {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_E', {'robot_location': 'regE', 'p4_location': 'regE', 'robot': 'empty'},
               {'p4_location': 'robot', "robot": 'full'}),
    Strips('pick_p4_out', {'robot_location': 'out', 'p4_location': 'out', 'robot': 'empty'},
           {'p4_location': 'robot', "robot": 'full'}),

    Strips('drop_p1_in', {'robot_location': 'in', 'p1_location': 'robot'}, {'p1_location': 'in', 'robot': 'empty'}),
    Strips('drop_p1_A', {'robot_location': 'regA', 'p1_location': 'robot'}, {'p1_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p1_B', {'robot_location': 'regB', 'p1_location': 'robot'}, {'p1_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p1_C', {'robot_location': 'regC', 'p1_location': 'robot'}, {'p1_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p1_D', {'robot_location': 'regD', 'p1_location': 'robot'}, {'p1_location': 'regD', 'robot': 'empty'}),
    Strips('drop_p1_E', {'robot_location': 'regE', 'p1_location': 'robot'}, {'p1_location': 'regE', 'robot': 'empty'}),
    Strips('drop_p1_out', {'robot_location': 'out', 'p1_location': 'robot'}, {'p1_location': 'out', 'robot': 'empty'}),
    Strips('drop_p2_in', {'robot_location': 'in', 'p2_location': 'robot'}, {'p2_location': 'in', 'robot': 'empty'}),
    Strips('drop_p2_A', {'robot_location': 'regA', 'p2_location': 'robot'}, {'p2_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p2_B', {'robot_location': 'regB', 'p2_location': 'robot'}, {'p2_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p2_C', {'robot_location': 'regC', 'p2_location': 'robot'}, {'p2_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p2_D', {'robot_location': 'regD', 'p2_location': 'robot'}, {'p2_location': 'regD', 'robot': 'empty'}),
    Strips('drop_p2_E', {'robot_location': 'regE', 'p2_location': 'robot'}, {'p2_location': 'regE', 'robot': 'empty'}),
    Strips('drop_p2_out', {'robot_location': 'out', 'p2_location': 'robot'}, {'p2_location': 'out', 'robot': 'empty'}),
    Strips('drop_p3_in', {'robot_location': 'in', 'p3_location': 'robot'}, {'p3_location': 'in', 'robot': 'empty'}),
    Strips('drop_p3_A', {'robot_location': 'regA', 'p3_location': 'robot'}, {'p3_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p3_B', {'robot_location': 'regB', 'p3_location': 'robot'}, {'p3_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p3_C', {'robot_location': 'regC', 'p3_location': 'robot'}, {'p3_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p3_D', {'robot_location': 'regD', 'p3_location': 'robot'}, {'p3_location': 'regD', 'robot': 'empty'}),
    Strips('drop_p3_E', {'robot_location': 'regE', 'p3_location': 'robot'}, {'p3_location': 'regE', 'robot': 'empty'}),
    Strips('drop_p3_out', {'robot_location': 'out', 'p3_location': 'robot'}, {'p3_location': 'out', 'robot': 'empty'}),
    Strips('drop_p4_in', {'robot_location': 'in', 'p4_location': 'robot'}, {'p4_location': 'in', 'robot': 'empty'}),
    Strips('drop_p4_A', {'robot_location': 'regA', 'p4_location': 'robot'}, {'p4_location': 'regA', 'robot': 'empty'}),
    Strips('drop_p4_B', {'robot_location': 'regB', 'p4_location': 'robot'}, {'p4_location': 'regB', 'robot': 'empty'}),
    Strips('drop_p4_C', {'robot_location': 'regC', 'p4_location': 'robot'}, {'p4_location': 'regC', 'robot': 'empty'}),
    Strips('drop_p4_D', {'robot_location': 'regD', 'p4_location': 'robot'}, {'p4_location': 'regD', 'robot': 'empty'}),
    Strips('drop_p4_E', {'robot_location': 'regE', 'p4_location': 'robot'}, {'p4_location': 'regE', 'robot': 'empty'}),
    Strips('drop_p4_out', {'robot_location': 'out', 'p4_location': 'robot'}, {'p4_location': 'out', 'robot': 'empty'}),

}

# creating delivery domain out of features and actions
delivery_domain = STRIPS_domain(
    feature_domain,
    actions_domain
)

# creating a problem for robot - move 2 packages from certain locations to different ones
problem = Planning_problem(delivery_domain,
                           {'robot_location': 'out', 'p1_location': 'in', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regA', 'robot': 'empty'},
                           {'p1_location': 'regC', 'p2_location': 'out', 'p3_location': 'regA', 'p4_location': 'in', 'robot': 'empty', 'robot_location': 'in'}
)

simple_problem1 = Planning_problem(delivery_domain,
                                   {'robot_location': 'in', 'p1_location': 'regB', 'p2_location': 'in', 'p3_location': 'in', 'p4_location': 'in', 'robot': 'empty'},
                                   {'p1_location': 'regE'})

simple_problem2 = Planning_problem(delivery_domain,
                                   {'robot_location': 'out', 'p1_location': 'regB', 'p2_location': 'in', 'p3_location': 'in', 'p4_location': 'in', 'robot': 'empty'},
                                   {'p1_location': 'regC', 'p2_location': 'regB'})

simple_problem3 = Planning_problem(delivery_domain,
                                   {'robot_location': 'in', 'p1_location': 'regB', 'p2_location': 'regD', 'p3_location': 'in', 'p4_location': 'in', 'robot': 'empty'},
                                   {'p1_location': 'regE', 'p2_location': 'out'})

advanced_problem1 = Planning_problem(delivery_domain,
                                   {'robot_location': 'regA', 'p1_location': 'regC', 'p2_location': 'regC', 'p3_location': 'regC', 'p4_location': 'regC', 'robot': 'empty'},
                                   {'p1_location': 'out', 'p2_location': 'regB', 'p3_location': 'regA', 'p4_location': 'regD', 'robot': 'empty'})

advanced_problem2 = Planning_problem(delivery_domain,
                                   {'robot_location': 'in', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'},
                                   {'robot_location': 'out', 'p1_location': 'out', 'p2_location': 'regC', 'p3_location': 'regB', 'p4_location': 'robot', 'robot': 'full'})

advanced_problem3 = Planning_problem(delivery_domain,
                                   {'robot_location': 'in', 'p1_location': 'regA', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'empty'},
                                   {'p1_location': 'out', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'empty'})

# A* search
start = time.time()
SearcherMPP(Forward_STRIPS(advanced_problem3)).search()
# AStarSearcher(Forward_STRIPS(advanced_problem1)).search()
# DF_branch_and_bound(Forward_STRIPS(simple_problem1)).search()
end = time.time()
print("Searching took %.6f seconds" % (end - start))