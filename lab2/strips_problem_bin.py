"""
STRIPS domain & problem - advanced problem
minimum 20 actions
over 50 states

warehouse robot - sorting packages with placement logic
4 different locations, 2 packages, robot state
"""
import time

from stripsProblem import STRIPS_domain, Strips, Planning_problem
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP


# 2^18 = 262 144 states
# robot may be one of the four locations - in, regal A, regal B, out
# same for packages but they may be also taken by the robot
# robot may be full or empty
boolean = {True, False}
feature_domain = {
    "robot_in": boolean,
    "robot_A": boolean,
    "robot_B": boolean,
    "robot_C": boolean,
    "robot_out": boolean,
    'p1_in': boolean,
    'p1_A': boolean,
    'p1_B': boolean,
    'p1_C': boolean,
    'p1_out': boolean,
    'p1_robot': boolean,
    'p2_in': boolean,
    'p2_A': boolean,
    'p2_B': boolean,
    'p2_C': boolean,
    'p2_out': boolean,
    'p2_robot': boolean,
    'robot_full': boolean
}

# move - robot moves from one location to another, sequential order preserved, 8 actions
# pick - robot picks given package in a given location, 10 actions
# drop - robot drops given package in a given location, 10 actions
# 28 actions in total
actions_domain = {
    Strips('move_in_A', {'robot_in': True}, {'robot_A': True, 'robot_in': False}),
    Strips('move_A_in', {'robot_A': True}, {'robot_in': True, 'robot_A': False}),
    Strips('move_A_B', {'robot_A': True}, {'robot_B': True, 'robot_A': False}),
    Strips('move_B_A', {'robot_B': True}, {'robot_A': True, 'robot_B': False}),
    Strips('move_B_C', {'robot_B': True}, {'robot_C': True, 'robot_B': False}),
    Strips('move_C_B', {'robot_C': True}, {'robot_B': True, 'robot_C': False}),
    Strips('move_C_out', {'robot_C': True}, {'robot_out': True, 'robot_C': False}),
    Strips('move_out_C', {'robot_out': True}, {'robot_C': True, 'robot_out': False}),

    Strips('pick_p1_in', {'robot_in': True, 'p1_in': True, 'robot_full': False}, {'p1_in': False, 'p1_robot': True, "robot_full": True}),
    Strips('pick_p1_A', {'robot_A': True, 'p1_A': True, 'robot_full': False}, {'p1_A': False, 'p1_robot': True, "robot_full": True}),
    Strips('pick_p1_B', {'robot_B': True, 'p1_B': True, 'robot_full': False}, {'p1_B': False, 'p1_robot': True, "robot_full": True}),
    Strips('pick_p1_C', {'robot_C': True, 'p1_C': True, 'robot_full': False}, {'p1_C': False, 'p1_robot': True, "robot_full": True}),
    Strips('pick_p1_out', {'robot_out': True, 'p1_out': True, 'robot_full': False}, {'p1_out': False, 'p1_robot': True, "robot_full": True}),
    Strips('pick_p2_in', {'robot_in': True, 'p2_in': True, 'robot_full': False}, {'p2_in': False, 'p2_robot': True, "robot_full": True}),
    Strips('pick_p2_A', {'robot_A': True, 'p2_A': True, 'robot_full': False}, {'p2_A': False, 'p2_robot': True, "robot_full": True}),
    Strips('pick_p2_B', {'robot_B': True, 'p2_B': True, 'robot_full': False}, {'p2_B': False, 'p2_robot': True, "robot_full": True}),
    Strips('pick_p2_C', {'robot_C': True, 'p2_C': True, 'robot_full': False},{'p2_C': False, 'p2_robot': True, "robot_full": True}),
    Strips('pick_p2_out', {'robot_out': True, 'p2_out': True, 'robot_full': False}, {'p2_out': False, 'p2_robot': True, "robot_full": True}),

    Strips('drop_p1_in', {'robot_in': True, 'p1_robot': True}, {'p1_in': True, 'p1_robot': False, 'robot_full': False}),
    Strips('drop_p1_A', {'robot_A': True, 'p1_robot': True}, {'p1_A': True, 'p1_robot': False, 'robot_full': False}),
    Strips('drop_p1_B', {'robot_B': True, 'p1_robot': True}, {'p1_B': True, 'p1_robot': False, 'robot_full': False}),
    Strips('drop_p1_C', {'robot_C': True, 'p1_robot': True}, {'p1_C': True, 'p1_robot': False, 'robot_full': False}),
    Strips('drop_p1_out', {'robot_out': True, 'p1_robot': True}, {'p1_out': True, 'p1_robot': False, 'robot_full': False}),
    Strips('drop_p2_in', {'robot_in': True, 'p2_robot': True}, {'p2_in': True, 'p2_robot': False, 'robot_full': False}),
    Strips('drop_p2_A', {'robot_A': True, 'p2_robot': True}, {'p2_A': True, 'p2_robot': False, 'robot_full': False}),
    Strips('drop_p2_B', {'robot_B': True, 'p2_robot': True}, {'p2_B': True, 'p2_robot': False, 'robot_full': False}),
    Strips('drop_p2_C', {'robot_C': True, 'p2_robot': True}, {'p2_C': True, 'p2_robot': False, 'robot_full': False}),
    Strips('drop_p2_out', {'robot_out': True, 'p2_robot': True}, {'p2_out': True, 'p2_robot': False, 'robot_full': False}),
}

# creating delivery domain out of features and actions
delivery_domain = STRIPS_domain(
    feature_domain,
    actions_domain
)

# creating a problem for robot - move 2 packages from certain locations to different ones
problem = Planning_problem(delivery_domain,
                           {"robot_in": False,
                                    "robot_A": False,
                                    "robot_B": False,
                                    "robot_C": False,
                                    "robot_out": True,
                                    'p1_in': True,
                                    'p1_A': False,
                                    'p1_B': False,
                                    'p1_C': False,
                                    'p1_out': False,
                                    'p1_robot': False,
                                    'p2_in': True,
                                    'p2_A': False,
                                    'p2_B': False,
                                    'p2_C': False,
                                    'p2_out': False,
                                    'p2_robot': False,
                                    'robot_full': False},
                           {'p1_C': True, 'p2_out': True, 'robot_full': False, 'robot_in': True}
)

# A* search
start = time.time()
SearcherMPP(Forward_STRIPS(problem)).search()
end = time.time()
print("Searching took %.6f seconds" % (end - start))