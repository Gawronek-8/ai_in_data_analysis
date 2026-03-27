import re

### Script for parsing the stdout to latex

def parse_to_latex(input_data):
    matches = re.findall(r'--([a-zA-Z0-9_]+)-->', input_data)

    latex_lines = []
    for i, match in enumerate(matches):
        escaped = match.replace('_', r'\_')
        line = f"\\texttt{{{escaped}}}"

        if i < len(matches) - 1:
            line += r" $\rightarrow$"

        if i > 0:
            line = "    " + line

        latex_lines.append(line)

    return "\n".join(latex_lines)


print(parse_to_latex("""y 
Solution: {'robot_location': 'in', 'p1_location': 'regA', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'empty'}
   --move_in_A--> {'robot_location': 'regA', 'p1_location': 'regA', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'empty'}
   --pick_p1_A--> {'robot_location': 'regA', 'p1_location': 'robot', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'full'}
   --move_A_B--> {'robot_location': 'regB', 'p1_location': 'robot', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'full'}
   --move_B_C--> {'robot_location': 'regC', 'p1_location': 'robot', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'full'}
   --move_C_D--> {'robot_location': 'regD', 'p1_location': 'robot', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'full'}
   --drop_p1_D--> {'robot_location': 'regD', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regD', 'robot': 'empty'}
   --pick_p4_D--> {'robot_location': 'regD', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'robot', 'robot': 'full'}
   --move_D_C--> {'robot_location': 'regC', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'robot', 'robot': 'full'}
   --move_C_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'robot', 'robot': 'full'}
   --drop_p4_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regB', 'robot': 'empty'}
   --move_B_C--> {'robot_location': 'regC', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regC', 'p4_location': 'regB', 'robot': 'empty'}
   --pick_p3_C--> {'robot_location': 'regC', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'robot', 'p4_location': 'regB', 'robot': 'full'}
   --move_C_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'robot', 'p4_location': 'regB', 'robot': 'full'}
   --move_B_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'robot', 'p4_location': 'regB', 'robot': 'full'}
   --drop_p3_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'empty'}
   --move_A_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regB', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'empty'}
   --pick_p2_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'}
   --move_B_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'}
   --move_A_in--> {'robot_location': 'in', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'} (cost: 19)
 10968 paths have been expanded and 5424 paths remain in the frontier
Solution: {'robot_location': 'in', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'}
   --move_in_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'robot', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'full'}
   --drop_p2_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'empty'}
   --move_A_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regA', 'p4_location': 'regB', 'robot': 'empty'}
   --pick_p4_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regA', 'p4_location': 'robot', 'robot': 'full'}
   --move_B_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regA', 'p4_location': 'robot', 'robot': 'full'}
   --drop_p4_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regA', 'p4_location': 'regA', 'robot': 'empty'}
   --pick_p3_A--> {'robot_location': 'regA', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'robot', 'p4_location': 'regA', 'robot': 'full'}
   --move_A_B--> {'robot_location': 'regB', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'robot', 'p4_location': 'regA', 'robot': 'full'}
   --move_B_C--> {'robot_location': 'regC', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'robot', 'p4_location': 'regA', 'robot': 'full'}
   --move_C_D--> {'robot_location': 'regD', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'robot', 'p4_location': 'regA', 'robot': 'full'}
   --drop_p3_D--> {'robot_location': 'regD', 'p1_location': 'regD', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'empty'}
   --pick_p1_D--> {'robot_location': 'regD', 'p1_location': 'robot', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'full'}
   --move_D_E--> {'robot_location': 'regE', 'p1_location': 'robot', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'full'}
   --move_E_out--> {'robot_location': 'out', 'p1_location': 'robot', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'full'}
   --drop_p1_out--> {'robot_location': 'out', 'p1_location': 'out', 'p2_location': 'regA', 'p3_location': 'regD', 'p4_location': 'regA', 'robot': 'empty'} (cost: 15)
 4529 paths have been expanded and 3357 paths remain in the frontier"""))