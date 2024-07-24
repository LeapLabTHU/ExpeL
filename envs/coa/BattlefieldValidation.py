import json
import numpy as np
from typing import Dict, List, Tuple, Set

class BattlefieldValidation:

    def __init__(self, supporting_information: str, validation_json: str = "") -> None:

        # Ensure that there are no single quotes in the json string. If not, then the code will crash.
        # The json.loads() function recognizes double quotes around keys and values.
        supporting_information = supporting_information.replace("'", '"')

        # Change later for future validation tests. This is a placeholder to use for train.py
        self.resp = json.loads("""{"input_locations": [], "model_output": []}""")
        if len(validation_json) > 0:
            self.resp = json.loads(validation_json)

        # Afterwards, initialize all the attributes.
        self.supporting_information = json.loads(supporting_information)
        self.input = self.get_initial_positions()
        self.output = self.get_model_output()
        self.border = 200

        print(f"SI: {type(supporting_information)}")

    def get_model_output(self) -> Dict:
        return self.resp['model_output']
    
    def get_initial_positions(self) -> Dict:
        return self.resp['input_locations']
    
    def extract_positions(self, troops: Dict) -> List[Dict]:
        positions = []
        for unit in range(int(len(troops)/2)):
            positions.append(troops[unit]['position'])
        return positions
    
    def check_metadata(self) -> None:
        self.p_id=0
        self.p_type=0
        self.p_alliance=0
        self.p_pos=0
        for unit in self.output['coa_id_0']['task_allocation']:
            if (unit['unit_id']==self.input[int(unit['unit_id'])-1]['unit_id']) : self.p_id+=1
            if (unit['unit_type']==self.input[int(unit['unit_id'])-1]['unit_type']) : self.p_type+=1
            if (unit['alliance']==self.input[int(unit['unit_id'])-1]['alliance']) : self.p_alliance+=1
            if (unit['position']==self.input[int(unit['unit_id'])-1]['position']) : self.p_pos+=1
        self.p_id/=len(self.output['coa_id_0']['task_allocation'])
        self.p_type/=len(self.output['coa_id_0']['task_allocation'])
        self.p_alliance/=len(self.output['coa_id_0']['task_allocation'])
        self.p_pos/=len(self.output['coa_id_0']['task_allocation'])
    
    def get_tasks(self) -> List[Dict]:
        tasks=[]
        for unit in self.output['coa_id_0']['task_allocation']:
            tasks.append({"unit_id" : unit['unit_id'],"tasks" : unit['command'].split("; ")})
        return tasks
    
    def check_movement(self) -> None:

        # store all movements into an array
        self.movement_check_arr=[]
        for index, unit_tasks in enumerate(self.get_tasks()):
            is_valid=True

            # select the initial location/position of each unit (including enemies)
            location=self.input[index]['position']

            # evaluate whether each unit's tasks/actions are valid (i.e., does it cross the river?)
            for task in unit_tasks['tasks']:

                # aviation are allowed to cross the river, but we need to check if armor and artillery have
                if not self.input[int(unit_tasks['unit_id'])-1]['unit_type']=="Aviation":
                    try:
                        if "attack_move_unit" in task:

                            # access the parameters for an attack/engage/stand command
                            move_location=task[task.index("(")+1:task.index(")")].split(", ")
                            print(move_location)

                            # 
                            if not self.check_bridge_cross(location, {"x":int(move_location[1]), "y":int(move_location[2])}): is_valid=False
                            location={"x":int(move_location[1]), "y":int(move_location[2])}
                        elif "engage_target_unit" in task:
                            move_location=task[task.index("(")+1:task.index(")")].split(", ")
                            print(move_location)
                            if not self.check_bridge_cross(location, self.input[int(move_location[1])-1]['position']): 
                                is_valid=False
                    except:
                        is_valid=False
            if not is_valid:
                self.movement_check_arr.append(False)
            else:
                self.movement_check_arr.append(True)

    """
    match_bridge_slope() function
    Evaluates whether a ground unit crosses a bridge rather than sinking in the river

    Inputs: start_point   -> Initial ground unit position
            end_point     -> Final ground unit position
            bridge_points -> Set of bridge coordinates
            TOLERANCE     -> Compares slope differences to evaluate bridge crossing match
    
    Output: True/False based on whether a valid bridge crossing is identifiable
    """
    def match_bridge_slope(self, start_point: Tuple[int, int], end_point: Tuple[int, int],
                           bridge_points: Set[Tuple[int, int]], TOLERANCE=0.1) -> bool:
        
        # Extract the x and y-coordinates for every point
        x1, y1 = start_point
        x2, y2 = end_point

        # Iterate through all bridge coordinates
        for bridge_point in bridge_points:

            """
            For each bridge, extract the x and y-coordinates.
            
            If it is possible to create a line that goes through the start_point, bridge_point, and
            the end_point with a (roughly) consistent slope (within a set tolerance), then we can
            consider this bridge crossing to be valid.
            """
            bridge_x, bridge_y = bridge_point
            slope_1, intercept_1 = np.polyfit(np.array[x1, bridge_x], np.array[y1, bridge_y], deg=1)
            slope_2, intercept_2 = np.polyfit(np.array[bridge_x, x2], np.array[bridge_y, y2], deg=1)

            if abs(slope_2 - slope_1) <= TOLERANCE:
                return True
        
        # If there exist no valid bridge crossings, return False
        return False
    
    """
    check_bridge_cross() function
    Input:  Two dicts structured as {"x": INT_1, "y": INT_2}
    Output: Returns True/False based on whether a bridge crossing occurred
    """
    def check_bridge_cross(self, start_location: Dict, end_location: Dict) -> bool:

        # Extract both the start and end coordinates
        x1, y1 = start_location['x'], start_location['y']
        x2, y2 = end_location['x'], end_location['y']

        # Store the bridge coordinates in a set
        # These hardcoded values should be changed later.
        bridge_coordinates = {(100, 50), (100, 150)}

        """
        Check if the armor/artillery crossed into enemy territory.
        If not, it never crossed the bridge.
        """
        crossed_enemy_territory = (x1 < 100 and x2 > 100)

        """
        Case 1: Check if the start location matches the bridge coordinates.
        If so, then we have technically crossed the bridge.
        """
        starts_on_bridge = (x1 == 100 and (y1 == 50 or y1 == 150))

        """
        Case 2: Check if the end location matches the bridge coordinates
        If so, we have crossed the bridge.
        """
        ends_on_bridge = (x2 == 100 and (y2 == 50 or y2 == 150))

        """
        Case 3: Check if we crossed the bridge when moving from the start location
        to the end location. If this case, y1 must equal y2, and the y-coordinate
        must be equal to one of the bridge's y-coordinates.
        """
        crosses_bridge_midway = any(y1 == y2 for y1 in bridge_coordinates)

        """
        Evaluation:
        If the armor/artillery crossed into enemy territory, in order to not to sink in the river, it must have:
        1. started on the bridge
        2. ended on the bridge
        3. crossed the bridge midway

        If none of these three criteria were met, or the armor/artillery never crossed into enemy territory,
        then the armor/artilerry never crossed the bridge (and so we would return False).
        """
        crossed_bridge = not (crossed_enemy_territory and not (starts_on_bridge or ends_on_bridge or crosses_bridge_midway))
        return crossed_bridge
    
    """
    check_enemy_within_range helper function

    Input:  Integer IDs corresponding to the friendly and enemy unit within range: [1, num_units]
    Output: True/False based on whether or not the friendly unit can strike the enemy unit
    """
    def check_enemy_within_range(self, friendly_id: int, enemy_id: int) -> bool:

        # Store the effective range of all artillery units
        effective_ranges = {"Armor": 25, "Artillery": 30, "Aviation": 10}

        # Find the effective range of the friendly unit
        friendly_type = self.supporting_information[friendly_id-1]["unit_type"]
        friendly_distance = effective_ranges[friendly_type]

        # Extract the location of the friendly and enemy unit
        friendly_location = fl = self.supporting_information[friendly_id-1]["position"]
        enemy_location = el = self.supporting_information[enemy_id-1]["position"]

        # Calculate the distance between the friendly and enemy unit
        dx = fl["x"]-el["x"]
        dy = fl["y"]-el["y"]
        distance_to_enemy = (dx*dx + dy*dy)**0.5

        # Determine whether or not the enemy unit is within range
        return (distance_to_enemy <= friendly_distance)
    
    """
    is_valid_stand_location helper function

    Input:  Integer ID corresponding to the friendly unit commanded to stand location
    Output: True/False based on whether or not the friendly unit can stand at this location
    """
    def is_valid_stand_location(self, unit_id: int) -> bool:
        
        # Extract the x and y-coordinates from the unit_id
        coordinates = self.battlefield.supporting_information[unit_id-1]["position"]
        x, y = coordinates["x"], coordinates["y"]

        # Determine whether the unit is within bounds
        return (0 <= x < self.border) and (0 <= y < self.border)
