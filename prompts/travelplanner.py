from typing import Tuple, Dict, Any, List
import pickle
import re

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    ChatMessage,
)

FEWSHOTS = ["""Query: Please devise a travel plan that starts from St. Petersburg and ends in Appleton, taking place for only 1 day on March 19, 2022. This itinerary is for an individual, with a budget allocated at $1,000.
Thought 1: I need to plan a one-day trip from St. Petersburg to Appleton for an individual on March 19, 2022, with a budget of $1,000. I will need to include transportation, meals, attractions, and accommodation within the given budget.
Action 1: CostEnquiry[{"budget": 1000,"people_number": 1,"day": 1,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Subway, Appleton","attraction": "The History Museum at the Castle, Appleton","lunch": "Bliss Bakery, Appleton","dinner": "Mikata Japanese Steakhouse, Appleton","accommodation": "Cozy room + bright apartment + rooftop lounge, Appleton"}]
Observation 1: Cost: Sorry, the cost of your plan is 1389.0 dollars, which exceeds the budget of 1000 dollars. Find either cheaper accomodation or flights to reduce the total cost of the plan. If the cheapest accomodation and flight have been selected, and the cost of the plan still exceeds the budget, start finding cheaper restaurants for breakfast, lunch, and dinner.
Thought 2: I need to find a cheaper accommodation or flight to reduce the total cost of the plan. If the cost still exceeds the budget, I will need to find cheaper restaurants for meals.
Action 2: CostEnquiry[{"budget": 1000,"people_number": 1,"day": 1,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Subway, Appleton","attraction": "The History Museum at the Castle, Appleton","lunch": "Bliss Bakery, Appleton","dinner": "Mikata Japanese Steakhouse, Appleton","accommodation": "Large Rooms with Bar and PullOut Prime Greenpoint, Appleton"}]
Observation 2: Cost: The cost of your plan is 897.0 dollars, which fits within the budget. For your next action, call Finish[Final Plan] to output the plan.
Thought 3: The plan for the one-day trip from St. Petersburg to Appleton on March 19, 2022, for an individual with a budget of $1,000 is as follows:Day 1:- Current City: from St. Petersburg to Appleton- Transportation: Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42- Breakfast: Subway, Appleton- Attraction: The History Museum at the Castle, Appleton- Lunch: Bliss Bakery, Appleton- Dinner: Mikata Japanese Steakhouse, Appleton- Accommodation: Large Rooms with Bar and PullOut Prime Greenpoint, AppletonFinish[Day 1Current City: from St. Petersburg to AppletonTransportation: Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42Breakfast: Subway, AppletonAttraction: The History Museum at the Castle, AppletonLunch: Bliss Bakery, AppletonDinner: Mikata Japanese Steakhouse, AppletonAccommodation: Large Rooms with Bar and PullOut Prime Greenpoint, Appleton]
Action 3: Finish[Day 1Current City: from St. Petersburg to AppletonTransportation: Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42Breakfast: Subway, AppletonAttraction: The History Museum at the Castle, AppletonLunch: Bliss Bakery, AppletonDinner: Mikata Japanese Steakhouse, AppletonAccommodation: Large Rooms with Bar and PullOut Prime Greenpoint, Appleton]
Observation 3: The plan is finished.""",
"""Query: Please devise a travel plan that starts from St. Petersburg and ends in Appleton, taking place across 2 days from March 19th to March 20th, 2022. This itinerary is for an individual, with a budget allocated at $2,000.
Thought 1: I need to plan a two-day trip from St. Petersburg to Appleton for an individual with a budget of $2,000. I will need to include transportation, accommodation, attractions, and meals within the allocated budget. I will start by looking for a flight from St. Petersburg to Appleton on March 19th and a return flight on March 21st. I will also need to find affordable accommodation, attractions to visit, and restaurants for meals within the budget.
Action 1: CostEnquiry[{"budget": 2000,"people_number": 1,"day": 1,"total_days": 2,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Subway, Appleton","attraction": "The History Museum at the Castle, Appleton","lunch": "Bliss Bakery, Appleton","dinner": "Mikata Japanese Steakhouse, Appleton","accommodation": "Cozy room + bright apartment + rooftop lounge, Appleton"}]
Observation 1: Cost: Sorry, the cost of your plan is 1389.0 dollars, which exceeds the targeted sub-budget of 1250 dollars. Find either cheaper accomodation or flights to reduce the total cost of the plan. If the cheapest accomodation and flight have been selected, and the cost of the plan still exceeds the budget, start finding cheaper restaurants. When creating your next plan, only change one attribute from your current plan. For instance, if you will change the accomodation for the next plan, do not change anything else, such as the breakfast or the flight.
Thought 2: I need to find a cheaper accommodation option to reduce the cost of the plan. I will look for a more affordable place to stay in Appleton while keeping the same flight, attractions, and meal options. This should help me stay within the budget while still providing a comfortable and enjoyable trip for the individual traveler.
Action 2: CostEnquiry[{"budget": 2000,"people_number": 1,"day": 1,"total_days": 2,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Subway, Appleton","attraction": "The History Museum at the Castle, Appleton","lunch": "Bliss Bakery, Appleton","dinner": "Mikata Japanese Steakhouse, Appleton","accommodation": "Large Rooms with Bar and PullOut Prime Greenpoint, Appleton"}]
Observation 2: Cost: The cost of your subplan so far is 897.0 dollars, which tentatively fits within the budget. Now, plan restaurants, accomodations, and transportation (if needed) for the next day, using CostEnquiry[subplan] to ensure that subsequent days within the plan.
Thought 3: The cost of the plan with the new, more affordable accommodation option is within the budget. Now, I need to plan the activities, meals, and accommodation for the second day of the trip. I will ensure that the total cost of the two-day plan remains within the allocated budget of $2,000.
Action 3: CostEnquiry[{"budget": 2000,"people_number": 1,"day": 2,"total_days": 2,"current_city": "Appleton","transportation": "Flight Number: F3578689, from Appleton to St. Petersburg, Departure Time: 07:19, Arrival Time: 11:01","breakfast": "Bliss Bakery, Appleton","attraction": "Hearthstone Historic House Museum, Appleton","lunch": "Colony, Appleton","dinner": "Swati Snacks, Appleton","accommodation": "Large Rooms with Bar and PullOut Prime Greenpoint, Appleton"}]
Observation 3: Cost: The cost of your plan is 747.0 dollars, which fits within the budget. For your next action, call Finish[Final Plan] to output the plan.
Thought 4: The cost of the plan for the second day is within the budget. I will now finalize the travel plan for the two-day trip from St. Petersburg to Appleton and back, ensuring that all details are included in the final plan.
Action 4: Finish[Day 1Current City: from St. Petersburg to AppletonTransportation: Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42Breakfast: Subway, AppletonAttraction: The History Museum at the Castle, AppletonLunch: Bliss Bakery, AppletonDinner: Mikata Japanese Steakhouse, AppletonAccommodation: Large Rooms with Bar and PullOut Prime Greenpoint, AppletonDay 2:Current City: AppletonTransportation: Flight Number: F3578689, from Appleton to St. Petersburg, Departure Time: 07:19, Arrival Time: 11:01Breakfast: Bliss Bakery, AppletonAttraction: Hearthstone Historic House Museum, AppletonLunch: Colony, AppletonDinner: Swati Snacks, AppletonAccommodation: Large Rooms with Bar and PullOut Prime Greenpoint, Appleton]
Observation 4: The plan is finished."""]

REFLECTION_FEWSHOTS = ["""Query: Please devise a travel plan that starts from St. Petersburg and heads to Appleton, taking place across 3 days from March 19th to March 21st, 2022. This itinerary is for an individual, with a budget allocated at $1,200.
Thought 1: I need to plan a 3-day trip from St. Petersburg to Appleton for an individual with a budget of $1,200. I will need to include transportation, accommodation, and meals for each day of the trip. I should also consider visiting various attractions and dining at different restaurants to ensure a diverse experience for the traveler.
Action 1: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Subway, St. Petersburg","attraction": "The History Museum at the Castle, Appleton","lunch": "Mikata Japanese Steakhouse, Appleton","dinner": "Brick 29, Appleton","accommodation": "Cozy room + bright apartment + rooftop lounge, Appleton"}]
Observation 1: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check. 
Thought 2: The breakfast information for the first day is not valid. I need to find a valid breakfast option in St. Petersburg for the traveler before proceeding with the cost inquiry.
Action 2: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Cafe Vienna, St. Petersburg","attraction": "The History Museum at the Castle, Appleton","lunch": "Mikata Japanese Steakhouse, Appleton","dinner": "Brick 29, Appleton","accommodation": "Cozy room + bright apartment + rooftop lounge, Appleton"}]
Observation 2: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check. 
Thought 3: I need to find a valid breakfast option in St. Petersburg for the traveler before proceeding with the cost inquiry. I will look for a different breakfast option in St. Petersburg and then make another cost inquiry.
Action 3: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from St. Petersburg to Appleton","transportation": "Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42","breakfast": "Cafe Vienna, St. Petersburg","attraction": "The History Museum at the Castle, Appleton","lunch": "Mikata Japanese Steakhouse, Appleton","dinner": "Brick 29, Appleton","accommodation": "Cozy room + bright apartment + rooftop lounge, Appleton"}]
Observation 3: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check.

You have attempted to give a sub plan before and failed. The following reflection(s) give a suggestion to avoid failing to answer the query in the same way you did previously. Use them to improve your strategy of correctly planning.
Reflections:
- The possible reason for failure is that the breakfast information provided for the traveler in St. Petersburg was not valid, leading to the inability to calculate the cost. To mitigate this failure, a new, concise, high-level plan could involve ensuring that all meal and accommodation information is valid and accurate before proceeding with the cost inquiry. This could involve double-checking the details of the breakfast, lunch, dinner, and accommodation options to ensure they are valid and match the available options in the given locations.

Thought 4: I'm unable to find a valid breakfast option in St. Petersburg for the traveler. I will need to revise my plan to ensure that all meal and accommodation information is valid and accurate before proceeding with the cost inquiry. This could involve double-checking the details of the breakfast, lunch, dinner, and accommodation options to ensure they are valid and match the available options in the given locations.
Action 4: Finish[Day 1Current City: from St. Petersburg to AppletonTransportation: Flight Number: F3574992, from St. Petersburg to Appleton, Departure Time: 22:51, Arrival Time: 00:42Breakfast: Cafe Vienna, St. PetersburgAttraction: The History Museum at the Castle, AppletonLunch: Mikata Japanese Steakhouse, AppletonDinner: Brick 29, AppletonAccommodation: Cozy room + bright apartment + rooftop lounge, AppletonDay 2:Current City: AppletonTransportation: -Breakfast: Bliss Bakery, AppletonAttraction: Hearthstone Historic House Museum, Appleton; Atlas Science Center Center, AppletonLunch: Parantha Gurus, AppletonDinner: The Millionaire Express, AppletonAccommodation: Large Rooms with Bar and PullOut Prime Greenpoint, AppletonDay 3:Current City: from Appleton to St. PetersburgTransportation: Flight Number: F3578689, from Appleton to St. Petersburg, Departure Time: 07:19, Arrival Time: 11:01Breakfast: Mathew's Cafe, AppletonAttraction: Butterfly Garden of Wisconsin, AppletonLunch: Asian Curry, AppletonDinner: Fire n Ice, AppletonAccommodation: -]
Observation 4: The plan is finished.""",
"""Query: Could you assist in creating a week-long travel plan for one person, starting from Indianapolis and venturing through 3 cities in North Carolina from March 7th to March 13th, 2022? The planned budget for the trip is $6,500.
Thought 1: I need to plan a week-long trip for one person, starting from Indianapolis and traveling through three cities in North Carolina from March 7th to March 13th, 2022. The budget for the trip is $6,500. I'll need to consider transportation, accommodations, attractions, and meals in each city while staying within the budget.
Action 1: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from Indianapolis to Charlotte","transportation": "Flight Number: F3659001, from Indianapolis to Charlotte, Departure Time: 08:07, Arrival Time: 09:50","breakfast": "On the go Cafe, Indianapolis","attraction": "Freedom Park, Charlotte","lunch": "Nagaland's Kitchen, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}]
Observation 1: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check. 
Thought 2: I need to revise the breakfast information for the first day of the trip from Indianapolis to Charlotte. The current plan includes "On the go Cafe, Indianapolis" as the breakfast location, which is not a valid option. I will need to find a suitable breakfast location in Indianapolis and then resubmit the cost inquiry for the first day of the trip.
Action 2: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from Indianapolis to Charlotte","transportation": "Flight Number: F3659001, from Indianapolis to Charlotte, Departure Time: 08:07, Arrival Time: 09:50","breakfast": "Cafe Patachou, Indianapolis","attraction": "Freedom Park, Charlotte","lunch": "Nagaland's Kitchen, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}]
Observation 2: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check. 
Thought 3: It seems that there may be an issue with the breakfast information for the first day of the trip from Indianapolis to Charlotte. I will need to verify the availability and validity of the breakfast options in Indianapolis before resubmitting the cost inquiry for the first day of the trip.
Action 3: CostEnquiry[{"people_number": 1,"day": 1,"current_city": "from Indianapolis to Charlotte","transportation": "Flight Number: F3659001, from Indianapolis to Charlotte, Departure Time: 08:07, Arrival Time: 09:50","breakfast": "Cafe Patachou, Indianapolis","attraction": "Freedom Park, Charlotte","lunch": "Nagaland's Kitchen, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}]
Observation 3: Cost: Sorry, the cost of your plan is not available because of the following reasons:1. The breakfase information is not valid, please check.

You have attempted to give a sub plan before and failed. The following reflection(s) give a suggestion to avoid failing to answer the query in the same way you did previously. Use them to improve your strategy of correctly planning.
Reflections:
- The possible reason for failure in creating a travel plan was the invalid breakfast information for the first day of the trip from Indianapolis to Charlotte. The breakfast location "On the go Cafe, Indianapolis" was not a valid option. To mitigate this failure, a new, concise, high-level plan could involve verifying the availability and validity of breakfast options in Indianapolis before resubmitting the cost inquiry for the first day of the trip. Additionally, ensuring that all meal and attraction information is accurate and valid before submitting the cost inquiry for each day of the trip will be crucial to the success of the plan.
 
Thought 4: It seems that there may be an issue with the breakfast information for the first day of the trip from Indianapolis to Charlotte. I will need to verify the availability and validity of the breakfast options in Indianapolis before resubmitting the cost inquiry for the first day of the trip.
Action 4: Finish[Day 1:Current City: from Indianapolis to CharlotteTransportation: Flight Number: F3659001, from Indianapolis to Charlotte, Departure Time: 08:07, Arrival Time: 09:50Breakfast: Cafe Patachou, IndianapolisAttraction: Freedom Park, CharlotteLunch: Nagaland's Kitchen, CharlotteDinner: Bombay Vada Pav, CharlotteAccommodation: Affordable Spacious Refurbished Room in Bushwick!, CharlotteDay 2:Current City: CharlotteTransportation: -Breakfast: Olive Tree Cafe, CharlotteAttraction: The Mint Museum, Charlotte; Romare Bearden Park, CharlotteLunch: Birbal Ji Dhaba, CharlotteDinner: Pind Balluchi, CharlotteAccommodation: Affordable Spacious Refurbished Room in Bushwick!, CharlotteDay 3:Current City: from Charlotte to WilmingtonTransportation: Flight Number: F3666307, from Charlotte to Wilmington, Departure Time: 12:54, Arrival Time: 13:46Breakfast: Subway, CharlotteAttraction: Bellamy Mansion Museum, WilmingtonLunch: Azteca, WilmingtonDinner: Bandit Burrito, WilmingtonAccommodation: -]
Observation 4: The plan is finished."""]

SYSTEM_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Solve this task by alternating between Thought, Action, and Observation steps. The 'Thought' phase involves reasoning about the current situation. The 'Action' phase can be of two types:
(1) CostEnquiry[Sub Plan]: This function calculates the cost of a detailed sub plan, which you need to input the people number and plan in JSON format. The sub plan should encompass a complete one-day plan. An example will be provided for reference.
(2) Finish[Final Plan]: Use this function to indicate the completion of the task. You must submit a final, complete plan as an argument.
***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
You can call CostEnquiry like CostEnquiry[{{"budget": 30200,"people_number": 7,"day": 1,"total_days": 3,"current_city": "from Ithaca to Charlotte","transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46","breakfast": "Nagaland's Kitchen, Charlotte","attraction": "The Charlotte Museum of History, Charlotte","lunch": "Cafe Maple Street, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}}]
You can call Finish like Finish[Day: 1
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -]
***** Example Ends *****

{reflections}

If the cost of the plan exceeds the allocated budget, prioritize finding a cheaper accomodation or flight.
If the cheapest accomodation and flight have been selected, and the cost of the plan still exceeds the budget, start finding cheaper restaurants for breakfast, lunch, and dinner.

You must use Finish to indict you have finished the task. And each action only calls one function once.
Given information: {text}
Query: {query}{scratchpad} """

human_instruction_template = """{instruction}You may take maximum of {max_steps} steps.
Here are some examples:"""

HUMAN_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_template)

human_instruction_reflection_template = """Here are some examples:"""
HUMAN_REFLECTION_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_reflection_template)

SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION = """You will be given two previous task trials in which you were given access to travel itinerary information and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."""
SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION = """You will be given successful tasks trials in which you were given access travel itinerary information and a question to answer."""
SYSTEM_REFLECTION_INSTRUCTION = """You will be given a previous reasoning trial in which you were given access to travel itinerary information and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences."""

def LLM_PARSER(llm_output, step: int, ai_message: bool) -> Tuple[ChatMessage, str, Dict[str, Any]]:

    # Searches for the keyword "action:"
    pattern = r'(?i)action\s*(?:\d+|)\s*(?::|)\s*'

    # Searches for an action enclosed by an opening square bracket, and an optional closing bracket
    action_pattern = r'(?i)\w+\[[^\]]+(?:\]|)'

    # Determine whether the llm_output features an "action:" string
    match = re.match(pattern, llm_output)

    # If so, prepare to return a formatted action string
    if match:
        action = llm_output[match.end():]
        content = f"Action {step}: {action}"

        # Handle bracket cases by triggering an invalid action
        if len(re.findall(action_pattern, action)) > 1:
            return (
                AIMessage(content=content) if ai_message else HumanMessage(content=content),
                'action',
                {'action': ''} # triggers invalid action
            )
        
        # Handle non-bracket cases
        else:
            return (
                AIMessage(content=content) if ai_message else HumanMessage(content=content),
                'action',
                {'action': action}
            )
        
    # If the "action: " substring wasn't found, look for square brackets
    actions = re.findall(action_pattern, llm_output)
    if len(actions) == 1:
        action = actions[0]
        if action[-1] != ']':
            action += ']'
        content = f"Action {step}: {action}"
        return (
            AIMessage(content=content) if ai_message else HumanMessage(content=content),
            'action',
            {'action': action}
        )
    
    # If a string has both "action: " and square brackets, it's an invalid string
    if len(actions) > 1:
        content = re.sub(r"(?i)action\s*(?:\d*|)\s*(?::|)", "", llm_output)
        return (
            AIMessage(content=f"Action {step}: {content}"),
            'action',
            {'action': ''} # triggers invalid action
        )

    # everthing else will be assumed to be a inner thought
    thought_pattern = r'(?i)thought\s*(?:\d+|)\s*(?::|)\s*(.*)'
    match = re.match(thought_pattern, llm_output)
    if match:
        # Extract the thought word and content
        thought_word = match.group(1)
        content = f"Thought {step}: {thought_word.rstrip(':')}"
    else:
        content = f"Thought {step}: {llm_output.rstrip(':')}"
    return (
        AIMessage(content=content) if ai_message else HumanMessage(content=content),
        'thought',
        {}
    )

def OBSERVATION_FORMATTER(observation: str, step: int, *args, **kwargs) -> Tuple[ChatMessage, str]:
    return HumanMessage(content=f"Observation {step}: " + observation.rstrip(':')), 'append'

def STEP_IDENTIFIER(line: str) -> str:
    line = line.strip()
    pattern = re.compile(r'^(?i)action(?:\s+(\d+))?:')
    match = pattern.match(line)
    if match:
        return 'action'
    pattern = re.compile(r'^(?i)observation(?:\s+(\d+))?:')
    match = pattern.match(line)
    if match:
        return 'observation'
    return 'thought'

def CYCLER(lines: str) -> List[str]:
    new_lines = []
    scratch_pad = ''
    for line in lines.split('\n'):

        # line is action
        pattern = re.compile(r'^(?i)action(?:\s+(\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue

        # line is thought
        pattern = re.compile(r'^(?i)thought(?:\s+(\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue

        # step is observation
        scratch_pad += line + '\n'

    # the rest of the scratch pad
    if scratch_pad != '':
        new_lines.append(scratch_pad.strip())
    return new_lines

REFLECTION_PREFIX = '\nReflection:'
def PREVIOUS_TRIALS_FORMATTER(reflections: List[str], include_prefix: bool = True) -> str:
    if reflections == []:
        return ''
    if include_prefix:
        memory_prefix = "You have attempted to solve the task before but failed. The following reflection(s) give a plan to avoid failing the task in the same way you did previously. Use them to improve your strategy of solving the task successfully."
    else:
        memory_prefix = ''
    memory_prefix += '\nReflections:'
    for reflection in reflections:
        memory_prefix += f"\n- {reflection.strip()}"
    return memory_prefix

def STEP_STRIPPER(step: str, step_type: str):
    if step_type == 'observation':
        return re.sub(r'^(?i)observation(?:\s+(\d+))?:', 'Observation:', step)
    if step_type == 'action':
        return re.sub(r'^(?i)action(?:\s+(\d+))?:', 'Action:', step)
    if step_type == 'thought':
        return re.sub(r'^(?i)thought(?:\s+(\d+))?:', 'Thought:', step)
    return step