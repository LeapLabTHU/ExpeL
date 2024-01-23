from typing import Tuple, Dict, Any, List
import re

from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    ChatMessage,
)

FEWSHOTS = ["""Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: think[All options are properly selected, I can now buy it.]
Observation: OK.            

Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 1.0""",
"""Instruction:
i am looking for long sleeve men t-shirt.and please also choose the black one, and price lower than 30.00 dollars
[Search]
Action: search[shirt man]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B09NNMV9LN]
JSPOYOU Mens Short Sleeve Crewneck 3D Graphic Tunic Shirts Big & Tall Tie Dye Summer Top Basic Designed Classic Cotton Shirt
$3.99 to $10.99
[B09Q67H373]
JSPOYOU 2022 Newly T-Shirt for Mens Funny 3D Graphics Pattern Crewneck Short Sleeve Tees Big and Tall Summer Casual Comfy Top
$1.99 to $8.99
[B00I3O3BSS]
Blue by Betsey Johnson Women's SB Ever Ballet Flat
$89.99

Action: think[It seems the first page doesn't mention long sleeves. I should check the next page]
Observation: OK.

Action: click[Next >]
Observation:
[Back to Search]
Page 2 (Total results: 50)
[< Prev]
[Next >]
[B09Q8RD8YN]
Bungo Stray Anime Dogs Anime Character, Long Sleeve, Sweatshirt, Hoodie, T shirt
$19.99
[B09KLQLLT2]
Long Sleeve Superhero T Shirt Tank Top Mens Compression Shirt Men Workout Fitness Gym Shirt
$19.99
[B09R9YCM6R]
XXBR Summer T-shirts for Mens, Soldier Short Sleeve 3D Street Vintage Printed Shirt Slim Fit Muscle Casual Tee Tops
$8.98 to $11.99

Action: think[It looks like we have long sleeves tshirt but not black. I will not bother looking at the others pages and will focus on rephrasing the query. I first need to click 'Back to Search' button.]
Observation: OK.

Action: click[Back to Search]
Observation:
i am looking for long sleeve men t-shirt and please also choose the black one, and price lower than 40.00 dollars
[Search]

Action: search[long sleeves t-shirt man black]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B09QGK5XHZ]
WENKOMG1 Men's Long Sleeve Undershirt with Mask Turtleneck Hooded T-Shirt Solid Color Workout Tops Zipper Side Slit Shirts Slim Fit Sweatshirt Spring/Summer Tee Shirts(Gray,)
$8.39
[B09QQP3356]
HAUKLIE Men's Sports Waffle Ribbed Polo Shirts Summer Long Sleeve Cotton Muscle Quarter-Zip Henley T-Shirt Tunics Tops
$10.99
[B09Q67H373]
JSPOYOU 2022 Newly T-Shirt for Mens Funny 3D Graphics Pattern Crewneck Short Sleeve Tees Big and Tall Summer Casual Comfy Top
$1.99 to $8.99

Action: think[B09QQP3356 seems to be a good match now. It is under 40.00 dollars and has long sleeves. I will check it first.]
Observation: OK.

Action: click[B09QQP3356]
Observation:
[Back to Search]
[< Prev]
color [black][gray][light blue]
size [small][medium][large][x-large][xx-large][3x-large]
HAUKLIE Men's Sports Waffle Ribbed Polo Shirts Summer Long Sleeve Cotton Muscle Quarter-Zip Henley T-Shirt Tunics Tops
Price: $10.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[There is 2 options, color and size. I need to select them one by one according to the instruction. First, I will choose the black color.]
Observation: OK.

Action: click[black]
Observation: You have clicked black.

Action: think[The item has also a size option, I must select one. Since no information is given about the size, I will the first one.]
Observation: OK.

Action: click[small]
Observation: You have clicked small.

Action: think[All options are proprely selected, I can now buy it.]
Observation: OK.

Action: click[Buy Now]
Observation: Your score (min 0.0, max 1.0): 1.0"""
]

NEW_REFLECTION_FEWSHOTS = [
"""Previous Trial:
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has 2 options, flavor and size. I need to select them one by one according to the instruction. First, I will choose the variety pack.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: think[Then, since there is no required size, I will choose the first one.]
Observation: OK.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]
Observation: Wrong item! TASK FAILED

Next plan: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.""",
"""Previous Trial:
Instruction:
i need gluten free vegetarian smoked peppered bacon - 4 ounce (pack of 2), and price lower than 40.00 dollars
[Search]

Action: search[gluten free vegetarian smoked peppered bacon 4 ounce pack of 2]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: think[The first two items are not what I want. The third item is not what I want, but it is the closest one.]
Observation: OK.

Action: think[The third item is not what I want, but it is the closest one.]
Observation: OK.

Action: click[B07GJTKYJQ]
Observation:
[Back to Search]
[< Prev]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
Price: $42.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The price is too high and it is not what I want.]
Observation: OK.

Action: click[< Prev]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: think[The first two items are not what I want.]
Observation: OK.

Action: think[I need to refine my search.]
Observation: OK.

Action: click[< Back to Search]
Observation: Invalid action!

Action: click[Back to Search]
Observation: Ran out of steps! TASK FAILED

Next plan: In this attempt, I was unsuccessful. The initial search results were not the products that I was looking for. Next time, I will do search["vegetarian bacon"] for a broader search. I will check that the new results will fulfill the gluten free and 4 ounce pack of 2 constraints. I will continue to refine my searches and think less so that I can find more products."""
]

ACT_REFLECTION_FEWSHOTS = [
"""Previous Trial:
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]
Observation: Wrong item! TASK FAILED

Next plan: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.""",
"""Previous Trial:
Instruction:
i need gluten free vegetarian smoked peppered bacon - 4 ounce (pack of 2), and price lower than 40.00 dollars
[Search]

Action: search[gluten free vegetarian smoked peppered bacon 4 ounce pack of 2]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: click[B07GJTKYJQ]
Observation:
[Back to Search]
[< Prev]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
Price: $42.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: click[< Prev]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: click[< Back to Search]
Observation: Invalid action!

Action: click[Back to Search]
Observation: Ran out of steps! TASK FAILED

Next plan: In this attempt, I was unsuccessful. The initial search results were not the products that I was looking for. Next time, I will do search["vegetarian bacon"] for a broader search. I will check that the new results will fulfill the gluten free and 4 ounce pack of 2 constraints. I will continue to refine my searches and think less so that I can find more products."""
]

REFLECTION_FEWSHOTS = [
"""Previous Trial:
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]
Observation: Wrong item! TASK FAILED

Next plan: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.""",
"""Previous Trial:
Instruction:
i need gluten free vegetarian smoked peppered bacon - 4 ounce (pack of 2), and price lower than 40.00 dollars
[Search]

Action: search[gluten free vegetarian smoked peppered bacon 4 ounce pack of 2]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: think[The first two items are not what I want. The third item is not what I want, but it is the closest one.]
Observation: OK.

Action: think[The third item is not what I want, but it is the closest one.]
Observation: OK.

Action: click[B07GJTKYJQ]
Observation:
[Back to Search]
[< Prev]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
Price: $42.99
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The price is too high and it is not what I want.]
Observation: OK.

Action: click[< Prev]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B06Y96MXJV]
Smoked Bacon Sea Salt 3-Pack: Smoked Bacon Chipotle, Smoked Bacon and Onion, Smoked Peppered Bacon - All-Natural Sea Salts Slowly Smoked Over Alderwood - No Gluten, No MSG, Non-GMO (12 total oz.)
$29.99
[B06Y96N1KG]
Spicy Hot Pepper Sea Salt 3-Pack: Ghost Pepper, Jalapeno, Habanero - All Natural, Delicious Blends of Peppers & Sea Salt - Gluten-Free, Kosher, No MSG, Non-GMO (12 total oz.)
$29.99
[B07GJTKYJQ]
Louisville Vegan Jerky - 5 Flavor Variety Pack, Non-GMO Soy Protein, Gluten-Free (Black Pepper, Buffalo Dill, Pepperoni, Maple Bacon, and Carolina BBQ, 3 Ounces)
$42.99

Action: think[The first two items are not what I want.]
Observation: OK.

Action: think[I need to refine my search.]
Observation: OK.

Action: click[< Back to Search]
Observation: Invalid action!

Action: click[Back to Search]
Observation: Ran out of steps! TASK FAILED

Next plan: In this attempt, I was unsuccessful. The initial search results were not the products that I was looking for. Next time, I will do search["vegetarian bacon"] for a broader search. I will check that the new results will fulfill the gluten free and 4 ounce pack of 2 constraints. I will continue to refine my searches and think less so that I can find more products."""
]

SYSTEM_INSTRUCTION = """Follow the syntax of the examples closely when taking actions."""

human_instruction_template = """{instruction}You may take maximum of {max_steps} steps.
Here are two examples:"""

HUMAN_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_template)

human_instruction_reflection_template = """Here are two examples:"""
HUMAN_REFLECTION_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_reflection_template)

SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION = """You will be given two previous task trials in which you could interact with an online store website in order to buy the requested item: one successful and one unsuccessful trial. You failed the trial either because you reached the maximum allowed number of steps or you bought a wrong item."""
SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION = """You will be given successful task trials in which you could interact with an online store website in order to buy the requested items."""

def LLM_PARSER(llm_output, step: int, ai_message: bool) -> Tuple[ChatMessage, str, Dict[str, Any]]:
    pattern = r'(?i)action\s*\d*\s*:\s*'
    formatted_output = re.sub(pattern, '', llm_output)
    if '[' not in formatted_output:
        formatted_output = f'think[{formatted_output.lstrip("Observation:").strip()}]'
    elif '[' in formatted_output and not formatted_output.endswith(']'):
        formatted_output += ']'
    return (
        AIMessage(content=f'Action: {formatted_output}') if ai_message else HumanMessage(content=f'Action: {formatted_output}'),
        'action',
        {'action': formatted_output}
    )

def OBSERVATION_FORMATTER(observation: str, *args, **kwargs) -> Tuple[ChatMessage, str]:
    return HumanMessage(content=f"Observation: " + observation + '\n'), 'append'

def STEP_IDENTIFIER(line: str) -> str:
    line = line.strip()
    if line.lower().startswith('action: think['):
        return 'thought'
    if line.lower().startswith('action'):
        return 'action'
    return 'observation'

# Should return [action1, observation1, action2, observation2, ...]
def CYCLER(lines: str) -> List[str]:
    listed_lines = []
    listed_steps = lines.split('\n\n')
    for step in listed_steps:
        # assuming action only takes one line
        listed_lines.extend(step.split('\n', 1))
    return listed_lines

REFLECTION_PREFIX = '\nNext plan:'
def PREVIOUS_TRIALS_FORMATTER(reflections: List[str], *args, **kwargs) -> str:
    if reflections == []:
        return ''
    memory_prefix = 'Your memory for the task below:'
    for i, reflection in enumerate(reflections):
        memory_prefix += f"\nTrial {i}:\n{reflection.strip()}"
    return memory_prefix

def STEP_STRIPPER(step: str, step_type: str):
    if step_type != 'task':
        return step
    return step.strip('Instruction:\n').strip()
