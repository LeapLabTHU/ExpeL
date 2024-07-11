from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from evaluation.hard_constraint import extract_from_to,get_valid_name_city
import math

from envs.base import BaseEnv

class TravelPlannerEnv(BaseEnv):
    def __init__(self):
        
        self.flight = Flights()
        self.accommodation = Accommodations()
        self.restaurants = Restaurants()
        self.googleDistanceMatrix = GoogleDistanceMatrix()
        self.attractions = Attractions()
    
    def run(self, tested_data):

        total_cost = 0
        unit = tested_data
        people_number = tested_data['people_number']
        returned_info = []

        if 'transportation' in unit and unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if 'flight number' in value.lower():
                    try:
                        res = self.flight.data[self.flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                        if len(res) > 0:
                            total_cost += res['Price'].values[0] * people_number
                        else:
                            returned_info.append('The filght information is not valid')
                    except:
                        returned_info.append('The filght information is not valid')

            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    if 'self-driving' in value.lower():
                        # print(org_city,dest_city)
                        cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                        if cost == None:
                            returned_info.append('The transporation information is not valid, please check.')
                        else:
                            total_cost += cost * math.ceil(people_number * 1.0 / 5)
                    else:
                        cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                        if cost == None:
                            returned_info.append('The transporation information is not valid, please check.')
                        else:
                            total_cost += cost * math.ceil(people_number * 1.0 / 4)
                except:
                    returned_info.append('The transporation information is not valid, please check. You have to make sure there are two cities (from A to B) in your transportation plan.')

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The breakfase information is not valid, please check.')

        if 'lunch' in unit and  unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The lunch information is not valid, please check.')

        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append('The dinner information is not valid, please check.')

        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name != '-' and city != '-':
                res = self.accommodation.data[(self.accommodation.data['NAME'] == name) & (self.accommodation.data['city'] == city)]
                if len(res) > 0:
                    total_cost += res['price'].values[0] * math.ceil(people_number * 1.0 / res['maximum occupancy'].values[0])
                else:
                    returned_info.append('The accommodation information is not valid, please check.')
        
        if len(returned_info) == 0:

            """
            Confirm that the total_cost of the plan fits within the budget.
            """
            budget = tested_data['budget']
            if total_cost <= budget:
                return f"The cost of your plan is {total_cost} dollars, which fits within the budget. For your next action, call Finish[Final Plan] to output the plan."
            else:
                return f"Sorry, the cost of your plan is {total_cost} dollars, which exceeds the budget of {budget} dollars. Find either cheaper accomodation or flights to reduce the total cost of the plan. If the cheapest accomodation and flight have been selected, and the cost of the plan still exceeds the budget, start finding cheaper restaurants for breakfast, lunch, and dinner."

        else:
            message = "Sorry, the cost of your plan is not available because of the following reasons:"
            for idx, info in enumerate(returned_info):
                message += str(idx + 1) + ". " + info + " " + '\t'
            return message





class TravelPlannerReflectEnv(TravelPlannerEnv):
    def __init__(self):
        super().__init__()
        self.is_terminated = False
        self.max_retry_step = 3
        self.retry_step = 0

        """
        New Variable: self.cheapest_plan

        Stores the string representation of the cheapest plan. If the current plan was more expensive
        than the cheapest plan generated in a previous iteration, remind the LLM of this plan.
        """
        self.cheapest_plan = ""

        """
        New Variable: self.cheapest_plan_cost

        Store the cost of the cheapest plan. If the current plan was more expensive than the
        cheapest plan, alert the LLM to revert to its cheapest plan, so it increases the likelihood
        that it finds a plan that falls under its allocated budget.
        """
        self.cheapest_plan_cost = (1<<30)

    def reset(self):
        self.is_terminated = False
        self.retry_step = 0
        self.cheapest_plan = ""
        self.cheapest_plan_cost = (1<<30)

    def run(self, tested_data):
        total_cost = 0
        unit = tested_data
        people_number = tested_data['people_number']
        returned_info = []

        if 'transportation' in unit and unit['transportation'] and  unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city == None or dest_city == None:
                org_city, dest_city = extract_from_to(unit['current_city'])
                
            if org_city == None:
                returned_info.append('The origin city does not exist. Make sure to confirm that it exists.')
            if dest_city == None:
                returned_info.append('The destination city does not exist. Make sure to confirm that it exists.')

            else:    
                if 'flight number' in value.lower():
                        try:
                            res = self.flight.data[self.flight.data['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                            if len(res) > 0:
                                total_cost += res['Price'].values[0] * people_number
                            else:
                                returned_info.append('This flight does not exist. Select a valid flight for your next iteration.')
                        except:
                            returned_info.append('This flight does not exist. Select a valid flight for your next plan.')

                elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                        if 'self-driving' in value.lower():
                            cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'self-driving')['cost']
                            if cost == None:
                                returned_info.append('Self-driving is not an option. Select either a flight or taxi for your next plan.')
                            else:
                                total_cost += cost * math.ceil(people_number * 1.0 / 5)
                        else:
                            cost = self.googleDistanceMatrix.run_for_evaluation(org_city,dest_city,'taxi')['cost']
                            if cost == None:
                                returned_info.append('Riding a taxi is not an option. Select either a flight or self-drive for your next plan.')
                            else:
                                total_cost += cost * math.ceil(people_number * 1.0 / 4)

        if 'breakfast' in unit and unit['breakfast'] and unit['breakfast'] != '-':
            name, city = get_valid_name_city(unit['breakfast'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append(f'This breakfast restaurant does not exist in {city}. Select a restaurant that exists in {city} for your next plan.')

        if 'lunch' in unit and  unit['lunch'] and unit['lunch'] != '-':
            name, city = get_valid_name_city(unit['lunch'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append(f'This lunch restaurant does not exist in {city}. Select a restaurant that exists in {city} for your next plan.')

        if 'dinner' in unit and unit['dinner'] and unit['dinner'] != '-':
            name, city = get_valid_name_city(unit['dinner'])
            if name != '-' and city != '-':
                res = self.restaurants.data[(self.restaurants.data['Name'] == name) & (self.restaurants.data['City'] == city)]
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * people_number
                else:
                    returned_info.append(f'This dinner restaurant does not exist in {city}. Select a restaurant that exists in {city} for your next plan.')

        """
        CostEnquiry checks that the accomodation exists in the current city, but does not check other constraints, such as minimum nights or pets allowed.
        """
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if name != '-' and city != '-':
                res = self.accommodation.data[(self.accommodation.data['NAME'] == name) & (self.accommodation.data['city'] == city)]
                if len(res) > 0:
                    total_cost += res['price'].values[0] * math.ceil(people_number * 1.0 / res['maximum occupancy'].values[0])
                else:
                    returned_info.append(f'This accomodation does not exist in {city}. Select an accp,pdatopm that exists in {city} for your next plan.')
        
        if len(returned_info) == 0:
            self.retry_step = 0
            self.is_terminated = False

            # Confirm that the total_cost of the plan fits within the budget.
            budget = tested_data['budget']
            num_days = unit['total_days']

            """
            For each day, try to aim for a "targeted" budget. This goes with the philosophy that
            the first and last day will inherently be more expensive than the intermediate days,
            as travel can take the bulk of expenses.
            """
            targeted_budget = int(budget * 1.25 / num_days)

            if total_cost <= targeted_budget:

                # Reset plans for the next day
                self.cheapest_plan = ""
                self.cheapest_plan_cost = (1<<30)

                # Determine if we need to find subplans for successive days
                if unit['total_days'] == unit['day']:
                    return f"The cost of your plan is {total_cost} dollars, which fits within the budget. " \
                            "For your next action, call Finish[Final Plan] to output the plan."
                else:
                    return f"The cost of your subplan so far is {total_cost} dollars, which tentatively fits within the budget. " \
                            "Now, plan restaurants, accomodations, and transportation (if needed) for the next day, " \
                            "using CostEnquiry[subplan] to ensure that subsequent days within the plan."
        
            
            # Tell the LLM to remember the cheapest plan it made when its newer plan is more expensive.
            elif self.cheapest_plan_cost < total_cost:
                return f"Sorry, the cost of your plan is {total_cost} dollars, which exceeds the targeted sub-budget of " \
                       f"{targeted_budget} dollars, and was more expensive than a cheaper plan that you made previously, " \
                       f"which cost {self.cheapest_plan_cost} dollars. Forget your current plan and remember the " \
                        "cheapest plan that you made, as shown here: \n\n" \
                        \
                        "***CHEAPEST PLAN BEGINS***\n" \
                       f"{str(self.cheapest_plan)}\n" \
                        "***CHEAPEST PLAN ENDS***\n\n" \
                        \
                        "Find either cheaper accomodation or flights to reduce the total " \
                        "cost of the plan. If the cheapest accomodation and flight have been selected, and " \
                        "the cost of the plan still exceeds the budget, start finding cheaper restaurants. " \
                        "When creating your next plan, only change one attribute from your current plan. " \
                        "For instance, if you will change the accomodation for the next plan, do not change " \
                        "anything else, such as the breakfast or the flight."

            # The plan still exceeds the budget, but it is the cheapest so far. Record this in self.cheapest_plan
            else:
                self.cheapest_plan = tested_data
                self.cheapest_plan_cost = total_cost
                return f"Sorry, the cost of your plan is {total_cost} dollars, which exceeds the targeted sub-budget of " \
                       f"{targeted_budget} dollars. Find either cheaper accomodation or flights to reduce the total " \
                        "cost of the plan. If the cheapest accomodation and flight have been selected, and " \
                        "the cost of the plan still exceeds the budget, start finding cheaper restaurants. " \
                        "When creating your next plan, only change one attribute from your current plan. " \
                        "For instance, if you will change the accomodation for the next plan, do not change " \
                        "anything else, such as the breakfast or the flight."
            
        else:
            message = "Sorry, we can't calculate the cost of your plan of the following reasons: "
            for idx, info in enumerate(returned_info):
                message += str(idx + 1) + ". " + info + " " + '\t'
            self.retry_step += 1
            if self.retry_step >= self.max_retry_step:
                self.is_terminated = True
            return message
