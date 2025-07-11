
plan_links = [
    "https://www.cigna.com/individuals-families/shop-plans/plans-through-employer/open-access-plus", # OAP
    "https://www.cigna.com/employers/medical-plans/localplus?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # LP
    "https://www.cigna.com/employers/medical-plans/hmo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # HMO
    "https://www.cigna.com/employers/medical-plans/network?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MN
    "https://www.cigna.com/employers/medical-plans/ppo?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # PPO
    "https://www.cigna.com/employers/medical-plans/surefit?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SF
    "https://www.cigna.com/employers/medical-plans/indemnity?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # MI
    "https://www.cigna.com/employers/small-business/small-group-health-insurance-plans?gclid=CjwKCAjw7MLDBhAuEiwAIeXGIYKFxgNbYP3kz7KY-srG0rulaFAcLz9yaLzV5vG7gzf2erJQgCzeaxoCmaQQAvD_BwE", # SG 
    # ^ INCLUDE PDF FOR SG
]































# You look through the plan links
    # ROUND 1 -> tell me what fields are in each plan? what do you feel is relevant to include? 
                # what are the key things that differentiate each plan and would influence an employeers decision when buying it? 

    # ROUND 2 -> go through the material again and fit it into the fields you identified in the first round. then upload these to pinecone

# create master list of all lists 
# upload each of the items in the list to pinecone 


# fully_insured_plan = """

#     # one for each state, size, industry
#     state: str
#     size: Literal["2–99", "100–499", "500–2,999", "3,000+"]
#     industry: Literal[
#         "Hospital and Health Systems", "Higher Education", "K-12 Education",
#         "State and Local Governments", "Taft-Hartley and Federal",
#         "Third Party Administrators (Payer Solutions)"
#     ]

#     plan_type: Literal["OAP", "PPO", "EPO", "LP", "SF", "HMO", "MN", "MI"]
#     pcp_required: bool 
#     pcp_auto_assign: bool 
#     refferal_to_specialist: bool
#     network_type: Literal["National", "Local"]
#     out_of_network_coverage: bool
#     urgent_care_coverage: bool
#     prior_authorization_required: bool
#     self_funded_options: bool
#     open_to_HSA: bool
#     coverage_highlights: str
#     plan_info: str
#     """

# Different Plan Types
# Set rules for how care is accessed 
# Affects cost structure but within a comparable range
"""
OAP -> Open Access Plan (all states and sizes)
PPO -> preferred provider organization (all states and sizes)
HMO -> health maintenance organization (all states and sizes)
EPO -> Exclusive Provider Information -> (all states, all sizes)
MN -> Medical Network (all states, all sizes)
MI -> Medical Indemnity (all states, all sizes)

## randomly generate eligble states
LP -> LocalPLus (some states, all sizes)
SF -> SureFit (some states, all sizes)

SG -> Small Group -> (TN, GA, AZ, 2-50 employees)
"""

# Cost Differences Between Plans 

# Tiers within each plan -> these dominate cost
"""
Tier -> Bronze, Silver, Gold, Platinum
Monthly Premium -> Low, Moderate, High. Very High 
OOP -> Out of Pocket Maximum -> High, Moderate, Low, Very Low
Copays -> None (pay full cost till deductible), Some, Most Services, Low Copay or None
Perks -> Few, Limited extras, Wellness/Telehealth, Most perks
"""

# Industry Specific Info
"""
Specific Group Benefits 
e.g Smart Support Program for specialized customer service for public sector clients
"""

