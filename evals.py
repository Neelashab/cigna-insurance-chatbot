from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval import evaluate
from deepeval.metrics import ConversationalGEval
from test_data import *
from insurance_agent import plan_discovery_node, SessionState


# PLAN DISCOVERY TESTING PIPELINE
def simulate_profile_building_goldens():
    """
    Creates a list of goldens for profile building evaluation.
    Each golden uses the same scenario and profile_eo but different user profiles.
    """
    goldens = []
    
    # Loop through each user profile and create a golden
    for user_profile in profile_users:
        golden = ConversationalGolden(
            scenario=profile_scenario,
            expected_outcome=profile_eo,
            user_description=user_profile
        )
        goldens.append(golden)
    
    return goldens

# Define callback function for plan discovery
async def plan_discovery_callback(input):
    session = SessionState()
    response = plan_discovery_node(input, session)
    return Turn(role="assistant", content=response)


# Generate plan discovery test cases
def gen_plan_discovery_test_cases():
    plan_discovery_simulator = ConversationSimulator(model_callback=plan_discovery_callback)
    plan_discovery_goldens = simulate_profile_building_goldens()
    return plan_discovery_simulator.simulate(conversational_goldens=plan_discovery_goldens)

# Evaluate plan discovery conversations
def evaluate_plan_discovery():
    """
    Evaluates plan discovery test cases using ConversationalGEval metrics
    """
    # Generate test cases
    test_cases = gen_plan_discovery_test_cases()
    
    # Define evaluation metrics
    information_extraction_metric = ConversationalGEval(
        name="Information Extraction",
        criteria="Determine whether the assistant accurately captures business size, location, and coverage preference from the user's responses. The assistant should correctly identify and record these three key pieces of information."
    )
    
    conversational_style_metric = ConversationalGEval(
        name="Conversational Style", 
        criteria="Evaluate whether the assistant uses natural, friendly questioning without revealing rigid data categories. The conversation should feel organic and not like a form-filling exercise."
    )
    
    completeness_metric = ConversationalGEval(
        name="Completeness",
        criteria="Assess whether the assistant successfully collects all required profile data (business size, location, coverage preference) before declaring the information gathering phase is complete."
    )
    
    flow_management_metric = ConversationalGEval(
        name="Flow Management",
        criteria="Evaluate whether the assistant asks one question at a time with logical progression. The conversation should flow naturally from one topic to the next without overwhelming the user."
    )
    
    metrics = [
        information_extraction_metric,
        conversational_style_metric, 
        completeness_metric,
        flow_management_metric
    ]
    
    # Run evaluation
    evaluate(test_cases=test_cases, metrics=metrics)
    
    return test_cases, metrics


if __name__ == "__main__":
    evaluate_plan_discovery()
