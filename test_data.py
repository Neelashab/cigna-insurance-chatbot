
# SIMULATE CONVERSATIONS FOR
# EVALS FOR BUILDING BUSINESS PROFILE 

profile_scenario = "The user is a business owner and is looking to buy a group insurance plan from Cigna. The LLM chatbot's job is to conversationally collect information from this user and create a business profile which includes the business' size, the state where it is located, and whether they prefer local or national coverage."

profile_eo = "The chatbot should obtain the user's business' size, the state where it is located, and their preference on local or national coverage for their group insurance plan, in a casual conversation. They should collect one piece of information at a time and not reveal rigid data categories. When information is missing, they should ask for it casually and politely. Once all information is provided by the user, this should be recognized by the LLM and the chatbot should indicate that they are done with collecting information and will now begin plan discovery."


profile_users = [
    "User lives in California but his business is in Arizona. He prefers local coverage. His business has 100 people. He does not divulge more than one piece of information at a time.", 
    "User's business is in Massachusetts, he preferes national coverage, and his business is 30 people. He divulges all of this information at once",
    "User divulges at once that he preferes national coverage. His business is in Alaska and is comprised of 15 people, but he is reluctant to divulge this information.",
    "User's first langaguage is not English and their conversation is not syntatically correct and is riddled with spelling mistakes. They do not operate their business from America, they prefer local coverage and have 400 employees. User divulges business information sequentially, one piece of information at a time.",
    "User has a business where all 30 employees are remote and located all over the USA, she is not sure where. She says that they prefer local coverage however. Her business has 2,000 employees.",
    "User's business is just comprised of themself. They are located in Texas and prefer national coverage.",
    "User first claims that their business is in Alabama and they prefer local coverage and that their business is comprised of 200 employees. However they later change their mind, saying their business is 300 people, located in Illinois, and they prefer national coverage.",
    "User claims they do not know where their business is located, whether they prefer local or national coverage, or how many employees they have."
    "User's business has 50 employees, is located in New York, and needs help deciding between local and national coverage."
    "User's business is located in New Jersey, prefers national coverage, and has 1,200 members."
    ""
]