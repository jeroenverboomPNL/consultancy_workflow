import time

from openai import AzureOpenAI, pydantic_function_tool
import os
import dotenv
import json
from pydantic import BaseModel, Field
from typing import Optional, List, TypedDict, Literal
from json_repair import repair_json
from pydantic_router_experiment import chat_with_assistant

dotenv.load_dotenv()
client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_POSTNL'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY_POSTNL'),
            api_version="2024-10-01-preview",
            timeout=180.0,
        )

def chat_with_assistant(assistant_id: str, chat_history: list) -> str:

    # Create a thread
    thread = client.beta.threads.create()

    # Add previous messages to the thread
    for message in chat_history:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role=message["role"],
            content=message["content"],
        )

    # Run the assistant on the thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        tool_choice="required"
    )

    # Poll for the run status
    while True:

        # Retrieve the run status
        time.sleep(.01)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        # if the run is completed, return the assistant's response
        if run.status == "completed":
            # Retrieve messages and find the assistant's reply
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages:
                if message.role == "assistant":
                    content = message.content[0]
                    if content.type == "text":
                        return content.text.value

        # If the run requires action, it means we've used a tool meaning we asked for a structured output.
        # We can cancel the run and return the structured output
        if run.status == "requires_action":

            # Get the string of the tool output. It should be JSON but I've built in a check to handle non-JSON strings
            json_string = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments

            # If the JSON string is not valid, try to repair it. If it can't be repaired, return an error message
            try:
                json_dict = json.loads(json_string)
            except:
                try:
                    json_dict = repair_json(json_string)
                except Exception as e:
                    ValueError(f"ERROR: Could not repair JSON string. Error: {e}")
                    quit()

            # We canceL the run and return the repaired JSON string
            run = client.beta.threads.runs.cancel(
                thread_id=thread.id,
                run_id=run.id
            )
            return str(json_dict)

        elif run.status in ['cancelling', 'cancelled', 'failed', 'incomplete', 'expired']:
            raise RuntimeError(f"Error: Run failed with status: {run.status}")



chat_history = []

# Add a system message to guide the assistant's behavior and add a user message that contains a task request
# chat_history.append({"role": "assistant", "content": "You are a helpful assistant. You have to choose which task you can be relevant to perform given the user query."})
chat_history.append({"role": "user", "content": """
Teams Call Transcript between Strategy Execution Consultants and TenneT Leadership Team
Participants:
•	Sarah (Lead Consultant)
•	Mark (Junior Consultant)
•	Peter (CEO)
•	Linda (COO)
•	James (CFO)
•	Emma (HR Director)
•	Tom (Head of Operations)
•	Sophie (Head of Communications)
 
[Teams call starts]
Sarah: "Good morning, everyone! Can everyone hear me okay?"
Peter: "Morning, Sarah. Yep, loud and clear."
Linda: "Hi Sarah, hi Mark. Good to see you both."
James: "Morning. Yes, I can hear you."
Emma: "Hello everyone!"
Tom: "Hey folks, just joined. Sorry I'm a minute late; had a slight issue with my connection."
Sophie: "Hi all! Can you see me?"
Mark: "Yes, Sophie, we can see and hear you."
Sarah: "Great, looks like we're all here. First off, thank you all for making the time to join this call. We know everyone's schedules are packed."
Peter: "No problem at all. These things are important."
Linda: "Absolutely. So, what's on the agenda for today?"
Sarah: "We wanted to have an open discussion about how the strategic direction is communicated and reinforced within TenneT. We've prepared some questions to guide the conversation, but we want this to be as interactive as possible."
James: "Sounds good. Hopefully, we can provide some useful insights."
Emma: "And maybe learn a thing or two ourselves."
Sarah: Smiles "That's the spirit! So unless there are any questions, shall we dive right in?"
Tom: "Sure thing."
Mark: "Great. So, our first question is: Can you describe how you communicate the organization's strategic direction to your teams? What methods have you found most effective?"
Peter: "Well, we have our annual strategy meeting where we outline the key objectives for the year. That's been our main avenue."
Linda: "Yeah, and we usually follow that up with an email summary that goes out to all staff."
Sophie: "From a communications perspective, we also post updates on the intranet and occasionally on our internal social media platform."
James: "I tend to include a section on strategy in my quarterly financial reports."
Emma: "We used to have town halls, but attendance was dropping, so we haven't done one in a while."
Tom: "In operations, we have team meetings where we touch on strategic points, but to be honest, the focus is often on immediate issues."
Sarah: "I see. So a mix of meetings, emails, and online platforms. Which of these do you feel resonates most with your teams?"
Linda: "Hard to say. People consume information differently."
Peter: "Exactly. Some prefer reading, others like face-to-face interactions—well, virtual face-to-face these days."
Sophie: "We've noticed that engagement metrics on the intranet aren't very high. Posts about strategy don't get many views."
James: "Emails tend to get buried in inboxes. I'm guilty of that myself."
Emma: "Perhaps we need to explore new channels."
Mark: "That's an interesting point. Do you think the current methods are effective in ensuring everyone understands the strategic direction?"
Tom: "Honestly, I'm not sure. In my team, I get the sense that people are more focused on their day-to-day tasks."
Linda: "Yeah, it's challenging to keep strategy top-of-mind when there are pressing operational issues."
Peter: "We could probably do better, but resources are limited."
Sarah: "Understood. Let's move on to the next question: What routines or practices do you have in place to keep the strategic objectives at the forefront of everyone's daily work?"
Emma: "Well, we have an annual performance review where employees set personal goals."
James: "Do those goals align with the strategic objectives?"
Emma: "They can, but it's up to the managers to make that connection."
Tom: "In operations, we have KPIs, but they're more operational than strategic."
Linda: "We tried implementing monthly strategy updates, but it fell off after a couple of months."
Sophie: "We could consider revitalizing that initiative."
Peter: "But then again, who has the time? Everyone's plates are already full."
Sarah: "I see. So it sounds like there aren't consistent routines in place. Is that fair to say?"
Peter: "Well, we have some, but perhaps not as consistent as they could be."
Mark: "How do you link individual or team goals with the broader strategic direction during performance reviews or meetings?"
Emma: "As I mentioned, we have a section in the performance review forms where employees can mention how their goals align with company strategy."
James: "But is that section mandatory? I think some people skip it."
Emma: "It's encouraged but not enforced."
Linda: "Maybe that's something we need to look at."
Tom: "From my experience, unless it's required, people might not take it seriously."
Peter: "But we don't want to overburden people with paperwork."
Sophie: "Perhaps we could make it more engaging somehow."
Sarah: "Have you considered integrating strategic goals into daily workflows or project management tools?"
Linda: "We talked about that once, but implementing it seemed complicated."
James: "And expensive."
Emma: "Plus, we already have several systems in place. Adding another might confuse people."
Mark: "Understood. Moving on, how do you gauge whether your teams are aligned with the strategic direction? What indicators do you look for?"
Peter: "I like to think that if the company is performing well, then alignment is there."
James: "Financial metrics are my main indicators. If we're hitting our targets, we're aligned."
Tom: "But that doesn't necessarily reflect individual alignment. Some teams might be underperforming while others are overachieving."
Linda: "Good point, Tom."
Emma: "We did an employee engagement survey last year, but the results were... mixed."
Sophie: "And the response rate was low, which makes it hard to draw conclusions."
Sarah: "What were some of the key takeaways from that survey?"
Emma: "Well, some employees mentioned a lack of clarity around the company's strategic goals."
Peter: "I remember that, but we weren't sure how representative it was."
Mark: "Have you taken any steps to address those concerns?"
Linda: "We meant to, but then the pandemic hit, and priorities shifted."
Sarah: "Understandable. Can you share examples of how you've inspired your teams to take ownership of the strategic goals?"
Peter: "Inspiration is a strong word." Laughs slightly
Linda: "Well, I try to motivate my team by setting an example—coming in early, being available."
Tom: "I offer incentives for teams that meet their targets."
James: "Financial incentives?"
Tom: "More like recognition. A shout-out in the team meeting."
Emma: "We used to have 'Employee of the Month,' but it didn't gain much traction."
Sophie: "Maybe we need to revamp our recognition programs."
Peter: "But again, it's about resources and time."
Sarah: "I understand. In what ways do you demonstrate commitment to the strategic direction through your actions and decisions?"
Peter: "I focus on steering the company towards profitability, which is a key strategic goal."
James: "I ensure our budgets reflect our strategic priorities."
Linda: "Operations-wise, we aim for efficiency, which aligns with our strategy."
Tom: "But sometimes, operational demands force us to deviate."
Sophie: "From a communications standpoint, I try to highlight strategic successes in our internal channels."
Emma: "We promote training programs that are supposed to develop skills relevant to our strategy."
Mark: "Supposed to?"
Emma: "Well, attendance is often low, and feedback isn't always positive."
Sarah: "Have you looked into why that might be?"
Emma: "We've gotten some comments that the training isn't engaging or directly applicable to their jobs."
Linda: "Maybe we need to reassess the training content."
Peter: "Again, that's a resource issue."
Sarah: "Circling back, how do you handle challenges when teams might not be fully aligned with the strategy?"
Tom: "We try to address issues as they come up, but it's hard to catch everything."
Linda: "Managers are supposed to handle that at their level."
James: "But we don't always get reports on misalignments unless it's a major issue."
Emma: "Perhaps we need better feedback mechanisms."
Sophie: "Agreed. Open communication channels could help."
Peter: "But we can't force people to speak up."
Sarah: "Have you considered anonymous feedback options?"
Emma: "We have an anonymous suggestion box, but it hasn't been very active."
James: "Probably because it's digital, and people don't trust that it's truly anonymous."
Linda: "Good point."
Mark: "Do you think a culture shift is needed to encourage more openness?"
Peter: "Culture shifts take time and, you guessed it, resources."
Sarah: "Understood. It seems like resource constraints are a recurring theme."
Peter: "Well, in today's economic climate, we have to be prudent."
James: "Every initiative has to prove its ROI."
Tom: "But sometimes investing in alignment and culture pays off in the long run."
Emma: "Exactly. Employee satisfaction can lead to better performance."
Linda: "But we need to balance that with immediate operational needs."
Sarah: "Of course. Shifting gears a bit, have you explored any new methods or technologies to improve strategic communication?"
Sophie: "We've looked into some internal communication platforms, but they come with a hefty price tag."
James: "And there's the issue of adoption. No point investing if people won't use it."
Tom: "Perhaps we could pilot it with a small team first."
Linda: "That's not a bad idea."
Peter: "But who would manage that? Our IT team is stretched thin."
Emma: "Maybe we could outsource it?"
James: "Outsourcing adds cost."
Sarah: "It might be worth calculating the potential benefits versus the costs."
Mark: "Sometimes the initial investment leads to greater efficiency down the line."
Peter: "Perhaps, but we need to see solid numbers before making that decision."
Sarah: "Fair enough. Before we wrap up, is there anything else you'd like to add about how the strategic direction is communicated and reinforced within TenneT?"
Linda: "I think we've covered the main points. We acknowledge there are areas for improvement."
James: "But we have to be realistic about what's feasible."
Emma: "Maybe small incremental changes would be more manageable."
Tom: "Agreed. We don't have to overhaul everything at once."
Sophie: "I'm willing to explore new communication strategies if we can get support."
Peter: "Let's not get ahead of ourselves. We need to prioritize."
Sarah: "Absolutely. Our aim is to help you identify practical steps that can enhance strategic alignment without overextending resources."
Peter: "We appreciate that."
Mark: "Thank you for your time and candidness. Your insights are invaluable for our assessment."
Emma: "Happy to help."
James: "Looking forward to seeing your recommendations."
Linda: "Yes, perhaps you can find the magic solution we've been missing." Smiles wryly
Sarah: Laughs softly "We'll do our best. Just before we conclude, are there any immediate concerns or challenges you'd like us to be aware of?"
Tom: "Well, staffing is always a concern. We're operating with a lean team, and that affects our ability to focus on strategic initiatives."
Emma: "Recruitment has been tough lately. The talent pool seems to be shrinking."
James: "And we have budget constraints that limit new hires."
Linda: "Not to mention, we're facing increased competition in the market."
Peter: "Exactly. We have to be nimble, and sometimes that means deviating from the set strategy."
Mark: "Do you find that the current strategic plan is flexible enough to accommodate these challenges?"
Peter: "It's a good question. The strategy is a guideline, but we need to adapt as circumstances change."
Sarah: "Perhaps building more flexibility into the strategic plan could help."
Sophie: "That could be beneficial, but we'd need to communicate those changes effectively."
Emma: "And ensure that employees understand the reasons behind any shifts."
Linda: "Which brings us back to our communication challenges."
Sarah: "Indeed. It seems like enhancing communication could address multiple issues."
Peter: "Easier said than done."

"""})


# Define pydantic base model for structured output
# class HypothesisEvaluation(BaseModel):
#     reasoning: str
#     conclusion: str
#     score: int

class HypothesisEvaluation(BaseModel):

    reasoning: str

    # Optional field for standardizing text (e.g., replacing specific words or patterns with standard forms)
    score_1: Optional[bool] = Field(
        default=None,
        description=(
            "Minimal or no effort to reinforce the strategic direction. "
            "Leadership actions are either absent or misaligned, causing confusion or a lack of clarity across the organization."
        )
    )
    # Optional field for formatting data (e.g., adjusting case or splitting/merging strings)
    score_2: Optional[bool] = Field(
        default=None,
        description=(
            "Limited and inconsistent efforts to reinforce the strategic direction. "
            "Some actions are aligned but lack follow-through or coherence, leading to fragmented understanding within the organization."
        )
    )
    score_3: Optional[bool] = Field(
        default=None,
        description=(
            "Moderate and consistent reinforcement of the strategic direction. "
            "Leadership actions are aligned and reliable, creating a baseline understanding and buy-in across most parts of the organization."
        )
    )
    score_4: Optional[bool] = Field(
        default=None,
        description=(
            "High degree of consistent reinforcement of the strategic direction. "
            "Leadership demonstrates clear alignment, fostering broad understanding and a sense of urgency to act on the strategy."
        )
    )
    score_5: Optional[bool] = Field(
        default=None,
        description=(
            "Exceptional and innovative reinforcement of the strategic direction. "
            "Leadership culture actively seizes every opportunity to embed the strategy, ensuring the entire workforce deeply understands it. "
            "Creative and impactful approaches are used to build alignment and momentum."
        )
    )
    # Required field for the certainty of task selection, represented as a float (e.g., 0.85 for 85%)
    certainty_of_correct_score_assignment: float

# Define tool object using openai pydanctic_function_tool
tool_obj = pydantic_function_tool(HypothesisEvaluation)

# Create test assistant
assistant = client.beta.assistants.create(
    name="hypothesis_eval_assistant",
    instructions="""
            You are an expert in identifying strategy execution issues.
             You need to test the following hypothesis: 'Hypothesis 1.3: Leadership does not consistently reinforce the strategic direction, leading to confusion or a lack of urgency'.
             Read the transcript and assess how the leadership scores on this hypothesis.
             First, write out your a very brief reasoning, no yapping. Then score the leadership on a likert scale from 1 to 5.
        """,
    model="gpt-4o",
    tools=[tool_obj])

response = chat_with_assistant(assistant_id=assistant.id, chat_history=chat_history)


a=2

