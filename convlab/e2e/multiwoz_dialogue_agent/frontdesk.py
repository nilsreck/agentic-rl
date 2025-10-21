from langchain.schema import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from convlab.dialog_agent import Agent
from convlab.e2e.multiwoz_dialogue_agent.agent_graph import get_workflow
from convlab.e2e.multiwoz_dialogue_agent.policy_utils import build_convlab3_empty_state
from convlab.e2e.multiwoz_dialogue_agent.state import AgentState


class DialogueAgent(Agent):
    def __init__(self, name="dialogue_agent"):
        super(DialogueAgent, self).__init__(name=name)
        self.workflow = get_workflow().compile()
        self.conversation_history = []
        self.state = build_convlab3_empty_state()

    def response(self, observation, **kwargs):
        """Generate response using the dialogue agent graph.

        Args:
            observation (str): User utterance

        Returns:
            str: Assistant response
        """
        user_message = HumanMessage(content=observation)
        user_message.pretty_print()

        self.conversation_history.append(user_message)

        state: AgentState = {
            "messages": self.conversation_history.copy(),
            "agent": None,
            "belief_state": self.state,
        }

        config = kwargs.get("config")

        try:
            result = self.workflow.invoke(state, config)

            if result and "messages" in result and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    response_text = last_message.content
                    last_message.pretty_print()
                    self.conversation_history.append(AIMessage(content=response_text))
                    return response_text
                elif hasattr(last_message, "content"):
                    response_text = str(last_message.content)
                    self.conversation_history.append(AIMessage(content=response_text))
                    return response_text
                else:
                    return "I'm sorry, I couldn't process your request."
            else:
                return "I'm sorry, I couldn't generate a response."

        except Exception as e:
            print(f"Error invoking graph: {e}")
            return "I'm sorry, there was an error processing your request."

    def init_session(self, **kwargs):
        """Reset the conversation history for a new session."""
        self.conversation_history = []
        self.belief_state = build_convlab3_empty_state()


if __name__ == "__main__":
    dialogue_agent = DialogueAgent()
    dialogue_agent.init_session()
    response = dialogue_agent.response("Hallo, wie sind Ihre Öffnungszeiten?")
    print(response)
