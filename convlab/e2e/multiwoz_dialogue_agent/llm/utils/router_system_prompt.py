ROUTER_SYSTEM_INSTRUCTION_TEMPLATE = """
These are the messages that have been exchanged so far from the user:
<Messages>
{messages}
</Messages>

Assess whether the user query has to be forwarded to the restaurant or hotel agent.
IMPORTANT: It is ABSOLUTELY NECESSARY to foward the query to one of the following two agents:

Here is the list of agents in JSON format, including their descriptions:

```json
{agents}
```

Respond in valid JSON with this exact key:
"agent": Literal["HOTEL", "RESTAURANT"]

If you consider the hotel agent the appropriate agent to answer the user query, return:
"agent": "HOTEL"

If you consider the restaurant agent the appropriate agent to answer the user query, return:
"agent": "RESTAURANT"


To select the most suitable agent, follow these steps:
1. Carefully read and understand the user query.
2. Examine the messages that have been exchanged so far to gather context.
3. Review the list of agents and their descriptions
4. Analyze how well each agent's description align with the user query.
5. Consider factors such as relevance, expertise, and specificity of the agent in relation to the query.
6. Select the agent whose description best aligns with the user's needs.
"""
