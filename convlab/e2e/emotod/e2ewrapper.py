from convlab.dialog_agent import Agent
from convlab.dst.rule.multiwoz import RuleDST


class E2EAgentWrapper(Agent):
    def __init__(self, e2e_model, name, dst=None):
        super().__init__(name=name)
        self.policy = e2e_model
        self.dst = dst
        self.nlg = None

        # create a dummy rule dst and update it with the system state
        if not self.dst:
            self.dst = RuleDST()
        else:
            raise NotImplementedError("Custom DST not supported yet")

    def init_session(self):
        self.policy.init_session()
        self.dst.init_session()

    def response(self, observation):
        self.dst.state["belief_state"] = self.policy.state
        return self.policy.response(observation)

    def get_in_da(self):
        return None

    def get_out_da(self):
        return None

    def state_return(self):
        return {"dst_state": self.policy.state}
