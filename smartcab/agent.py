import pdb
import random
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, trial, gamma):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'yellow'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        #Added:
        self.Actions = Environment.valid_actions #valid actions the agent can take
        self.state = None
        self.last_action = None
        self.last_state = None
        self.cumulative_reward = 0 # Sum the total rewards per trial
        self.last_reward = None

        # Q-Learning Parameters:
        self.Q = {} #Q(state,action)
        self.default_Q = 1
        self.epsilon = 0 #exploration prob of making a random move
        self.gamma = gamma #discount rate
        self.t = 1.0 # learning rate 'alpha' declines over time

        # Track number of times agent reaches destination
        self.trials = trial
        self.reach_dest = 0
        self.current_trial = 0

        # Track the number of penalties
        self.penalty = 0
        self.num_moves = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables....
        self.cumulative_reward = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = 0
        self.t = 1.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'],inputs['oncoming'],inputs['left'], inputs['right'], self.next_waypoint)
        #self.state = tuple(sorted(inputs.items())) # store in a tuple

        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions) # random action 
        Q, action = self.Qmax(self.state) # policy 

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward

        # TODO: Learn policy based on state, action, reward
        learning_rate = get_decline_rate(self.t)
        self.epsilon = get_decline_rate(self.t)
        
        if self.last_state != None:
            if (self.last_state, self.last_action) not in self.Q:
                self.Q[(self.last_state, self.last_action)] = 1.0

        ## Updating Qvalues(State,action) 
            self.Q[(self.last_state,self.last_action)] = \
            (1 - learning_rate) * self.Q[(self.last_state,self.last_action)] + \
            learning_rate * (self.last_reward + \
                self.gamma *(self.Qmax(self.state)[0] - self.Q[(self.last_state, self.last_action)]))

        
        # Store prevoious action -> use to update Qtable for next iteration time step
        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward
        self.cumulative_reward =+ reward 
        self.t += 1

        #[debug]
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, Cum. Reward = {}, LR = {}, E = {}".format(deadline, inputs, action, reward, self.cumulative_reward, round(learning_rate,2),round(self.epsilon,2))  

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, Cum. Reward = {}".format(deadline, inputs, action, reward, self.cumulative_reward)  

        """Statistics:"""
        self.num_moves += 1
        if reward < 0:
            self.penalty+= 1

        add_total = False
        if deadline == 0:
            add_total = True

        if reward >= 10: #agent has reached destination 
            self.reach_dest += 1
            add_total = True

        if add_total:
            self.current_trial += 1
            print self.statistics()
        
        self.env.status_text += self.statistics() # Edit status_text

    def statistics(self):
        if self.current_trial == 0:
            success_rate = 0
        else:
            success_rate = "{}/{} = %{}".format(self.reach_dest, self.current_trial, (round(float(self.reach_dest)/float(self.current_trial), 3))*100)
        penalty_ratio = "{}/{} = %{}".format(self.penalty, self.num_moves, (round(float(self.penalty)/float(self.num_moves),4))*100)
        text = "\nSuccess Rate: %s, Penalty Ratio %s \n" % (success_rate,penalty_ratio)
        return text

    ##########################
    """QLearning Functions"""
    ##########################

    def getQValue(self, state, action):
        """Returns Q(state,action)
        return zero if not in self.Q"""
        if (state, action) not in self.Q:
          self.Q[(state, action)] = 0.0
        return self.Q[(state, action)]

    def Qmax(self, state):
        """Greedy Exploration"""
        """Returns: max Q value and best action"""
        if random.random() < self.epsilon:
            best_action = random.choice(self.Actions) #Chooses Random Action
            maxQ = self.getQValue(state, best_action)
        else: #Choose action based on Q Value
            q = [self.getQValue(state, a) for a in self.Actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1: #Randomly choose between Qmax if more than one
                best = [i for i in range(len(self.Actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            best_action = self.Actions[i]
        return (maxQ, best_action)

def get_decline_rate(t):
        return 1.0 / float(t)

def run():
    """Run the agent for a finite number of trials."""

    trial = 100 #number of trials
    gamma = 0.90  #discount rate
    #epsilon = 0.30 #exploration prob
    #alpha = 1.0 #learning rate

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, trial, gamma)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=trial)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()


