import pdb
import random
import os
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import operator

cur_dir = os.path.dirname(__file__)
path = os.path.abspath(cur_dir)
fullpath = os.path.join(path, "sim-results/q_learn.txt")
os.remove("sim-results/q_learn.txt")
filename = open(fullpath, 'a')

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

        """Q-Learning Parameters:"""
        self.Q = {} # Q(state,action)
        self.epsilon = 0 #exploration prob of making a random move
        self.gamma = gamma #discount rate
        self.time_step = 1.0 # learning rate 'alpha' declines over time

        # Track number of times agent reaches destination
        self.time_steprials = trial
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
        self.time_step = 1.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #self.state = inputs
        #self.state = tuple((inputs['light'],inputs['oncoming'],inputs['left'], inputs['right'], self.next_waypoint))
        
        self.state = inputs
        self.state['next_waypoint'] = self.next_waypoint
        self.state = tuple(sorted(self.state.items()))

        alpha = get_decay_rate(self.time_step)
        self.epsilon = get_decay_rate(self.time_step)

        # TODO: Select action according to your policy
        #action = random.choice(Environment.valid_actions) # random action 
        Q, action = self.Qmax(self.state) # policy 

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        self.time_step += 1

        # TODO: Learn policy based on state, action, reward
        
        if self.last_state != None:
            if (self.last_state, self.last_action) not in self.Q:
                self.Q[(self.last_state, self.last_action)] = 1.0 #Assign 1 if (state,action) pair not in Qvalue
        
        # Updating Qvalues(State,action) 
            self.Q[(self.last_state,self.last_action)] = ((1 - alpha) * self.Q[(self.last_state,self.last_action)]) + alpha * (self.last_reward \
            + self.gamma *(self.Qmax(self.state)[0]))
        
        # Store prevoious action -> use to update Qtable for next iteration time step

        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward

       #Statistics
        self.num_moves += 1
        if reward < 0: #Assign penalty if reward is negative
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

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, E = {}".format(deadline, inputs, action, reward, self.epsilon) #[debug]
            
        # Edit status_text on game screen
        self.env.status_text += self.statistics() 

    ##########################
    """QLearning Functions"""
    ##########################

    def getQValue(self, state, action):
        """Returns Q(state,action)"""
        if (state, action) not in self.Q:
          self.Q[(state, action)] = 1.0 #initialize q-values to 1.0
        return self.Q[(state, action)]

    def Qmax(self, state):
        """Epsilon Greedy Exploration"""
        """Returns: max Q value and best action"""
        best_action = None
        if random.random() < self.epsilon:
            best_action = random.choice(self.Actions) #Chooses Random Action
            maxQ = self.getQValue(state, best_action)
        else: #Choose action based on policy
            maxQ = float('-inf')
            for action in self.Actions:
                q = self.getQValue(state,action)
                if q > maxQ: #if (-inf) q-value
                    maxQ = q
                else:
                    q = [self.getQValue(state, a) for a in self.Actions]
                    maxQ = max(q)
                    count = q.count(maxQ)
                    if count > 1: #if more than 1 Qmax, choose randomly among the Qmaxs
                        best = [i for i in range(len(self.Actions)) if q[i] == maxQ]
                        i = random.choice(best)
                    else:
                        i = q.index(maxQ)
                    best_action = self.Actions[i]
        return (maxQ, best_action)

    def statistics(self):
        if self.current_trial == 0:
            success_rate = 0
        else:
            success_rate = "{}/{} = %{}".format(self.reach_dest, self.current_trial, (round(float(self.reach_dest)/float(self.current_trial), 3))*100)
        penalty_ratio = "{}/{} = %{}".format(self.penalty, self.num_moves, (round(float(self.penalty)/float(self.num_moves),4))*100)
        text = "\nSuccess Rate: %s, Penalty Ratio %s \n" % (success_rate,penalty_ratio)

        suc = str(success_rate)
        pen = str(penalty_ratio)
        tri = str(self.current_trial)

        with open("sim-results/q_learn.txt","a") as myfile:
            myfile.write("Trial: "+ tri + " " + " Success Rate: "+ suc + " " + "Penalty Rate: "+ pen + "\n")

        return text

def get_decay_rate(t): #Decay rate for alpha and epsilon
        return 1.0 / float(t)

def run():
    """Run the agent for a finite number of trials."""

    trial = 1 #number of trials
    gamma = 0.10  #discount rate
    #epsilon = 0.30 #exploration prob
    #alpha = .50 #learning rate

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, trial, gamma)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.00001,display=False)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()