# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Initialize values for all states to 0
        self.values = util.Counter()

        #For the specified number of iterations
        for i in range(self.iterations):
            #Create a new counter to hold updated values
            updatedValues = util.Counter()
            #Iterate over all states in the MDP
            for state in self.mdp.getStates():
                #If the state is terminal, keep its value as 0
                if not self.mdp.isTerminal(state):
                    #Initialize the comparative Q-value to negative infinity
                    maxQValue = float('-inf')
                    #For each possible action from the current state,
                    for action in self.mdp.getPossibleActions(state):
                        #Compute the Q-value for that action
                        currentQValue = self.computeQValueFromValues(state, action)
                        #Save the greater Q-value
                        maxQValue = max(maxQValue, currentQValue)
                    #Update the new value for the state with the maximum Q-value found
                    updatedValues[state] = maxQValue
            #After checking all states, update self.values with updated values
            self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
            Compute the Q-value of action in state from the
            value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Initialize Q-value to 0
        QValue = 0

        #For each possible next state and its probability from the current state and action
        for Vk, t in self.mdp.getTransitionStatesAndProbs(state, action):
            #Compute the reward for the current state, action, and next state
            r = self.mdp.getReward(state, action, Vk)
            #Update the Q-value using the Bellman equation
            QValue += t * (r + self.discount * self.values[Vk])

        #Return the computed Q-value for the action in the given state
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Initialize variables to keep track of the best action and its maximum Q-value
        maxQValue = float('-inf')
        bestAction = None

        #For each possible action from the current state
        for action in self.mdp.getPossibleActions(state):
            #Compute the Q-value for that action using the current values
            qValue = self.computeQValueFromValues(state, action)

            #If the Q-value is greater than the current maximum
            if qValue > maxQValue:
                #Update the maximum Q-value and the best action
                maxQValue = qValue
                bestAction = action

        #Return the best action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Compute predecessors of all states.
        predecessors = collections.defaultdict(set)
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for Vk, t in self.mdp.getTransitionStatesAndProbs(s, a):
                    if t > 0:
                        predecessors[Vk].add(s)

        #Initialize an empty priority queue.
        priorityQueue = util.PriorityQueue()

        #For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
        for s in self.mdp.getStates():
            #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
            if not self.mdp.isTerminal(s):
                maxQValue = float('-inf')
                for a in self.mdp.getPossibleActions(s):
                    maxQValue = max(maxQValue, self.computeQValueFromValues(s, a))
                diff = abs(self.values[s] - maxQValue)
                #Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                priorityQueue.push(s, -diff)

        #For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            #If the priority queue is empty, then terminate.
            if priorityQueue.isEmpty():
                break

            #Pop a state s off the priority queue.
            s = priorityQueue.pop()

            #Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                maxQValue = float('-inf')
                for a in self.mdp.getPossibleActions(s):
                    maxQValue = max(maxQValue, self.computeQValueFromValues(s, a))
                self.values[s] = maxQValue

            #For each predecessor p of s, do:
            for p in predecessors[s]:
                #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
                maxQValue = float('-inf')
                for a in self.mdp.getPossibleActions(p):
                    maxQValue = max(maxQValue, self.computeQValueFromValues(p, a))
                diff = abs(self.values[p] - maxQValue)

                #If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    priorityQueue.update(p, -diff)