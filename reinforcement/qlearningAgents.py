# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        #Initialize Q-values as a Counter (dictionary-like structure)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #Return the Q-value for the given state-action pair
        if (state, action) in self.qValues:
            return self.qValues[(state, action)]
        else:
            return 0.0


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Initialize legal actions for the given state
        legalActions = self.getLegalActions(state)

        #If there are no legal actions (terminal state), return 0.0
        if not legalActions:
            return 0.0
        
        #Initialize maxQValue to negative infinity to find the maximum Q-value
        maxQValue = float('-inf')
        #Iterate through all legal actions to find the maximum Q-value
        for action in legalActions:
            #Get the Q-value for the current state-action pair
            qValue = self.getQValue(state, action)

            #If the current Q-value is greater than maxQValue
            if qValue > maxQValue:
                #Update maxQValue to the current Q-value
                maxQValue = qValue
        
        #Return the maximum Q-value found among all legal actions
        return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Get the legal actions for the given state
        legalActions = self.getLegalActions(state)

        #If there are no legal actions (terminal state), return none
        if not legalActions:
            return None
        
        #Initialize variables to keep track of the best action and its Q-value
        bestAction = None
        maxQValue = float('-inf')

        #Iterate through all legal actions to find the action with the maximum Q-value
        for action in legalActions:
            #Get the Q-value for the current state-action pair
            qValue = self.getQValue(state, action)

            #If the current Q-value is greater than maxQValue
            if qValue > maxQValue:
                #Update maxQValue to the current Q-value
                maxQValue = qValue
                bestAction = action

        #Return the action with the maximum Q-value
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #Flip a coin to decide whether to explore or exploit
        if util.flipCoin(self.epsilon):
            #Explore: choose a random action from legal actions
            action = random.choice(legalActions)
        else:
            #Exploit: choose the best action based on Q-values
            action = self.computeActionFromQValues(state)
        #Return the chosen action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q-value update formula:
        # Q(s,a) <- (1-alpha) * Q(s,a) + alpha * (reward + discount * max_a' Q(s',a'))

        #Get the current Q-value for the state-action pair
        currentQValue = self.getQValue(state, action)
        #Get the maximum Q-value for the next state over all possible actions
        maxNextQValue = self.computeValueFromQValues(nextState)
        #Update the Q-value for the current state-action pair using the Q-value update formula
        self.qValues[(state, action)] = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.discount * maxNextQValue)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #Get the feature vector for the current state and action
        features = self.featExtractor.getFeatures(state, action)
        #Initialize qValue to 0.0
        qValue = 0.0

        #For each feature and its corresponding value in the feature vector
        for feature, value in features.items():
            #Multiply the feature value by its corresponding weight and add it to qValue
            qValue += self.weights[feature] * value
        
        #Return the computed Q-value for the current state and action
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #Compute the difference between the observed reward and the estimated Q-value
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        #Get the features for the current state and action
        features = self.featExtractor.getFeatures(state, action)

        #Update the weights using the difference and the features
        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass