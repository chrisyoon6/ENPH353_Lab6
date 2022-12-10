import random
import pickle
import csv

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        with open(filename, 'rb') as handle:
            self.q = pickle.load(handle)
        # TODO: Implement loading Q values from pickle file.
        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename, 'wb') as handle:
            pickle.dump(self.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Wrote to file: {}".format(filename+".pickle"))

    def saveQcsv(self, filename):
        with open(filename, 'wb') as csv_file:
            wr = csv.writer(csv_file, delimiter = ',')
            for key in self.q:
                state, act = key
                qVal = self.q[key]
                row = state
                row.extend(state)
                row.append(act)
                row.append(qVal)
                wr.writerow(row)

    def loadQcsv(self, filename):
        with open(filename, newline='') as csvfile:
            rdr = csv.reader(csvfile, delimiter=',')
            for row in rdr:
                state = row[:10]
                act = row[10]
                qVal = row[11]
                self.q[(state, act)] = qVal
    
    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time

            If two actions result in the same max Q, returns a random action 
            of the max Qs.
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action
        doRandomAct = random.randint(0, 100) > 100*self.epsilon
        act = None
        q = 0

        if doRandomAct:
            randAct = random.choice(self.actions)
            act = randAct
            q = self.getQ(randAct, state)
        else:
            act, q = self.getMaxQ(state)
        if return_q:
            return (act, q)
        else:
            return act

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation

        Edge case: initializes new state, action combo with 0 q value.
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)
        if (state1, action1) not in self.q:
            self.q[(state1, action1)] = 0
        maxQ2, act2 = self.getMaxQ(state2)        

        self.q[(state1,action1)] += self.alpha*(reward+self.gamma*maxQ2-self.q[(state1,action1)])
    
    def getMaxQ(self, state):
        '''
        @brief
        Returns the max Q value and its action corresponding to the current state.
        Edge: 
            If two actions result in the same max Q, returns a random action 
            of the max Qs.
        '''
        bestQ = self.getQ(state, self.actions[0]) 
        bestActions = [self.actions[0]]
        for action in self.actions:
            q = self.getQ(state, action)
            # print(q)
            if q > bestQ:
                bestQ = q
                bestActions.clear()
                bestActions.append(action)
            elif q == bestQ:
                bestActions.append(action)
        bestAct = random.choice(bestActions)
        return (bestAct, bestQ)