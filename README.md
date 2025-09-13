# VALUE ITERATION ALGORITHM

## AIM
The aim of the experiment is to apply the value iteration algorithm on a custom Frozen Lake environment.

## PROBLEM STATEMENT
The problem statement involves simulating the Frozen Lake environment conditions and deciding the optimal path for the agent to take depending on the process of the value iteration function and form an optimal policy to navigate the presented environment.

## VALUE ITERATION ALGORITHM

# STEP 1: 
Initialize the state value function from the given MDP parameter.

# STEP 2: 
Initialize the action value function and run an iteration for every single state and all their corresponding action

# STEP 3: 
Calculate the the values for action value function for the given state and all of its actions to determine the optimal value

# STEP 4: 
Check it against the threshold and determine if the abs difference is less than theta (the threshold value)

# STEP 5: 
If it meet the criteria then break else keep iterating until we reach an optimal value

## VALUE ITERATION FUNCTION
### Name: Priyadharshan S
### Register Number: 212223240127


```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    # Write your code here
    while True:
      Q = np.zeros((len(P),len(P[0])),dtype=np.float64)
      
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_s,reward,done in P[s][a]:

            Q[s][a]+=prob*(reward+gamma*(V[next_s])*(not done))

      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      
      V=np.max(Q,axis=1)
    
    pi = lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]

    return V, pi
```

## OUTPUT:
<img width="1301" height="847" alt="image" src="https://github.com/user-attachments/assets/0326d28b-65d3-4915-a6ad-0cfc6787ce5b" />

## RESULT:

Thus we have successfully verified the given custom Frozen Lake MDP against a value iteration function and obtained the evaluation of the given environment.
