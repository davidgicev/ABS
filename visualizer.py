import pickle

from matplotlib import pyplot as plt

file_name = "reward_towards_fruit_multiagent_fixed/results.pickle"

file = open(file_name, 'rb')
results = pickle.load(file)
file.close()

rewards = results
trials = []

buckets = len(results) // 10

for i in range(buckets):
    trials.append(rewards[i*10: (i+1)*10])

plt.plot(range(0, buckets*20, 20), [max(t) for t in trials])
plt.title('Rewards over time')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

pairs = zip(range(0, buckets*20, 20), [max(t) for t in trials])

print(" ".join([str(p) for p in pairs]))
